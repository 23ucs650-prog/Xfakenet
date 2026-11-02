"""TextFooler (Jin et al., AAAI 2020, "Is BERT Really Robust?") without the TextAttack dependency.

Algorithm, as in the paper:
1. Rank words by importance: how much the true-class probability drops when the word is deleted.
2. For each word in that order, take its nearest neighbours in counter-fitted word
   embeddings (synonyms, cos >= 0.5).
3. Keep candidates that preserve the part of speech and keep the sentence meaning
   (sentence-embedding cosine >= threshold to the ORIGINAL text).
4. If a candidate flips the prediction, return the one closest in meaning (success).
   Otherwise keep the candidate that lowers the true-class probability most and continue.

Differences from the paper (documented for the report):
- Sentence similarity uses all-MiniLM-L6-v2 instead of the Universal Sentence Encoder.
- Model queries are batched and constraints are checked lazily (only on candidates
  the model output makes useful), which gives the same result order much faster.
- At most `max_change` (default 25%) of the words may be replaced.
- v2 quality filters (added after inspecting v1 output, where 28% of examples swapped a
  person/party/programme name and so changed the factual claim):
    * proper nouns, numbers and acronyms are never replaced (POS tags NNP/NNPS/CD, any
      capitalised word that is not sentence-initial, all-caps tokens);
    * candidates that are WordNet antonyms of the original word are rejected;
    * candidates on an offensive-word blocklist are rejected.

The class only needs four callables, so it can be unit-tested without a GPU.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Sequence

import numpy as np

# Stopwords are never replaced (subset of the list used by TextFooler).
STOP = set("""a about above across after afterwards again against ain all almost alone along already also
although am among amongst an and another any anyhow anyone anything anyway anywhere are aren aren't around as
at back be became because become becomes becoming been before beforehand behind being below beside besides
between beyond both but by can cannot could couldn couldn't d didn didn't doesn doesn't don don't down due during
either else elsewhere empty enough even ever everyone everything everywhere except first for former formerly from
hadn hadn't hasn hasn't haven haven't he hence her here hereafter hereby herein hereupon hers herself him himself
his how however hundred i if in indeed into is isn isn't it it's its itself just latter latterly least ll may me
meanwhile mightn mightn't mine more moreover most mostly must mustn mustn't my myself namely needn needn't neither
never nevertheless next no nobody none noone nor not nothing now nowhere o of off on once one only onto or other
others otherwise our ours ourselves out over per please s same shan shan't she she's should've shouldn shouldn't
somehow something sometime somewhere such t than that that'll the their theirs them themselves then thence there
thereafter thereby therefore therein thereupon these they this those through throughout thru thus to too toward
towards under unless until up upon used ve was wasn wasn't we were weren weren't what whatever when whence
whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole
whom whose why with within without won won't would wouldn wouldn't y yet you you'd you'll you're you've your yours
yourself yourselves says said""".split())

_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$")

# Candidates never used as replacements (offensive words that appear in embedding neighbourhoods).
BLOCKLIST = set("""slut sluts whore whores bitch bitches bastard bastards fuck fucking fucked shit shitty
crap cunt dick dicks cock pussy asshole arse arsehole retard retarded faggot fag nigger nigga
negro spic chink kike wetback tranny dyke hooker prostitute rapist""".split())


def split_tok(tok: str) -> tuple[str, str, str]:
    m = _AFFIX.match(tok)
    return m.group(1), m.group(2), m.group(3)


def match_case(src: str, cand: str) -> str:
    if src.isupper() and len(src) > 1:
        return cand.upper()
    if src[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


class TextFooler:
    def __init__(self,
                 predict: Callable[[Sequence[str]], np.ndarray],          # texts -> probs [n, C]
                 neighbours: Callable[[str], list[str]],                   # word -> synonyms
                 similarity: Callable[[str, Sequence[str]], np.ndarray],   # orig, texts -> cos [n]
                 pos_ok: Callable[[list[str], int, str], bool],            # words, idx, cand -> ok
                 protect: Callable[[list[str]], set] | None = None,        # words -> indices never replaced
                 banned: Callable[[str, str], bool] | None = None,         # (orig word, cand) -> reject?
                 sim_threshold: float = 0.80,
                 max_change: float = 0.25,
                 query_budget: int = 2000,
                 max_checks: int = 8):
        self.predict, self.neighbours = predict, neighbours
        self.similarity, self.pos_ok = similarity, pos_ok
        self.sim_threshold, self.max_change = sim_threshold, max_change
        self.query_budget, self.max_checks = query_budget, max_checks
        self.protect, self.banned = protect, banned

    def _ok(self, orig: str, words: list[str], i: int, cand_core: str, text: str) -> tuple[bool, float]:
        if not self.pos_ok(words, i, cand_core):
            return False, 0.0
        s = float(self.similarity(orig, [text])[0])
        return s >= self.sim_threshold, s

    def attack(self, text: str, label: int) -> dict:
        toks = text.split()
        parts = [split_tok(t) for t in toks]
        cores = [c for _, c, _ in parts]
        p0 = self.predict([text])[0]
        q = 1
        if int(np.argmax(p0)) != label:
            return {"status": "skipped", "queries": q}

        protected = set(self.protect(cores)) if self.protect else set()
        elig = [i for i, c in enumerate(cores)
                if c.isalpha() and len(c) >= 3 and c.lower() not in STOP
                and i not in protected
                and not (i > 0 and c[:1].isupper())        # capitalised mid-sentence: a name
                and not c.isupper()]                       # acronyms (CDC, NATO)
        if not elig:
            return {"status": "failed", "queries": q, "reason": "no_eligible_words"}

        # 1. word importance by deletion
        dels = [" ".join(toks[:i] + toks[i + 1:]) for i in elig]
        pd_ = self.predict(dels)
        q += len(dels)
        imp = p0[label] - pd_[:, label]
        flip = pd_.argmax(1) != label
        imp = imp + flip * (pd_.max(1) - p0[pd_.argmax(1)])   # paper's extra term when deletion flips
        order = [elig[k] for k in np.argsort(-imp)]

        cur = list(toks)
        cur_p = float(p0[label])
        changed, max_changes = 0, max(1, math.ceil(self.max_change * len(toks)))

        for i in order:
            if changed >= max_changes or q >= self.query_budget:
                break
            pre, core, post = parts[i]
            syns = [s for s in self.neighbours(core.lower())
                    if s.lower() != core.lower() and s.lower() not in BLOCKLIST
                    and not (self.banned and self.banned(core.lower(), s.lower()))]
            if not syns:
                continue
            cands = [match_case(core, s) for s in syns]
            texts = []
            for c in cands:
                t = list(cur)
                t[i] = pre + c + post
                texts.append(" ".join(t))
            pr = self.predict(texts)
            q += len(texts)
            words_now = [split_tok(t)[1] for t in cur]

            flipped = np.where(pr.argmax(1) != label)[0]
            if len(flipped):
                sims = self.similarity(text, [texts[k] for k in flipped])
                for k_idx in np.argsort(-sims):
                    k = flipped[k_idx]
                    if sims[k_idx] >= self.sim_threshold and self.pos_ok(words_now, i, cands[k]):
                        out = texts[k]
                        return {"status": "success", "text": out, "queries": q,
                                "words_changed": changed + 1,
                                "similarity": round(float(sims[k_idx]), 4),
                                "new_label_id": int(pr[k].argmax()),
                                "orig_conf": round(float(p0[label]), 4),
                                "adv_conf": round(float(pr[k][label]), 4)}

            # no valid flip: keep the best probability-lowering candidate that passes the constraints
            for k in np.argsort(pr[:, label])[: self.max_checks]:
                if pr[k, label] >= cur_p:
                    break
                ok, _ = self._ok(text, words_now, i, cands[k], texts[k])
                if ok:
                    cur[i] = pre + cands[k] + post
                    cur_p = float(pr[k, label])
                    changed += 1
                    break
        return {"status": "failed", "queries": q, "words_changed": changed}


# ---------- real components (used on Colab / the GPU server) ----------

class CounterFitted:
    """Nearest neighbours in counter-fitted vectors (Mrksic et al., 2016), as used by TextFooler."""

    def __init__(self, path: str, k: int = 50, min_cos: float = 0.5):
        words, vecs = [], []
        opener = open
        if path.endswith(".zip"):
            import io
            import zipfile
            z = zipfile.ZipFile(path)
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            fh = io.TextIOWrapper(z.open(name), encoding="utf-8")
        else:
            fh = opener(path, encoding="utf-8")
        for line in fh:
            w, *v = line.rstrip().split(" ")
            if len(v) < 10:
                continue
            words.append(w)
            vecs.append(np.asarray(v, dtype=np.float32))
        m = np.vstack(vecs)
        self.M = m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-8)
        self.words, self.idx = words, {w: i for i, w in enumerate(words)}
        self.k, self.min_cos, self.cache = k, min_cos, {}

    def __call__(self, w: str) -> list[str]:
        if w in self.cache:
            return self.cache[w]
        i = self.idx.get(w)
        if i is None:
            self.cache[w] = []
            return []
        sims = self.M @ self.M[i]
        kk = min(self.k + 1, len(sims) - 1)
        top = np.argpartition(-sims, kk)[: kk + 1]
        top = top[np.argsort(-sims[top])]
        res = [self.words[j] for j in top if j != i and sims[j] >= self.min_cos][: self.k]
        self.cache[w] = res
        return res


class WordNetSyn:
    """Fallback synonym source if the counter-fitted vectors cannot be downloaded."""

    def __init__(self, k: int = 50):
        from nltk.corpus import wordnet as wn
        self.wn, self.k, self.cache = wn, k, {}

    def __call__(self, w: str) -> list[str]:
        if w not in self.cache:
            out = []
            for s in self.wn.synsets(w):
                for l in s.lemma_names():
                    l = l.replace("_", " ")
                    if " " not in l and l.lower() != w and l not in out:
                        out.append(l)
            self.cache[w] = out[: self.k]
        return self.cache[w]


def make_pos_ok(window: int = 3):
    import nltk

    def coarse(tag: str) -> str:
        return tag[0] if tag[0] in "NVJR" else tag

    def pos_ok(words: list[str], i: int, cand: str) -> bool:
        lo, hi = max(0, i - window), min(len(words), i + window + 1)
        a = [w if w else "," for w in words[lo:hi]]
        b = list(a)
        b[i - lo] = cand
        ta, tb = nltk.pos_tag(a), nltk.pos_tag(b)
        return coarse(ta[i - lo][1]) == coarse(tb[i - lo][1])

    return pos_ok


def make_protect():
    """Indices of proper nouns and numbers (never replaced)."""
    import nltk

    def protect(words: list[str]) -> set:
        tags = nltk.pos_tag([w if w else "," for w in words])
        return {i for i, (_, t) in enumerate(tags) if t in ("NNP", "NNPS", "CD")}

    return protect


def make_banned():
    """Reject WordNet antonyms of the original word (e.g. died -> lives)."""
    from nltk.corpus import wordnet as wn
    cache: dict[str, set] = {}

    def antonyms(w: str) -> set:
        if w not in cache:
            out = set()
            for syn in wn.synsets(w):
                for lem in syn.lemmas():
                    for a in lem.antonyms():
                        out.add(a.name().lower())
                        for rel in a.derivationally_related_forms():
                            out.add(rel.name().lower())
            cache[w] = out
        return cache[w]

    def banned(orig: str, cand: str) -> bool:
        if cand in antonyms(orig) or orig in antonyms(cand):
            return True
        # also compare base forms (died/die vs lives/live)
        base_o, base_c = wn.morphy(orig) or orig, wn.morphy(cand) or cand
        return base_c in antonyms(base_o) or base_o in antonyms(base_c)

    return banned


def make_similarity(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)
    cache: dict[str, np.ndarray] = {}

    def sim(orig: str, texts: Sequence[str]) -> np.ndarray:
        if orig not in cache:
            if len(cache) > 50:
                cache.clear()
            cache[orig] = st.encode([orig], normalize_embeddings=True)[0]
        e = st.encode(list(texts), normalize_embeddings=True, batch_size=64)
        return e @ cache[orig]

    return sim


def make_predict(model_dir: str, max_len: int = 64, batch_size: int = 64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()

    @torch.no_grad()
    def predict(texts: Sequence[str]) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch_size):
            enc = tok(list(texts[s:s + batch_size]), truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    return predict
