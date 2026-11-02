"""Week 4: are the explanations faithful? (deletion test + exact Shapley/SHAP agreement)

1. Word deletion (comprehensiveness): remove the top-k highlighted words and measure how much
   the probability of the predicted class drops. Compare with removing k RANDOM words
   (average of 3 draws). Faithful highlights should cause a clearly larger drop.
   Conditions: k = 1, k = 3, and "all highlighted words" (importance >= 0.65).
2. Segment deletion (statements with 2+ segments): remove the most-attended segment vs a
   random other segment.
3. SHAP agreement (statements with 2+ segments): exact Shapley values of each segment
   (all 2^K subsets, K <= 8, segment removed = deleted from the text). Report how often the
   Shapley top segment equals the attention top segment, vs the 1/K chance level, and the
   mean Spearman rank correlation for statements with 3+ segments.

Output: results/week4/faithfulness.json, results/week4/faithfulness.md
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402

MODELS = {"XFakeNet v2 (augmented)": "runs/xfakenet_v2", "XFakeNet, no augmentation": "runs/xfakenet_noaug"}


def drop_words(words: list[str], remove: set[int]) -> str:
    return " ".join(w for i, w in enumerate(words) if i not in remove)


def word_deletion(pred, ex: list[dict], thr: float, rng: np.random.Generator, n_random: int = 3) -> dict:
    """Mean drop in predicted-class probability: top-k words vs random-k words."""
    conds = {"k=1": 1, "k=3": 3, "highlighted": None}
    texts, meta = [], []  # meta: (example idx, condition, kind)
    for i, e in enumerate(ex):
        words = [w["w"] for w in e["words"]]
        weights = np.array([w["weight"] for w in e["words"]])
        if len(words) < 4:
            continue
        order = list(np.argsort(-weights))
        for name, k in conds.items():
            top = set(order[:k]) if k else {j for j in range(len(words)) if weights[j] >= thr}
            if not top or len(top) >= len(words):
                continue
            texts.append(drop_words(words, top))
            meta.append((i, name, "top"))
            for _ in range(n_random):
                rnd = set(rng.choice(len(words), len(top), replace=False).tolist())
                texts.append(drop_words(words, rnd))
                meta.append((i, name, "random"))
    probs = pred.predict_proba(texts) if texts else np.zeros((0, 1))
    classes = list(ex[0]["probs"]) if ex else []
    rows = []
    for (i, name, kind), pr in zip(meta, probs):
        c = classes.index(ex[i]["pred"])
        p0 = ex[i]["probs"][ex[i]["pred"]]
        rows.append({"i": i, "cond": name, "kind": kind, "drop": p0 - pr[c], "flip": int(pr.argmax() != c)})
    df = pd.DataFrame(rows)
    out = {}
    for name in conds:
        d = df[df.cond == name]
        if not len(d):
            continue
        top, rnd = d[d.kind == "top"], d[d.kind == "random"]
        out[name] = {"n": int(top.i.nunique()),
                     "drop_top": round(float(top["drop"].mean()), 4),
                     "drop_random": round(float(rnd["drop"].mean()), 4),
                     "flip_rate_top": round(float(top["flip"].mean()), 4),
                     "flip_rate_random": round(float(rnd["flip"].mean()), 4),
                     # paired: per statement, does removing the top words hurt more than random?
                     "top_beats_random_pct": round(float(
                         (top.set_index("i")["drop"] > rnd.groupby("i")["drop"].mean()).mean() * 100), 1)}
    return out


def shapley(values: dict, k: int) -> np.ndarray:
    """Exact Shapley values from v(S) for all subsets S (tuples of segment indices)."""
    phi = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(others, r):
                phi[i] += w * (values[tuple(sorted(S + (i,)))] - values[S])
    return phi


def segment_tests(pred, ex: list[dict], rng: np.random.Generator) -> dict:
    multi = [(i, e) for i, e in enumerate(ex) if 2 <= len(e["segments"]) <= 8]
    classes = list(ex[0]["probs"]) if ex else []
    texts, keys = [], []
    for i, e in multi:
        segs = [s["text"] for s in e["segments"]]
        k = len(segs)
        for r in range(k + 1):
            for S in itertools.combinations(range(k), r):
                texts.append(" ".join(segs[j] for j in S))
                keys.append((i, S))
    probs = pred.predict_proba(texts) if texts else np.zeros((0, 1))
    v = {}
    for (i, S), pr in zip(keys, probs):
        v.setdefault(i, {})[S] = pr[classes.index(ex[i]["pred"])]

    agree, chance, rhos, d_top, d_rnd = [], [], [], [], []
    for i, e in multi:
        k = len(e["segments"])
        att = np.array([s["weight"] for s in e["segments"]])
        phi = shapley(v[i], k)
        agree.append(int(np.argmax(phi) == np.argmax(att)))
        chance.append(1 / k)
        if k >= 3:
            r = spearmanr(att, phi).correlation
            if not np.isnan(r):
                rhos.append(r)
        full = tuple(range(k))
        top = int(np.argmax(att))
        other = int(rng.choice([j for j in range(k) if j != top]))
        d_top.append(v[i][full] - v[i][tuple(j for j in full if j != top)])
        d_rnd.append(v[i][full] - v[i][tuple(j for j in full if j != other)])
    n = len(multi)
    return {
        "n_multi_segment": n,
        "segment_deletion": {"drop_top_segment": round(float(np.mean(d_top)), 4) if n else None,
                             "drop_random_segment": round(float(np.mean(d_rnd)), 4) if n else None},
        "shap_agreement": {"top1_agreement_pct": round(float(np.mean(agree) * 100), 1) if n else None,
                           "chance_pct": round(float(np.mean(chance) * 100), 1) if n else None,
                           "spearman_mean_3plus": round(float(np.mean(rhos)), 3) if rhos else None,
                           "n_3plus": len(rhos)},
    }


def main() -> None:
    from xfakenet.model import Predictor
    cfg = load_config()
    out_dir = cfg["week4"]["out_dir"]
    thr = cfg["xfakenet"]["highlight_threshold"]
    test = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_test.csv"))
    res = {}
    for name, mdir in MODELS.items():
        if not p(mdir).exists():
            continue
        rng = np.random.default_rng(cfg["seed"])
        pred = Predictor(p(mdir))
        ex = pred.explain(test["text"].tolist(), thr)
        res[name] = {"word_deletion": word_deletion(pred, ex, thr, rng), **segment_tests(pred, ex, rng)}
        print(f"[faith] {name}: {res[name]}", flush=True)
    write_json(res, f"{out_dir}/faithfulness.json")

    L = ["| Model | Removed | Drop (top) | Drop (random) | Flip rate top / random | Top hurts more than random |",
         "|---|---|---|---|---|---|"]
    for name, r in res.items():
        for cond, w in r["word_deletion"].items():
            L.append(f"| {name} | {cond} words | {w['drop_top']:.3f} | {w['drop_random']:.3f} | "
                     f"{w['flip_rate_top']*100:.0f}% / {w['flip_rate_random']*100:.0f}% | {w['top_beats_random_pct']}% |")
        sd = r["segment_deletion"]
        if sd["drop_top_segment"] is not None:
            L.append(f"| {name} | top segment | {sd['drop_top_segment']:.3f} | {sd['drop_random_segment']:.3f} | – | – |")
    L += ["", "| Model | Statements (2+ segments) | Attention top segment = SHAP top segment | Chance | "
              "Spearman (3+ segments) |", "|---|---|---|---|---|"]
    for name, r in res.items():
        a = r["shap_agreement"]
        L.append(f"| {name} | {r['n_multi_segment']} | {a['top1_agreement_pct']}% | {a['chance_pct']}% | "
                 f"{a['spearman_mean_3plus']} (n={a['n_3plus']}) |")
    open(p(f"{out_dir}/faithfulness.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
