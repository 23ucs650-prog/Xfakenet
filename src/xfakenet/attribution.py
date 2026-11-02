"""Faithful (perturbation-based) explanations for any text classifier.

Week 4 showed that the built-in attention weights are not faithful: removing the
highlighted words hurts the prediction barely more than removing random words, and
the attention top segment agrees with SHAP only at chance level. So the explanations
shown to the user are now computed from the model's actual behaviour:

- word_shap     : Shapley values of each word, estimated by permutation sampling
                  (Strumbelj & Kononenko 2014; the estimator behind SHAP's sampling explainer).
                  A word's value = its average change in the predicted-class probability when
                  added to random subsets of the other words.
- occlusion     : leave-one-out: drop in predicted-class probability when the word is deleted.
- segment_shap  : exact Shapley values over segments (all 2^K subsets, K <= 8).

All functions batch every text they need into ONE call to `predict_proba`, so they are
fast on a GPU. Removed words/segments are deleted from the text.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

import numpy as np

Predict = Callable[[Sequence[str]], np.ndarray]


def _join(words: list[str], keep) -> str:
    return " ".join(w for i, w in enumerate(words) if i in keep)


def plan_word_shap(words: list[str], m: int, rng: np.random.Generator):
    """Texts needed for permutation Shapley + bookkeeping to turn their scores into values."""
    n = len(words)
    texts, steps = [""], []          # texts[0] = empty statement (shared start of every chain)
    for _ in range(m):
        perm = rng.permutation(n)
        prev, keep = 0, set()
        for j in perm:
            keep.add(int(j))
            texts.append(_join(words, keep))
            steps.append((int(j), prev, len(texts) - 1))   # value(j) += p[cur] - p[prev]
            prev = len(texts) - 1
    return texts, steps


def finish_word_shap(n: int, m: int, steps, p: np.ndarray) -> np.ndarray:
    phi = np.zeros(n)
    for j, a, b in steps:
        phi[j] += p[b] - p[a]
    return phi / max(1, m)


def plan_occlusion(words: list[str]):
    return [_join(words, set(range(len(words))) - {i}) for i in range(len(words))]


def plan_segment_shap(segs: list[str]):
    k = len(segs)
    subsets = [S for r in range(k + 1) for S in itertools.combinations(range(k), r)]
    return [" ".join(segs[j] for j in S) for S in subsets], subsets


def finish_segment_shap(k: int, subsets, p: np.ndarray) -> np.ndarray:
    v = dict(zip(subsets, p))
    phi = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(others, r):
                phi[i] += w * (v[tuple(sorted(S + (i,)))] - v[S])
    return phi


def explain_batch(predict: Predict, items: list[dict], classes: list[str], m: int = 8,
                  seed: int = 42, methods=("shap", "occlusion", "segment")) -> list[dict]:
    """items: [{"words": [...], "segments": [...], "pred": class name}] -> adds scores per method.

    One predict_proba call for everything (chunk `items` yourself to bound memory)."""
    rng = np.random.default_rng(seed)
    all_texts, plans = [], []
    for it in items:
        words, plan = it["words"], {}
        if "shap" in methods and words:
            t, steps = plan_word_shap(words, m, rng)
            plan["shap"] = (len(all_texts), len(t), steps)
            all_texts += t
        if "occlusion" in methods and words:
            t = plan_occlusion(words)
            plan["occ"] = (len(all_texts), len(t))
            all_texts += t + [" ".join(words)]
        if "segment" in methods and 2 <= len(it["segments"]) <= 8:
            t, subsets = plan_segment_shap(it["segments"])
            plan["seg"] = (len(all_texts), len(t), subsets)
            all_texts += t
        plans.append(plan)
    probs = predict(all_texts) if all_texts else np.zeros((0, len(classes)))
    out = []
    for it, plan in zip(items, plans):
        c = classes.index(it["pred"])
        r = {}
        if "shap" in plan:
            s, n, steps = plan["shap"]
            r["shap"] = finish_word_shap(len(it["words"]), m, steps, probs[s:s + n, c]).tolist()
        if "occ" in plan:
            s, n = plan["occ"]
            full = probs[s + n, c]
            r["occlusion"] = (full - probs[s:s + n, c]).tolist()
        if "seg" in plan:
            s, n, subsets = plan["seg"]
            r["segment_shap"] = finish_segment_shap(len(it["segments"]), subsets, probs[s:s + n, c]).tolist()
        out.append(r)
    return out


def highlight(words: list[str], scores: Sequence[float], thr: float = 0.65) -> list[str]:
    """Words whose positive contribution is >= thr x the largest positive contribution."""
    s = np.asarray(scores, dtype=float)
    top = s.max() if len(s) else 0.0
    if top <= 0:
        return []
    return [w for w, x in zip(words, s) if x > 0 and x / top >= thr]
