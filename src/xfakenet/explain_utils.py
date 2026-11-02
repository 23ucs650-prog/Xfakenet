"""Word- and segment-level explanation from attention weights (no torch needed)."""
from __future__ import annotations

from bisect import bisect_right

import numpy as np


def word_explanation(text, offsets, seg_ids, tok_att, seg_att, spans, probs, classes, thr) -> dict:
    """Turn token/segment attention into word-level and segment-level explanations (pure numpy)."""
    words, w_imp, pos = [], [], 0
    for w in text.split():
        a = text.index(w, pos)
        words.append((a, a + len(w), w))
        w_imp.append(0.0)
        pos = a + len(w)
    starts = [a for a, _, _ in words]
    for j, (a, z) in enumerate(offsets):
        if z > a and seg_ids[j] >= 0 and words:
            k = max(0, bisect_right(starts, a) - 1)
            w_imp[k] += float(tok_att[j])      # a word's importance = sum over its sub-word tokens
    m = max(w_imp) if w_imp and max(w_imp) > 0 else 1.0
    norm = [x / m for x in w_imp]
    segs = [{"text": text[a:z], "weight": round(float(seg_att[k]), 4)}
            for k, (a, z) in enumerate(spans) if k < len(seg_att)]
    top = int(np.argmax([s["weight"] for s in segs])) if segs else 0
    return {
        "text": text,
        "pred": classes[int(np.argmax(probs))],
        "probs": {c: round(float(p_), 4) for c, p_ in zip(classes, probs)},
        "words": [{"w": w, "weight": round(x, 4)} for (_, _, w), x in zip(words, norm)],
        "highlighted": [w for (_, _, w), x in zip(words, norm) if x >= thr],
        "segments": segs,
        "top_segment": segs[top]["text"] if segs else text,
    }
