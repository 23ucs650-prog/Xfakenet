"""Split a statement into segments (sentences and clauses) for the sentence-attention layer.

About 81% of LIAR statements are a single sentence, so splitting only at sentence
boundaries would give the attention layer nothing to choose between. We also split
at ; : commas and dashes, then merge pieces shorter than `min_words` into their
neighbour so a segment is always a meaningful clause.
"""
from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?;:])\s+|,\s+|\s+--?\s+")


def segment_spans(text: str, max_segments: int = 8, min_words: int = 3) -> list[tuple[int, int]]:
    """Return character spans (start, end) of the segments of `text`."""
    spans, start = [], 0
    for m in _SPLIT.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        return [(0, len(text))]

    words = lambda s: len(text[s[0]:s[1]].split())  # noqa: E731
    merged: list[tuple[int, int]] = []
    for s in spans:
        if merged and (words(s) < min_words or words(merged[-1]) < min_words):
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    if len(merged) > max_segments:
        merged = merged[: max_segments - 1] + [(merged[max_segments - 1][0], merged[-1][1])]
    return merged


def segments(text: str, max_segments: int = 8, min_words: int = 3) -> list[str]:
    return [text[a:b] for a, b in segment_spans(text, max_segments, min_words)]
