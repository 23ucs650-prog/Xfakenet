"""Module 1: export token- and segment-level explanations from XFakeNet v2 on the LIAR test set.

For each statement: predicted class + probabilities, every word's importance (0-1,
normalised to the most important word), the highlighted words (importance >= 0.65,
as in the outline), every segment's attention weight, and the most suspicious segment.

Output: results/week3/explanations_test.jsonl
        results/week3/explanations.html      (40 examples, colour-coded, for the report)
        results/week3/explanation_stats.json
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402

CSS = """
:root{--bg:#fbfaf7;--ink:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--hi:217,119,6;--ok:#2f7d4f;--bad:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;margin:0 0 6px}p.lead{color:var(--muted);margin:0 0 24px}
.card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 14px;background:#fff}
.meta{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:13px;color:var(--muted);margin-bottom:10px}
.tag{font-weight:600}.ok{color:var(--ok)}.bad{color:var(--bad)}
.text span{padding:1px 2px;border-radius:3px}.text span.hl{outline:1.5px solid rgb(var(--hi));font-weight:600}
.segs{margin-top:10px;font-size:13px}.seg{display:grid;grid-template-columns:70px 1fr;gap:10px;align-items:center;margin:3px 0}
.bar{height:8px;background:#eee;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:rgb(var(--hi))}
.legend{font-size:13px;color:var(--muted);margin-bottom:18px}
"""


def render(rows: list[dict], title: str, thr: float) -> str:
    cards = []
    for r in rows:
        words = " ".join(
            f'<span class="{"hl" if w["weight"] >= thr else ""}" '
            f'style="background:rgba(217,119,6,{min(1, w["weight"]) * 0.55:.2f})" '
            f'title="{w["weight"]:.2f}">{html.escape(w["w"])}</span>' for w in r["words"])
        segs = "".join(
            f'<div class="seg"><div>{s["weight"]:.2f}</div><div><div class="bar"><i style="width:{s["weight"]*100:.0f}%">'
            f'</i></div><div>{html.escape(s["text"])}</div></div></div>' for s in r["segments"])
        ok = r["pred"] == r["label"]
        probs = ", ".join(f"{k} {v:.2f}" for k, v in r["probs"].items())
        cards.append(f'<div class="card"><div class="meta"><span>Gold: <span class="tag">{r["label"]}</span></span>'
                     f'<span>Predicted: <span class="tag {"ok" if ok else "bad"}">{r["pred"]}</span></span>'
                     f'<span>{probs}</span></div><div class="text">{words}</div>'
                     f'<div class="segs">{segs}</div></div>')
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>XFakeNet explanations</title><style>{CSS}</style>"
            f"</head><body><main><h1>{html.escape(title)}</h1><p class='lead'>Token and segment attention from "
            f"XFakeNet v2 on LIAR test statements.</p><p class='legend'>Shading = word importance (0–1). Outlined "
            f"words are highlighted (importance ≥ {thr}). Bars = segment attention; the longest bar is the most "
            f"suspicious segment.</p>{''.join(cards)}</main></body></html>")


def main() -> None:
    from xfakenet.model import Predictor
    cfg = load_config()
    out_dir = cfg["week3"]["out_dir"]
    thr = cfg["xfakenet"]["highlight_threshold"]
    test = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_test.csv"))
    pred = Predictor(p("runs/xfakenet_v2"))
    ex = pred.explain(test["text"].tolist(), thr)
    for e, row in zip(ex, test.itertuples(index=False)):
        e["id"], e["label"] = str(row.id), row.label
    with open(p(f"{out_dir}/explanations_test.jsonl"), "w") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")

    nseg = np.array([len(e["segments"]) for e in ex])
    nhl = np.array([len(e["highlighted"]) for e in ex])
    nwords = np.array([len(e["words"]) for e in ex])
    multi = [e for e in ex if len(e["segments"]) > 1]
    top_w = np.array([max(s["weight"] for s in e["segments"]) for e in multi]) if multi else np.array([0])
    stats = {"n": len(ex), "accuracy": round(float(np.mean([e["pred"] == e["label"] for e in ex])), 4),
             "segments_mean": round(float(nseg.mean()), 2),
             "multi_segment_pct": round(float((nseg > 1).mean() * 100), 1),
             "top_segment_weight_mean_multi": round(float(top_w.mean()), 3),
             "highlighted_words_mean": round(float(nhl.mean()), 2),
             "highlighted_pct_of_words": round(float(nhl.sum() / nwords.sum() * 100), 1),
             "threshold": thr}
    write_json(stats, f"{out_dir}/explanation_stats.json")
    print(f"[explain] {stats}")

    rng = np.random.default_rng(cfg["seed"])
    pool_multi = [e for e in multi if 10 <= len(e["words"]) <= 40]
    right = [e for e in pool_multi if e["pred"] == e["label"]]
    wrong = [e for e in pool_multi if e["pred"] != e["label"]]
    pick = lambda xs, k: [xs[i] for i in rng.choice(len(xs), min(k, len(xs)), replace=False)] if xs else []  # noqa
    sample = pick(right, 25) + pick(wrong, 15)
    open(p(f"{out_dir}/explanations.html"), "w").write(render(sample, "XFakeNet – explanation examples", thr))
    print(f"[explain] wrote explanations.html with {len(sample)} examples")


if __name__ == "__main__":
    main()
