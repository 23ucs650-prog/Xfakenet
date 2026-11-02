"""Week 4 fix: faithful explanations (SHAP / occlusion) vs the built-in attention, same tests.

For XFakeNet v2 on LIAR test statements:
  * word scores from: attention (built-in), occlusion (leave-one-out), SHAP (permutation Shapley)
  * segment scores from: attention (built-in), exact Shapley over segments
Faithfulness tests (probability of the predicted class):
  * comprehensiveness: remove the top-k words (k = 1, 3) -> how much does it drop? (higher = better)
  * sufficiency: keep ONLY the top-3 words -> how much does it drop? (lower = better)
  * each vs the same number of random words (3 draws); "beats random" = % of statements
  * segments: remove the top segment by each method vs a random segment
Note: occlusion at k = 1 is the comprehensiveness test itself, so that cell is not informative.

GPU: all 1,283 test statements, 10 permutations. CPU only: 400 statements, 6 permutations.

Output: results/week4/faithful_explanations.json / .md, results/week4/explanations_shap.html,
        results/week4/explanations_shap_test.jsonl
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.attribution import explain_batch, highlight  # noqa: E402
from xfakenet.common import load_config, p, write_json  # noqa: E402

METHODS = ["attention", "occlusion", "shap"]


def evaluate(predict, rows: list[dict], classes: list[str], rng: np.random.Generator) -> dict:
    texts, meta = [], []
    for i, r in enumerate(rows):
        w, n = r["words"], len(r["words"])
        if n < 5:
            continue
        for meth in METHODS:
            order = list(np.argsort(-np.asarray(r[meth])))
            for k in (1, 3):
                texts.append(" ".join(x for j, x in enumerate(w) if j not in set(order[:k])))
                meta.append((i, meth, f"remove {k}"))
            texts.append(" ".join(x for j, x in enumerate(w) if j in set(order[:3])))
            meta.append((i, meth, "keep only 3"))
        for _ in range(3):
            for k in (1, 3):
                rnd = set(rng.choice(n, k, replace=False).tolist())
                texts.append(" ".join(x for j, x in enumerate(w) if j not in rnd))
                meta.append((i, "random", f"remove {k}"))
            keep = set(rng.choice(n, 3, replace=False).tolist())
            texts.append(" ".join(x for j, x in enumerate(w) if j in keep))
            meta.append((i, "random", "keep only 3"))
        segs = r["segments"]
        if 2 <= len(segs) <= 8 and r.get("segment_shap") is not None:
            for meth, sc in [("attention", r["seg_attention"]), ("shap", r["segment_shap"])]:
                top = int(np.argmax(sc))
                texts.append(" ".join(s for j, s in enumerate(segs) if j != top))
                meta.append((i, meth, "remove top segment"))
            other = int(rng.integers(len(segs)))
            texts.append(" ".join(s for j, s in enumerate(segs) if j != other))
            meta.append((i, "random", "remove top segment"))
    probs = predict(texts)
    df = pd.DataFrame([{"i": i, "method": m, "test": t,
                        "drop": rows[i]["p0"] - pr[classes.index(rows[i]["pred"])]}
                       for (i, m, t), pr in zip(meta, probs)])
    res = {}
    for test, g in df.groupby("test"):
        rnd = g[g.method == "random"].groupby("i")["drop"].mean()
        res[test] = {}
        for meth in METHODS + ["random"]:
            x = g[g.method == meth]
            if not len(x):
                continue
            per = x.groupby("i")["drop"].mean()
            better = (per < rnd.reindex(per.index)) if test == "keep only 3" else (per > rnd.reindex(per.index))
            res[test][meth] = {"n": int(per.size), "mean_drop": round(float(per.mean()), 4),
                               "beats_random_pct": None if meth == "random" else round(float(better.mean() * 100), 1)}
    return res


def render(rows, classes, thr):
    def line(words, scores, color):
        s = np.asarray(scores, dtype=float)
        top = s.max() if len(s) and s.max() > 0 else 1.0
        out = []
        for w, x in zip(words, s):
            a = max(0.0, x / top)
            cls = "hl" if a >= thr else ""
            out.append(f'<span class="{cls}" style="background:rgba({color},{a*0.55:.2f})">{html.escape(w)}</span>')
        return " ".join(out)
    cards = []
    for r in rows:
        ok = r["pred"] == r["label"]
        cards.append(
            f'<div class="card"><div class="meta">Gold <b>{r["label"]}</b> · Predicted '
            f'<b class="{"ok" if ok else "bad"}">{r["pred"]}</b> ({r["p0"]:.2f})</div>'
            f'<div class="row"><span class="lab">Attention</span><div>{line(r["words"], r["attention"], "120,120,120")}</div></div>'
            f'<div class="row"><span class="lab">SHAP</span><div>{line(r["words"], r["shap"], "217,119,6")}</div></div></div>')
    css = (":root{--bg:#fbfaf7;--ink:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8}*{box-sizing:border-box}"
           "body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}"
           "main{max-width:920px;margin:0 auto;padding:28px 16px 60px}h1{font-size:22px;margin:0 0 4px}p{color:var(--muted)}"
           ".card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 12px}"
           ".meta{font-size:13px;color:var(--muted);margin-bottom:8px}.ok{color:#2f7d4f}.bad{color:#b3261e}"
           ".row{display:grid;grid-template-columns:84px 1fr;gap:10px;margin:4px 0}.lab{font-size:12px;color:var(--muted);padding-top:3px}"
           "span{padding:1px 2px;border-radius:3px}span.hl{outline:1.5px solid #b45309;font-weight:600}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>Attention vs SHAP</title><style>{css}</style></head><body><main>"
            f"<h1>Built-in attention vs SHAP explanations</h1><p>XFakeNet v2 on LIAR test statements. Shading = how much a "
            f"word supports the predicted class; outlined = highlighted (≥ {thr} of the strongest word).</p>"
            f"{''.join(cards)}</main></body></html>")


def main() -> None:
    import torch
    from xfakenet.model import Predictor
    cfg = load_config()
    ec = cfg["explain"]
    out_dir = cfg["week4"]["out_dir"]
    thr = cfg["xfakenet"]["highlight_threshold"]
    gpu = torch.cuda.is_available()
    m = ec["permutations_gpu"] if gpu else ec["permutations_cpu"]
    test = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_test.csv"))
    if not gpu:
        test = test.sample(min(len(test), ec["cpu_statements"]), random_state=cfg["seed"])
    print(f"[faith2] device={'GPU' if gpu else 'CPU'} statements={len(test)} permutations={m}", flush=True)

    pred = Predictor(p("runs/xfakenet_v2"))
    classes = pred.meta["classes"]
    ex = pred.explain(test["text"].tolist(), thr)
    rows = []
    for e, t in zip(ex, test.itertuples(index=False)):
        rows.append({"id": str(t.id), "text": t.text, "label": t.label, "pred": e["pred"],
                     "p0": e["probs"][e["pred"]], "words": [w["w"] for w in e["words"]],
                     "attention": [w["weight"] for w in e["words"]],
                     "segments": [s["text"] for s in e["segments"]],
                     "seg_attention": [s["weight"] for s in e["segments"]]})
    for s in range(0, len(rows), ec["chunk"]):
        part = rows[s:s + ec["chunk"]]
        for r, a in zip(part, explain_batch(pred.predict_proba, part, classes, m=m, seed=cfg["seed"] + s)):
            r["shap"], r["occlusion"] = a.get("shap", [0.0] * len(r["words"])), a.get("occlusion", [0.0] * len(r["words"]))
            r["segment_shap"] = a.get("segment_shap")
        print(f"[faith2] explained {min(s + ec['chunk'], len(rows))}/{len(rows)}", flush=True)

    res = evaluate(pred.predict_proba, rows, classes, np.random.default_rng(cfg["seed"]))
    multi = [r for r in rows if r.get("segment_shap") is not None]
    agree = float(np.mean([np.argmax(r["seg_attention"]) == np.argmax(r["segment_shap"]) for r in multi])) if multi else None
    out = {"device": "GPU" if gpu else "CPU", "statements": len(rows), "permutations": m, "tests": res,
           "attention_vs_shap_top_segment_agreement": None if agree is None else round(agree * 100, 1),
           "mean_highlighted_shap": round(float(np.mean([len(highlight(r["words"], r["shap"], thr)) for r in rows])), 2)}
    write_json(out, f"{out_dir}/faithful_explanations.json")

    names = {"attention": "Attention (built-in)", "occlusion": "Occlusion", "shap": "SHAP", "random": "Random"}
    L = [f"Faithfulness of word and segment explanations, XFakeNet v2 ({len(rows)} LIAR test statements, "
         f"{'GPU' if gpu else 'CPU'}, {m} permutations). Drop = fall in the predicted-class probability.", "",
         "| Test | " + " | ".join(names[k] for k in METHODS + ["random"]) + " |",
         "|---|" + "---|" * (len(METHODS) + 1)]
    for test in ["remove 1", "remove 3", "keep only 3", "remove top segment"]:
        if test not in res:
            continue
        cells = []
        for meth in METHODS + ["random"]:
            r = res[test].get(meth)
            if not r:
                cells.append("–")
                continue
            b = "" if r["beats_random_pct"] is None else f" (beats random {r['beats_random_pct']:.0f}%)"
            cells.append(f"{r['mean_drop']:.3f}{b}")
        lab = {"remove 1": "Remove top word ↑", "remove 3": "Remove top 3 words ↑",
               "keep only 3": "Keep only top 3 words ↓", "remove top segment": "Remove top segment ↑"}[test]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "↑ higher is better, ↓ lower is better. Occlusion at \"remove top word\" is the test itself (circular).",
          f"Attention and SHAP pick the same top segment in {out['attention_vs_shap_top_segment_agreement']}% of "
          f"{len(multi)} multi-segment statements."]
    open(p(f"{out_dir}/faithful_explanations.md"), "w").write("\n".join(L))
    print("\n".join(L))

    with open(p(f"{out_dir}/explanations_shap_test.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rng = np.random.default_rng(cfg["seed"])
    pool = [r for r in rows if 8 <= len(r["words"]) <= 30]
    pick = [pool[i] for i in rng.choice(len(pool), min(25, len(pool)), replace=False)] if pool else []
    open(p(f"{out_dir}/explanations_shap.html"), "w").write(render(pick, classes, thr))


if __name__ == "__main__":
    main()
