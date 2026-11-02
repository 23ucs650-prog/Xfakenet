"""Week 4: fit the deployable modality models and run the integrated pipeline on example inputs.

1. Fits text-only and image-only logistic models (CLIP features, all 389 FakeNewsNet pairs)
   and the fusion weights used for the TEXT / IMAGE flag; saves runs/modality.joblib.
2. Runs XFakeNetPipeline on 5 LIAR statements (text only) and 6 FakeNewsNet items
   (headline + image; 3 above and 3 below the C-score threshold).

Output: runs/modality.joblib, results/week4/pipeline_demo.json, results/week4/pipeline_demo.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, set_seed, write_json  # noqa: E402


def fit_modality(cfg: dict, df: pd.DataFrame, T: np.ndarray, I: np.ndarray) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (df["fnn_label"] == "fake").astype(int).to_numpy()
    C = cfg["modality"]["C"]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
    # out-of-fold logits for the fusion weights (as in script 14), final models on all data
    folds = list(StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(T, df["site"] + df["fnn_label"]))
    zt, zi = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in folds:
        zt[te] = mk().fit(T[tr], y[tr]).decision_function(T[te])
        zi[te] = mk().fit(I[tr], y[tr]).decision_function(I[te])
    sc = StandardScaler().fit(np.column_stack([zt, zi]))
    fus = LogisticRegression(max_iter=5000).fit(sc.transform(np.column_stack([zt, zi])), y)
    art = {"text": mk().fit(T, y), "image": mk().fit(I, y), "fusion_scaler": sc, "fusion_coef": fus.coef_[0].tolist()}
    joblib.dump(art, p(cfg["pipeline"]["modality_model"]))
    return art


def main() -> None:
    from xfakenet.clip_utils import Clip
    from xfakenet.pipeline import XFakeNetPipeline
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = cfg["week4"]["out_dir"]

    df = pd.read_csv(p(f"{cfg['paths']['processed']}/fnn_pairs.csv"))
    df = df[df["image_path"].map(lambda x: p(x).exists())].reset_index(drop=True)
    T, I = Clip(cfg["cscore"]["clip_model"]).embed(df["title"].tolist(), [p(x) for x in df["image_path"]])
    fit_modality(cfg, df, T, I)
    print("[pipeline] modality models saved", flush=True)

    pipe = XFakeNetPipeline.load()
    test = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_test.csv"))
    rows = []
    sample = pd.concat([g.sample(2, random_state=cfg["seed"]) for _, g in test.groupby("label")]).head(5)
    for r in sample.itertuples(index=False):
        o = pipe(r.text)
        rows.append({"input": "text", "text": r.text, "gold": r.label, **{k: o[k] for k in
                     ("label", "probs", "highlighted", "top_segment", "modality_flag", "cscore",
                      "deception_pattern", "attention_highlighted")}})
    c = 1.0 - (T * I).sum(1)
    above = np.where(c > pipe.thr)[0][:3]
    below = np.where(c <= pipe.thr)[0][:3]
    for k in list(above) + list(below):
        r = df.iloc[k]
        o = pipe(r["title"], image=str(p(r["image_path"])))
        rows.append({"input": "text+image", "text": r["title"], "gold": f"FakeNewsNet {r['fnn_label']}",
                     "image": r["image_path"], **{k2: o[k2] for k2 in
                     ("label", "probs", "highlighted", "top_segment", "modality_flag", "cscore",
                      "deception_pattern")}})
    write_json(rows, f"{out_dir}/pipeline_demo.json")
    L = ["| Input | Text | Gold | Predicted | Highlighted | Modality | C-score | Pattern |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["text"] if len(r["text"]) < 90 else r["text"][:87] + "…"
        L.append(f"| {r['input']} | {t} | {r['gold']} | {r['label']} | {', '.join(r['highlighted'][:4])} | "
                 f"{r['modality_flag']} | {'' if r['cscore'] is None else r['cscore']} | "
                 f"{(r['deception_pattern'] or '').replace('_', ' ')} |")
    open(p(f"{out_dir}/pipeline_demo.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
