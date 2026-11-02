"""Module 1 (modality level): flag each FakeNewsNet item as TEXT, IMAGE or CROSS-MODAL.

Outline rule: CROSS-MODAL when the C-score exceeds its calibrated threshold; otherwise
TEXT or IMAGE, depending on which modality carries more of the evidence.

Evidence per modality comes from two small fake-vs-real classifiers on FakeNewsNet:
  text  : logistic regression on the CLIP embedding of the headline
  image : logistic regression on the CLIP embedding of the image
A fusion model combines their (out-of-fold) logits; each modality's contribution is
coefficient x its logit. The bigger absolute contribution decides TEXT vs IMAGE.

All probabilities are out-of-fold (5-fold CV), so no item is scored by a model that saw it.
This also gives the C-score ablation on FakeNewsNet: fake/real AUC with and without it.

Output: results/week3/modality_flags.csv, results/week3/modality_results.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, set_seed, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent


def clip_embed(df: pd.DataFrame, cfg: dict):
    spec = importlib.util.spec_from_file_location("cscore", ROOT / "08_clip_cscore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.embed(df, cfg)


def oof_logit(X: np.ndarray, y: np.ndarray, folds, C: float = 1.0) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        m.fit(X[tr], y[tr])
        out[te] = m.decision_function(X[te])
    return out


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = cfg["week3"]["out_dir"]
    mc = cfg["modality"]

    df = pd.read_csv(p(f"{cfg['paths']['processed']}/fnn_pairs.csv"))
    df = df[df["image_path"].map(lambda x: p(x).exists())].reset_index(drop=True)
    thr = json.load(open(p(f"{cfg['cscore']['out_dir']}/cscore_results.json")))["chosen_threshold"]
    T, I = clip_embed(df, cfg)
    df["cscore"] = 1.0 - (T * I).sum(1)
    y = (df["fnn_label"] == "fake").astype(int).to_numpy()

    strat = df["site"] + "_" + df["fnn_label"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(df, strat))
    zt = oof_logit(T, y, folds, mc["C"])
    zi = oof_logit(I, y, folds, mc["C"])
    zc = df["cscore"].to_numpy()

    def fused(cols):
        X = np.column_stack(cols)
        return oof_logit(X, y, folds, 1.0)

    auc = lambda z: round(float(roc_auc_score(y, z)), 4)  # noqa: E731
    res_auc = {"text_only": auc(zt), "image_only": auc(zi), "cscore_only": auc(zc),
               "text+image": auc(fused([zt, zi])), "text+image+cscore": auc(fused([zt, zi, zc]))}

    # contributions for the flag: fusion model fit on all out-of-fold logits
    fz = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(np.column_stack([zt, zi]), y)
    sc, lr = fz.named_steps["standardscaler"], fz.named_steps["logisticregression"]
    Z = sc.transform(np.column_stack([zt, zi]))
    contrib_text, contrib_img = lr.coef_[0][0] * Z[:, 0], lr.coef_[0][1] * Z[:, 1]
    df["p_fake"] = fz.predict_proba(np.column_stack([zt, zi]))[:, 1]
    df["contrib_text"], df["contrib_image"] = contrib_text, contrib_img
    df["modality_flag"] = np.where(df["cscore"] > thr, "CROSS-MODAL",
                                   np.where(np.abs(contrib_text) >= np.abs(contrib_img), "TEXT", "IMAGE"))
    df[["id", "title", "fnn_label", "site", "image_path", "cscore", "p_fake", "contrib_text",
        "contrib_image", "modality_flag"]].to_csv(p(f"{out_dir}/modality_flags.csv"), index=False)

    res = {"n": len(df), "cscore_threshold": thr,
           "fake_vs_real_auc_5fold": res_auc,
           "flag_counts": df["modality_flag"].value_counts().to_dict(),
           "flag_by_label": df.groupby(["fnn_label", "modality_flag"]).size().unstack(fill_value=0).to_dict(),
           "fusion_weights": {"text": round(float(lr.coef_[0][0]), 4), "image": round(float(lr.coef_[0][1]), 4)}}
    write_json(res, f"{out_dir}/modality_results.json")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
