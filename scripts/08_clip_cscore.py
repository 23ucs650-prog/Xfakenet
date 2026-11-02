"""Module 2: CLIP contradiction score (C-score) and threshold calibration.

C-score = 1 - cos(E_text, E_image), CLIP ViT-B/32, headline vs article image.

There are no human labels for "this image contradicts this headline", so the
threshold is calibrated the standard way: real headline+image pairs are the
ALIGNED class, and the same headlines paired with another article's image
(shuffled within the same site, so politics images stay with politics) are the
MISMATCHED class. The threshold is fitted on a validation split and reported
on a held-out test split.

Input : data/processed/fnn_pairs.csv, data/images/
Output: results/cscore/cscore_pairs.csv     (every aligned + mismatched pair with its C-score)
        results/cscore/cscore_results.json  (thresholds, AUC, accuracy, fake-vs-real check)
        results/cscore/cscore_hist.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, set_seed, write_json  # noqa: E402


def derange_within(groups: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """For each row return the index of a DIFFERENT row from the same group."""
    idx = np.arange(len(groups))
    out = np.empty_like(idx)
    for _, g in pd.Series(idx).groupby(groups.values):
        g = g.to_numpy()
        if len(g) < 2:
            out[g] = g  # cannot mismatch a singleton; dropped later
            continue
        out[g] = _derange(g, rng)
    return out


def _derange(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(len(g))
    shifted = np.roll(perm, 1)              # cyclic shift of a random order: no fixed points
    res = np.empty_like(g)
    res[perm] = g[shifted]
    return res


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising TPR - FPR (labels: 1 = mismatched)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(t if np.isfinite(t) else np.median(scores))


def eval_at(scores, labels, t) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    pred = (scores > t).astype(int)
    return {"threshold": round(float(t), 4),
            "accuracy": round(accuracy_score(labels, pred), 4),
            "f1_mismatch": round(f1_score(labels, pred, zero_division=0), 4),
            "precision": round(precision_score(labels, pred, zero_division=0), 4),
            "recall": round(recall_score(labels, pred, zero_division=0), 4),
            "flag_rate_aligned": round(float(pred[labels == 0].mean()), 4),
            "flag_rate_mismatched": round(float(pred[labels == 1].mean()), 4)}


def embed(df: pd.DataFrame, cfg: dict):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    name = cfg["cscore"]["clip_model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name)
    T, I = [], []
    bs = cfg["cscore"]["batch_size"]
    for s in range(0, len(df), bs):
        b = df.iloc[s:s + bs]
        imgs = [Image.open(p(x)).convert("RGB") for x in b["image_path"]]
        inp = proc(text=b["title"].tolist(), images=imgs, return_tensors="pt",
                   padding=True, truncation=True, max_length=77).to(dev)
        with torch.no_grad():
            out = model(**inp)   # forward pass gives projected text_embeds/image_embeds in every version
        T.append(torch.nn.functional.normalize(out.text_embeds, dim=-1).cpu().numpy())
        I.append(torch.nn.functional.normalize(out.image_embeds, dim=-1).cpu().numpy())
        print(f"[cscore] embedded {min(s + bs, len(df))}/{len(df)}", flush=True)
    return np.concatenate(T), np.concatenate(I)


def main() -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    cfg = load_config()
    cc = cfg["cscore"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = cc["out_dir"]

    df = pd.read_csv(p(f"{cfg['paths']['processed']}/fnn_pairs.csv"))
    df = df[df["image_path"].map(lambda x: p(x).exists())].reset_index(drop=True)
    print(f"[cscore] {len(df)} usable pairs | {df.groupby(['site', 'fnn_label']).size().to_dict()}")

    E_t, E_i = embed(df, cfg)
    df["cscore"] = 1.0 - (E_t * E_i).sum(1)

    # mismatched: headline i with image j from the same site (j != i)
    j = derange_within(df["site"], rng)
    keep = j != np.arange(len(df))
    mis = df[keep].copy()
    mis["cscore"] = 1.0 - (E_t[keep] * E_i[j[keep]]).sum(1)
    mis["image_path"] = df["image_path"].to_numpy()[j[keep]]

    df["pair_type"], mis["pair_type"] = "aligned", "mismatched"
    df["is_mismatch"], mis["is_mismatch"] = 0, 1

    # split by HEADLINE so a headline's aligned and mismatched versions stay in the same split
    strat = df["site"] + "_" + df["fnn_label"]
    val_ids, test_ids = train_test_split(df["id"], test_size=0.5, stratify=strat,
                                         random_state=cfg["seed"])
    allp = pd.concat([df, mis], ignore_index=True)
    allp["split"] = np.where(allp["id"].isin(set(val_ids)), "valid", "test")
    allp.to_csv(p(f"{out_dir}/cscore_pairs.csv"), index=False)

    v, t = allp[allp.split == "valid"], allp[allp.split == "test"]
    t_youden = youden_threshold(v["cscore"].to_numpy(), v["is_mismatch"].to_numpy())
    t_pct = float(np.percentile(v.loc[v.is_mismatch == 0, "cscore"], cc["percentile"]))

    def stats(x):
        return {"mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
                "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}

    al = df  # aligned only, for the fake-vs-real check
    res = {
        "clip_model": cc["clip_model"],
        "n_pairs": {"aligned": len(df), "mismatched": len(mis),
                    "valid_headlines": len(val_ids), "test_headlines": len(test_ids)},
        "cscore_distribution": {"aligned": stats(df["cscore"]), "mismatched": stats(mis["cscore"])},
        "test_auc_aligned_vs_mismatched": round(roc_auc_score(t["is_mismatch"], t["cscore"]), 4),
        "thresholds": {
            "youden_on_valid": eval_at(t["cscore"].to_numpy(), t["is_mismatch"].to_numpy(), t_youden),
            f"p{cc['percentile']}_aligned_valid": eval_at(t["cscore"].to_numpy(),
                                                         t["is_mismatch"].to_numpy(), t_pct),
            "proposal_0.62": eval_at(t["cscore"].to_numpy(), t["is_mismatch"].to_numpy(), 0.62),
        },
        "chosen_threshold": round(t_youden, 4),
        "fake_vs_real_aligned": {
            "mean_cscore_fake": round(float(al.loc[al.fnn_label == "fake", "cscore"].mean()), 4),
            "mean_cscore_real": round(float(al.loc[al.fnn_label == "real", "cscore"].mean()), 4),
            "auc_cscore_predicts_fake": round(roc_auc_score((al.fnn_label == "fake").astype(int),
                                                            al["cscore"]), 4),
        },
        "by_site_test_auc": {s: round(roc_auc_score(g["is_mismatch"], g["cscore"]), 4)
                             for s, g in t.groupby("site") if g["is_mismatch"].nunique() == 2},
    }
    write_json(res, f"{out_dir}/cscore_results.json")
    print(pd.Series(res["thresholds"]).to_string())
    print(f"[cscore] test AUC aligned vs mismatched: {res['test_auc_aligned_vs_mismatched']}")
    print(f"[cscore] chosen threshold (Youden, valid): {res['chosen_threshold']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(allp["cscore"].min(), allp["cscore"].max(), 40)
        ax.hist(df["cscore"], bins=bins, alpha=0.6, label="Aligned (real pair)")
        ax.hist(mis["cscore"], bins=bins, alpha=0.6, label="Mismatched (shuffled image)")
        ax.axvline(t_youden, color="k", ls="--", label=f"Threshold {t_youden:.3f}")
        ax.set_xlabel("C-score = 1 - cos(text, image)")
        ax.set_ylabel("Pairs")
        ax.set_title("CLIP ViT-B/32 contradiction score, FakeNewsNet")
        ax.legend()
        fig.tight_layout()
        fig.savefig(p(f"{out_dir}/cscore_hist.png"), dpi=150)
    except Exception as e:  # plotting is a nice-to-have
        print("[cscore] plot skipped:", e)


if __name__ == "__main__":
    main()
