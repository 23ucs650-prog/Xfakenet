"""Merge LIAR and the extra-class source into one five-class corpus.

Output: data/processed/corpus_{train,valid,test}.csv  (columns: id,text,label,label_id,source)
        data/processed/corpus_stats.json  (class counts + class weights for the loss)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import class_names, label2id, load_config, p, write_json  # noqa: E402


def main() -> None:
    cfg = load_config()
    d = cfg["paths"]["processed"]
    l2i, classes = label2id(), class_names()
    stats = {}
    for split in ["train", "valid", "test"]:
        parts = [pd.read_csv(p(f"{d}/liar_{split}.csv"))]
        extra = p(f"{d}/extra_{split}.csv")
        if cfg["extra_classes"]["enabled"] and extra.exists():
            parts.append(pd.read_csv(extra))
        df = pd.concat(parts, ignore_index=True)[["id", "text", "label", "source"]]
        df = df[df["label"].isin(classes)]   # drop anything outside the active class list
        df = df.drop_duplicates(subset=["text"])
        df["label_id"] = df["label"].map(l2i)
        df = df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
        df.to_csv(p(f"{d}/corpus_{split}.csv"), index=False)
        counts = {c: int((df["label"] == c).sum()) for c in classes}
        stats[split] = {"n": len(df), "counts": counts,
                        "by_source": df.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()}
        print(f"[corpus] {split}: n={len(df)} {counts}")

    # Inverse-frequency class weights from TRAIN, normalised to mean 1. Absent classes get 0.
    c = np.array([stats["train"]["counts"][k] for k in classes], dtype=float)
    present = c > 0
    w = np.zeros_like(c)
    w[present] = c[present].sum() / (present.sum() * c[present])
    w[present] /= w[present].mean()
    stats["class_weights"] = dict(zip(classes, np.round(w, 4).tolist()))
    missing = [k for k, n in zip(classes, c) if n == 0]
    if missing:
        stats["warning"] = f"No training data for {missing}; model will never predict them."
        print(f"[corpus] WARNING: no training data for {missing}")
    print(f"[corpus] class weights: {stats['class_weights']}")
    write_json(stats, f"{d}/corpus_stats.json")


if __name__ == "__main__":
    main()
