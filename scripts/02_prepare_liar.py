"""Load LIAR, clean it, and apply the five-class mapping.

Reads the original TSVs from data/raw/liar/{train,valid,test}.tsv.
(The Hugging Face copy `ucsbnlp/liar` is script-based and no longer loads.)

Output: data/processed/liar_{train,valid,test}.csv with columns
    id, text, liar_label, label, source, split, speaker, party, subject, context
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import clean_text, load_config, load_mapping, p, write_json  # noqa: E402

COLS = ["id", "liar_label", "text", "subject", "speaker", "job", "state", "party",
        "barely_true_ct", "false_ct", "half_true_ct", "mostly_true_ct", "pants_fire_ct", "context"]
SPLITS = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}


def load_tsv(d: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split, fn in SPLITS.items():
        df = pd.read_csv(d / fn, sep="\t", header=None, names=COLS, quoting=3,
                         dtype=str, keep_default_na=False)
        out[split] = df
    return out


def main() -> None:
    cfg, mapping = load_config(), load_mapping()["liar"]
    d = p(cfg["liar"]["dir"])
    if not (d / "train.tsv").exists():
        sys.exit(f"[liar] {d}/train.tsv not found. Download liar_dataset.zip "
                 "(UCSB, or search 'LIAR dataset' on Kaggle) and unzip it into that folder.")
    raw = load_tsv(d)

    stats = {}
    for split, df in raw.items():
        df = df.copy()
        df["liar_label"] = df["liar_label"].str.strip().str.lower()
        unknown = set(df["liar_label"]) - set(mapping)
        if unknown:
            raise ValueError(f"Unmapped LIAR labels in {split}: {unknown}")
        df["text"] = df["text"].map(clean_text)
        n0 = len(df)
        df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
        df["label"] = df["liar_label"].map(mapping)
        df["source"] = "liar"
        df["split"] = split
        keep = ["id", "text", "liar_label", "label", "source", "split",
                "speaker", "party", "subject", "context"]
        df = df[[c for c in keep if c in df.columns]]
        df.to_csv(p(f"{cfg['paths']['processed']}/liar_{split}.csv"), index=False)
        stats[split] = {
            "rows_raw": n0,
            "rows_clean": len(df),
            "liar_labels": df["liar_label"].value_counts().to_dict(),
            "xfakenet_labels": df["label"].value_counts().to_dict(),
            "mean_words": round(df["text"].str.split().str.len().mean(), 1),
        }
        print(f"[liar] {split}: {n0} -> {len(df)} rows | {stats[split]['xfakenet_labels']}")
    write_json(stats, f"{cfg['paths']['processed']}/liar_stats.json")


if __name__ == "__main__":
    main()
