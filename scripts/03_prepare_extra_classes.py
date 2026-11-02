"""Build Satire and Propaganda examples from TSHP-17 (Rashkin et al., 2017).

LIAR's six truth ratings contain no satire or propaganda signal, so these two
classes need an external source. TSHP-17 labels news articles as
1=Satire, 2=Hoax, 3=Propaganda, 4=Trusted.

To keep the style close to LIAR's short statements, each article is cut to its
leading sentences (<= max_words) so the model cannot cheat on length.

Input : data/raw/tshp17/fulltrain.csv (+ balancedtest.csv), no header: label,text
Output: data/processed/extra_{train,valid,test}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, load_mapping, p, truncate_words, write_json  # noqa: E402


def cap_per_class(d: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in d.groupby("label")]
    return pd.concat(parts) if parts else d


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["tshp_label", "text"], usecols=[0, 1], dtype=str)
    # tolerate a header row or stray lines: keep rows whose label is 1-4
    df["tshp_label"] = pd.to_numeric(df["tshp_label"], errors="coerce")
    df = df[df["tshp_label"].isin([1, 2, 3, 4])].copy()
    df["tshp_label"] = df["tshp_label"].astype(int)
    return df


def main() -> None:
    cfg = load_config()
    ec = cfg["extra_classes"]
    out_dir = cfg["paths"]["processed"]
    if not ec["enabled"]:
        print("[extra] disabled in config; skipping")
        return
    f = p(ec["file"])
    if not f.exists():
        print(f"[extra] {f} not found. Satire/Propaganda will be EMPTY until you add a source.\n"
              "        See docs/label_mapping.md, section 'Getting Satire and Propaganda data'.")
        return

    mapping = {int(k): v for k, v in load_mapping()["tshp17"].items()}
    keep = {1, 3} | ({2, 4} if ec["use_hoax_trusted"] else set())

    df = read(f)
    test_f = p(ec["test_file"])
    df_test = read(test_f) if test_f.exists() else None

    def prep(d: pd.DataFrame) -> pd.DataFrame:
        d = d[d["tshp_label"].isin(keep)].copy()
        d["text"] = d["text"].map(lambda t: truncate_words(t, ec["max_words"]))
        d = d[d["text"].str.split().str.len() >= 5].drop_duplicates(subset=["text"])
        d["label"] = d["tshp_label"].map(mapping)
        d["source"] = "tshp17"
        return d

    df = prep(df)
    # cap per class, reproducibly
    df = cap_per_class(df, ec["max_per_class"], cfg["seed"])

    if df_test is not None:
        test = cap_per_class(prep(df_test), max(1, ec["max_per_class"] // 8), cfg["seed"])
        train, valid = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=cfg["seed"])
    else:
        train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=cfg["seed"])
        valid, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=cfg["seed"])

    stats = {}
    for split, d in [("train", train), ("valid", valid), ("test", test)]:
        d = d.copy()
        d["split"] = split
        d["id"] = [f"tshp-{split}-{i}" for i in range(len(d))]
        d[["id", "text", "label", "source", "split"]].to_csv(p(f"{out_dir}/extra_{split}.csv"), index=False)
        stats[split] = d["label"].value_counts().to_dict()
        print(f"[extra] {split}: {stats[split]}")
    write_json(stats, f"{out_dir}/extra_stats.json")


if __name__ == "__main__":
    main()
