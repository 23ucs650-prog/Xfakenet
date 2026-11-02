"""Build the augmented training corpus: LIAR train + accepted paraphrases + successful TextFooler examples.

Every augmented example keeps the gold label of the statement it came from.
Only TextFooler v2 output (names/numbers protected) is used.

Output: data/processed/corpus_aug_train.csv   (columns: id,text,label,source,label_id,orig_id)
        data/processed/corpus_aug_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402


def jl(path):
    f = p(path)
    return pd.DataFrame([json.loads(line) for line in open(f)]) if f.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-tf", default=None, help="add a second round of TextFooler examples (jsonl)")
    ap.add_argument("--out", default=None, help="output csv (default corpus_aug_train.csv)")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg["paths"]["processed"]
    orig = pd.read_csv(p(f"{d}/corpus_train.csv"))
    orig["source"], orig["orig_id"] = "liar", orig["id"].astype(str)

    para = jl(cfg["paraphrase"]["out_train"])
    para = para[para["status"] == "ok"] if len(para) else para
    tf = jl(cfg["textfooler"]["out_train"])
    if len(tf):
        tf = tf[(tf["status"] == "success") & (tf.get("tf_version", 0) == 2)]

    tf2 = jl(args.extra_tf) if args.extra_tf else pd.DataFrame()
    if len(tf2):
        tf2 = tf2[(tf2["status"] == "success") & (tf2.get("tf_version", 0) == 2)]
    parts = [orig[["id", "text", "label", "label_id", "source", "orig_id"]]]
    for df, src in [(para, "paraphrase"), (tf, "textfooler"), (tf2, "textfooler_r2")]:
        if src == "textfooler_r2" and not args.extra_tf:
            continue
        if len(df):
            x = pd.DataFrame({"id": {"paraphrase": "para", "textfooler": "tf", "textfooler_r2": "tf2"}[src]
                              + "-" + df["id"].astype(str), "text": df["text"],
                              "label": df["label"], "label_id": df["label_id"].astype(int),
                              "source": src, "orig_id": df["id"].astype(str)})
            parts.append(x)
        else:
            print(f"[augment] WARNING: no {src} data found")
    aug = pd.concat(parts, ignore_index=True)
    aug = aug[aug["text"].astype(str).str.strip().str.len() > 0]
    aug = aug.drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    out_csv = args.out or f"{d}/corpus_aug_train.csv"
    aug.to_csv(p(out_csv), index=False)

    n = len(aug)
    stats = {"n": n,
             "by_source": aug["source"].value_counts().to_dict(),
             "share_pct": {k: round(v / n * 100, 1) for k, v in aug["source"].value_counts().items()},
             "by_source_and_label": aug.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()}
    write_json(stats, out_csv.replace(".csv", "_stats.json") if args.out else f"{d}/corpus_aug_stats.json")
    print(f"[augment] {n} training rows | {stats['by_source']} | shares % {stats['share_pct']}")


if __name__ == "__main__":
    main()
