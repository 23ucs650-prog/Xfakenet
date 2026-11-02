"""Generate TextFooler adversarial examples against the week-1 baseline.

Uses src/xfakenet/textfooler.py (the Jin et al. algorithm, no TextAttack needed).

--split train : attacks TRAIN statements until the number of SUCCESSFUL examples
                reaches 15% of the final corpus (70/15/15), with per-class quotas.
                These are the augmentation data for week 3 (gold label kept).
--split test  : attacks a fixed stratified sample of the TEST split. The attack
                success rate is the robustness number for week 4. Never trained on.

Resumable: ids already in the output file are skipped and existing successes count.

Usage: python scripts/07_textfooler_generate.py --split train
       python scripts/07_textfooler_generate.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, set_seed  # noqa: E402
from xfakenet import textfooler as tf  # noqa: E402

TF_VERSION = 2   # v2: names/numbers protected, antonyms + offensive words blocked


def synonym_source(cfg: dict):
    import zipfile
    tc = cfg["textfooler"]
    path = p(tc["embeddings_path"])
    urls = [tc["embeddings_url"],
            "https://media.githubusercontent.com/media/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip"]
    for url in urls:
        if path.exists() and zipfile.is_zipfile(path):
            break
        if path.exists():
            os.remove(path)
        print(f"[textfooler] downloading counter-fitted vectors from {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"[textfooler] download failed ({e})", flush=True)
    if path.exists() and not zipfile.is_zipfile(path):
        os.remove(path)
        print("[textfooler] download was not a valid zip; using WordNet synonyms instead", flush=True)
    if path.exists():
        print("[textfooler] synonyms: counter-fitted vectors", flush=True)
        return tf.CounterFitted(str(path), k=tc["k"], min_cos=0.5), "counter-fitted"
    import nltk
    nltk.download("wordnet", quiet=True)
    return tf.WordNetSyn(k=tc["k"]), "wordnet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--victim", default=None)
    ap.add_argument("--victim-kind", default="hf", choices=["hf", "xfakenet"],
                    help="hf = Hugging Face classifier (baseline); xfakenet = XFakeNet model (week 3)")
    ap.add_argument("--out", default=None, help="override output file")
    args = ap.parse_args()

    cfg = load_config()
    tc = cfg["textfooler"]
    set_seed(cfg["seed"])

    import nltk
    for r in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "wordnet"]:
        nltk.download(r, quiet=True)

    syn, syn_name = synonym_source(cfg)
    victim = str(p(args.victim or tc["victim_dir"]))
    if args.victim_kind == "xfakenet":
        from xfakenet.model import Predictor
        predict_fn = Predictor(victim).predict_proba
    else:
        predict_fn = tf.make_predict(victim, cfg["baseline"]["max_len"])
    attacker = tf.TextFooler(
        predict=predict_fn,
        neighbours=syn, similarity=tf.make_similarity(tc["sim_model"]), pos_ok=tf.make_pos_ok(),
        protect=tf.make_protect(), banned=tf.make_banned(),
        sim_threshold=tc["sim_threshold"], max_change=tc["max_change"], query_budget=tc["query_budget"])

    df = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_{args.split}.csv"))
    df["id"] = df["id"].astype(str)
    out = p(args.out) if args.out else p(tc["out_train"] if args.split == "train" else tc["out_test"])
    prev = [json.loads(line) for line in open(out)] if out.exists() else []
    if prev and prev[0].get("tf_version") != TF_VERSION:
        old = out.with_name(out.stem + ".v1_unfiltered.jsonl")
        out.rename(old)
        print(f"[textfooler] archived old unfiltered output to {old.name}; regenerating", flush=True)
        prev = []
    done = {r["id"] for r in prev}

    if args.split == "train":
        a = cfg["augmentation"]
        target = args.n or round(len(df) * a["textfooler_share"] / a["original_share"])
        frac = df["label"].value_counts(normalize=True)
        quota = {c: round(target * f) for c, f in frac.items()}
        have = pd.Series([r["label"] for r in prev if r["status"] == "success"]).value_counts().to_dict()
        order = df.sample(frac=1.0, random_state=cfg["seed"])        # whole train, shuffled
    else:
        target = args.n or tc["test_n"]
        frac = df["label"].value_counts(normalize=True)
        order = pd.concat([g.sample(min(len(g), round(target * frac[c])), random_state=cfg["seed"])
                           for c, g in df.groupby("label")]).sample(frac=1.0, random_state=cfg["seed"])
        quota, have = None, {}
    todo = order[~order["id"].isin(done)]
    print(f"[textfooler] victim={victim} split={args.split} synonyms={syn_name} target={target} "
          f"already_done={len(done)} successes_so_far={sum(have.values())}", flush=True)

    t0, n, counts = time.time(), 0, {}
    with open(out, "a") as f:
        for row in todo.itertuples(index=False):
            if quota is not None:
                if all(have.get(c, 0) >= q for c, q in quota.items()):
                    break
                if have.get(row.label, 0) >= quota[row.label]:
                    continue
            try:
                res = attacker.attack(row.text, int(row.label_id))
            except Exception as e:  # keep the long job alive
                res = {"status": "error", "error": repr(e)[:200]}
            rec = {"tf_version": TF_VERSION, "id": row.id, "label": row.label, "label_id": int(row.label_id),
                   "source": row.source, "original": row.text, "synonyms": syn_name, **res}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if res["status"] == "success":
                have[row.label] = have.get(row.label, 0) + 1
            n += 1
            if n % 50 == 0:
                rate = n / (time.time() - t0)
                print(f"[textfooler] {n} attacked | {counts} | successes {sum(have.values())}/{target} "
                      f"| {rate:.1f}/s", flush=True)
    print(f"[textfooler] finished {counts} | successes by class {have}")


if __name__ == "__main__":
    main()
