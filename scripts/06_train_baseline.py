"""Week-1 baseline: RoBERTa-base, five classes, class-weighted cross-entropy.

This is the "fine-tuned RoBERTa without the proposed modules" baseline from the
evaluation plan. It is also the victim model for TextFooler generation.

Input : data/processed/corpus_{train,valid,test}.csv, corpus_stats.json
Output: runs/baseline/best/            (model + tokenizer, chosen by valid macro-F1)
        runs/baseline/metrics.json     (valid + test: acc, macro-F1, per-class, per-source)
        runs/baseline/confusion_test.csv

Usage : python scripts/06_train_baseline.py [--epochs N] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import class_names, load_config, p, set_seed, write_json  # noqa: E402


class TextDS(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tok, max_len: int):
        self.enc = tok(df["text"].tolist(), truncation=True, max_length=max_len)
        self.y = df["label_id"].astype(int).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights: torch.Tensor, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.functional.cross_entropy(out.logits, labels,
                                           weight=self.class_weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    pred = logits.argmax(-1)
    present = sorted(set(labels.tolist()))
    return {"accuracy": accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, labels=present, average="macro", zero_division=0)}


def report(df: pd.DataFrame, pred: np.ndarray, classes: list[str]) -> dict:
    y = df["label_id"].to_numpy()
    present = sorted(set(y.tolist()))
    r = {
        "n": len(df),
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, labels=present, average="macro", zero_division=0), 4),
        "classes_in_eval": [classes[i] for i in present],
        "per_class": classification_report(y, pred, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True, zero_division=0),
        "per_source": {},
    }
    for src, idx in df.groupby("source").groups.items():
        ys, ps = y[df.index.get_indexer(idx)], pred[df.index.get_indexer(idx)]
        pr = sorted(set(ys.tolist()))
        r["per_source"][src] = {"n": len(ys), "accuracy": round(accuracy_score(ys, ps), 4),
                                "macro_f1": round(f1_score(ys, ps, labels=pr, average="macro",
                                                           zero_division=0), 4)}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny model + 200 rows, CPU-friendly check")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--run-name", default="baseline", help="output folder under runs/")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="absolute output folder (e.g. Colab local disk instead of Drive)")
    args = ap.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg["seed"] = args.seed
    bc = cfg["baseline"]
    set_seed(cfg["seed"])
    classes = class_names()
    d = cfg["paths"]["processed"]
    out_dir = Path(args.out_dir) if args.out_dir else p(f"{cfg['paths']['runs']}/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, valid, test = (pd.read_csv(p(f"{d}/corpus_{s}.csv")) for s in ["train", "valid", "test"])
    stats = json.load(open(p(f"{d}/corpus_stats.json")))
    weights = torch.tensor([stats["class_weights"][c] for c in classes], dtype=torch.float)

    model_name = bc["model"]
    if args.smoke:
        model_name = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"
        train, valid, test = train.head(200), valid.head(100), test.head(100)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=args.smoke)

    ds = {k: TextDS(v, tok, bc["max_len"]) for k, v in
          [("train", train), ("valid", valid), ("test", test)]}

    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs or bc["epochs"],
        per_device_train_batch_size=bc["batch_size"],
        per_device_eval_batch_size=bc["batch_size"] * 2,
        learning_rate=args.lr or bc["lr"],
        # warmup as an integer step count: works on transformers 4.x and 5.x
        # (5.x removed warmup_ratio)
        warmup_steps=int(math.ceil(len(train) / bc["batch_size"])
                         * (args.epochs or bc["epochs"]) * bc["warmup_ratio"]),
        weight_decay=bc["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=bc["fp16"] and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds["train"],
                              eval_dataset=ds["valid"], data_collator=DataCollatorWithPadding(tok),
                              compute_metrics=metrics_fn, class_weights=weights)
    trainer.train()
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))

    results = {"model": model_name, "class_weights": stats["class_weights"], "config": bc,
               "run_name": args.run_name, "seed": cfg["seed"], "lr": args.lr or bc["lr"], "epochs": args.epochs or bc["epochs"]}
    for name, df in [("valid", valid), ("test", test)]:
        pred = trainer.predict(ds[name]).predictions.argmax(-1)
        results[name] = report(df.reset_index(drop=True), pred, classes)
        print(f"[{args.run_name}] {name}: acc={results[name]['accuracy']} macro_f1={results[name]['macro_f1']} "
              f"per_source={results[name]['per_source']}")
        if name == "test":
            cm = confusion_matrix(df["label_id"], pred, labels=list(range(len(classes))))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_test.csv")
            pd.DataFrame({"id": df["id"], "text": df["text"], "label": df["label"],
                          "pred": [classes[i] for i in pred]}).to_csv(out_dir / "test_predictions.csv",
                                                                       index=False)
    write_json(results, out_dir / "metrics.json")


if __name__ == "__main__":
    main()
