"""Train the XFakeNet classifier (RoBERTa + token/segment attention), class-weighted CE.

--data orig : LIAR training split only            -> runs/xfakenet_noaug  (ablation: no augmentation)
--data aug  : 70% LIAR + 15% paraphrase + 15% TF  -> runs/xfakenet_v2     (robust model v2)

The best epoch is chosen by validation macro-F1. Validation and test are always the
clean LIAR splits.

Output: runs/<run>/xfakenet.pt, xfakenet_meta.json, tokenizer, metrics.json,
        test_predictions.csv, confusion_test.csv

Usage : python scripts/11_train_xfakenet.py --data aug --run-name xfakenet_v2
        python scripts/11_train_xfakenet.py --smoke        (1-minute check on a few rows)
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet import model as xm  # noqa: E402
from xfakenet.common import class_names, load_config, p, set_seed, write_json  # noqa: E402


def report(df: pd.DataFrame, pred: np.ndarray, classes: list[str]) -> dict:
    y = df["label_id"].to_numpy()
    return {
        "n": len(df),
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "per_class": classification_report(y, pred, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True, zero_division=0),
    }


@torch.no_grad()
def predict(model, tok, texts, cfg, device, bs=128):
    model.eval()
    out = []
    for s in range(0, len(texts), bs):
        enc, _, _ = xm.encode(tok, texts[s:s + bs], cfg["max_len"], cfg["max_segments"], device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            out.append(model(**enc)["logits"].float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="aug", choices=["orig", "aug"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="absolute output folder (e.g. Colab local disk instead of Drive)")
    ap.add_argument("--train-file", default=None, help="override training csv (e.g. round-2 corpus)")
    ap.add_argument("--lr", type=float, default=None, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=None, help="attention layers + classifier learning rate")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--skip-test", action="store_true", help="grid search: never touch the test set")
    args = ap.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg["seed"] = args.seed
    xc = dict(cfg["xfakenet"])
    for key, val in [("lr", args.lr), ("head_lr", args.head_lr), ("batch_size", args.batch_size),
                     ("dropout", args.dropout)]:
        if val is not None:
            xc[key] = val
    xc.setdefault("dropout", 0.1)
    set_seed(cfg["seed"])
    classes = class_names()
    d = cfg["paths"]["processed"]
    run = args.run_name or ("xfakenet_v2" if args.data == "aug" else "xfakenet_noaug")
    if args.smoke:
        run = "xfakenet_smoke"
    out_dir = Path(args.out_dir) if args.out_dir else p(f"{cfg['paths']['runs']}/{run}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_file = args.train_file or (f"{d}/corpus_aug_train.csv" if args.data == "aug" else f"{d}/corpus_train.csv")
    train = pd.read_csv(p(train_file))
    valid = pd.read_csv(p(f"{d}/corpus_valid.csv"))
    test = pd.read_csv(p(f"{d}/corpus_test.csv"))
    epochs = args.epochs or xc["epochs"]
    if args.smoke:
        train, valid, test, epochs = train.sample(96, random_state=0), valid.head(64), test.head(64), 1
    if "source" not in train:
        train["source"] = "liar"
    print(f"[{run}] train={len(train)} {train['source'].value_counts().to_dict()} | device={device}", flush=True)

    counts = np.bincount(train["label_id"], minlength=len(classes)).astype(float)
    w = counts.sum() / (len(classes) * np.maximum(counts, 1))
    weights = torch.tensor(w / w.mean(), dtype=torch.float, device=device)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(xc["encoder"])
    model = xm.build(xc["encoder"], len(classes), xc["max_segments"], xc["att_dim"], xc["dropout"]).to(device)

    head = [p_ for n, p_ in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p_ for n, p_ in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": xc["lr"]},
                             {"params": head, "lr": xc["head_lr"]}], weight_decay=xc["weight_decay"])
    steps = math.ceil(len(train) / xc["batch_size"]) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * xc["warmup_ratio"]), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    texts, labels = train["text"].tolist(), train["label_id"].to_numpy()
    import tempfile
    best_f1, best_epoch, history = -1.0, -1, []
    tmp = Path(tempfile.gettempdir()) / f"{run}_best"   # local disk, not Drive (faster)
    rng = np.random.default_rng(cfg["seed"])
    for ep in range(1, epochs + 1):
        model.train()
        order, t0, run_loss = rng.permutation(len(train)), time.time(), 0.0
        for step, s in enumerate(range(0, len(order), xc["batch_size"]), 1):
            idx = order[s:s + xc["batch_size"]]
            enc, _, _ = xm.encode(tok, [texts[i] for i in idx], xc["max_len"], xc["max_segments"], device)
            y = torch.tensor(labels[idx], device=device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(**enc)["logits"]
            loss = torch.nn.functional.cross_entropy(logits.float(), y, weight=weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if step % 100 == 0:
                print(f"[{run}] epoch {ep} step {step} loss {run_loss / step:.4f}", flush=True)
        vp = predict(model, tok, valid["text"].tolist(), xc, device).argmax(-1)
        vf1 = f1_score(valid["label_id"], vp, average="macro")
        print(f"[{run}] epoch {ep} done in {time.time() - t0:.0f}s | train loss {run_loss / step:.4f} "
              f"| valid macro-F1 {vf1:.4f}", flush=True)
        history.append({"epoch": ep, "train_loss": round(run_loss / step, 4), "valid_macro_f1": round(float(vf1), 4)})
        if vf1 > best_f1:
            best_f1, best_epoch = vf1, ep
            xm.save(model, tok, tmp, {})

    # reload the best epoch and evaluate
    model.load_state_dict(torch.load(tmp / "xfakenet.pt", map_location=device))
    meta = {"classes": classes, "encoder": xc["encoder"], "max_len": xc["max_len"],
            "max_segments": xc["max_segments"], "att_dim": xc["att_dim"],
            "highlight_threshold": xc["highlight_threshold"], "data": args.data, "best_epoch": best_epoch,
            "dropout": xc["dropout"]}
    xm.save(model, tok, out_dir, meta)
    shutil.rmtree(tmp, ignore_errors=True)

    results = {"run_name": run, "data": args.data, "seed": cfg["seed"], "train_file": train_file, "train_sources": train["source"].value_counts().to_dict(),
               "config": xc, "epochs": epochs, "best_epoch": best_epoch, "lr": xc["lr"], "history": history}
    for name, df in [("valid", valid)] + ([] if args.skip_test else [("test", test)]):
        pred = predict(model, tok, df["text"].tolist(), xc, device).argmax(-1)
        results[name] = report(df.reset_index(drop=True), pred, classes)
        print(f"[{run}] {name}: acc={results[name]['accuracy']} macro_f1={results[name]['macro_f1']}", flush=True)
        if name == "test":
            cm = confusion_matrix(df["label_id"], pred, labels=list(range(len(classes))))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_test.csv")
            pd.DataFrame({"id": df["id"], "text": df["text"], "label": df["label"],
                          "pred": [classes[i] for i in pred]}).to_csv(out_dir / "test_predictions.csv", index=False)
    write_json(results, out_dir / "metrics.json")


if __name__ == "__main__":
    main()
