"""Retrain the best grid-search configuration with 3 seeds and compare it with the default settings.

For each seed: train XFakeNet v2 (augmented corpus) with the chosen hyperparameters, evaluate on the
clean LIAR test set, then attack it with a fresh TextFooler run on the same 499 test statements used
for every other model. The seed-42 model is kept on Drive as runs/xfakenet_tuned.

Output: results/grid/final/seed<k>.json, data/augment/textfooler_test_xfakenet_tuned_s<k>.jsonl,
        results/grid/tuned_vs_default.md / .json
Usage : python scripts/21_final_tuned.py [--stage run|summary|all]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402


def lines(f: Path) -> int:
    return sum(1 for _ in open(f)) if f.exists() else 0


def attack_stats(f: Path):
    if not f.exists():
        return None
    t = pd.DataFrame([json.loads(l) for l in open(f)])
    vc = t["status"].value_counts()
    s, fl, n = int(vc.get("success", 0)), int(vc.get("failed", 0)), len(t)
    return {"acc_under_attack": fl / n, "success_rate": s / max(1, s + fl)}


def sh(cmd: list[str], keep: tuple[str, ...]) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        tail = (tail + [line])[-30:]
        if any(k in line for k in keep):
            print("   " + line.rstrip(), flush=True)
    rc = proc.wait()
    if rc != 0:
        print("".join(tail), flush=True)
    return rc


def run(cfg: dict, local: str) -> None:
    gc = cfg["grid"]
    best = json.load(open(p(f"{gc['out_dir']}/grid_best.json")))
    print(f"[tuned] best config {best['config']} (valid macro-F1 {best['valid_macro_f1']:.4f})", flush=True)
    for seed in gc["final_seeds"]:
        res_f = p(f"{gc['out_dir']}/final/seed{seed}.json")
        atk_f = p(f"data/augment/textfooler_test_xfakenet_tuned_s{seed}.jsonl")
        run_dir = Path(local) / f"tuned_s{seed}"
        if res_f.exists() and lines(atk_f) >= 499:
            print(f"[tuned] seed {seed}: done", flush=True)
            continue
        if not (run_dir / "xfakenet.pt").exists():
            print(f"[tuned] seed {seed}: training", flush=True)
            sh([sys.executable, str(ROOT / "scripts/11_train_xfakenet.py"), "--data", "aug",
                "--run-name", f"xfakenet_tuned_s{seed}", "--seed", str(seed), "--epochs", str(gc["max_epochs"]),
                "--lr", str(best["lr"]), "--batch-size", str(best["batch_size"]), "--dropout", str(best["dropout"]),
                "--out-dir", str(run_dir)], (" done in ", "test:", "Error", "Traceback"))
            if (run_dir / "metrics.json").exists():
                shutil.copy(run_dir / "metrics.json", res_f)
        if lines(atk_f) < 499 and (run_dir / "xfakenet.pt").exists():
            print(f"[tuned] seed {seed}: TextFooler attack", flush=True)
            sh([sys.executable, str(ROOT / "scripts/07_textfooler_generate.py"), "--split", "test",
                "--victim", str(run_dir), "--victim-kind", "xfakenet", "--out", str(atk_f)],
               ("[textfooler] finished", "Error", "Traceback"))
        if seed == 42 and (run_dir / "xfakenet.pt").exists() and not p("runs/xfakenet_tuned/xfakenet.pt").exists():
            shutil.copytree(run_dir, p("runs/xfakenet_tuned"), dirs_exist_ok=True)


def summarise(cfg: dict) -> None:
    gc = cfg["grid"]
    best = json.load(open(p(f"{gc['out_dir']}/grid_best.json")))
    groups = {
        "Default settings (XFakeNet v2)": (
            [p("runs/xfakenet_v2/metrics.json"), p("runs/xfakenet_v2_s1/metrics.json"), p("runs/xfakenet_v2_s2/metrics.json")],
            [p("data/augment/textfooler_test_xfakenet_v2.jsonl"), p("data/augment/textfooler_test_xfakenet_v2_s1.jsonl"),
             p("data/augment/textfooler_test_xfakenet_v2_s2.jsonl")]),
        "Tuned (grid-search best)": (
            [p(f"{gc['out_dir']}/final/seed{s}.json") for s in gc["final_seeds"]],
            [p(f"data/augment/textfooler_test_xfakenet_tuned_s{s}.jsonl") for s in gc["final_seeds"]]),
    }

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "–"
        return f"{np.mean(xs):.3f}" if len(xs) == 1 else f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f}"

    out, L = {}, []
    xc = cfg["xfakenet"]
    L += [f"Default settings: encoder LR {xc['lr']:g}, batch size {xc['batch_size']}, dropout 0.1, "
          f"{xc['epochs']} epochs. Tuned: encoder LR {best['lr']:g}, batch size {best['batch_size']}, "
          f"dropout {best['dropout']:g}, up to {gc['max_epochs']} epochs (best epoch by validation).", "",
          "| Model | Seeds | Valid macro-F1 | Test accuracy | Test macro-F1 | Accuracy under attack | Attack success rate |",
          "|---|---|---|---|---|---|---|"]
    for name, (mfiles, afiles) in groups.items():
        ms_ = [json.load(open(f)) for f in mfiles if f.exists()]
        at = [a for a in (attack_stats(f) for f in afiles) if a]
        if not ms_:
            continue
        row = {"valid_f1": [m["valid"]["macro_f1"] for m in ms_],
               "test_acc": [m["test"]["accuracy"] for m in ms_ if "test" in m],
               "test_f1": [m["test"]["macro_f1"] for m in ms_ if "test" in m],
               "acc_under_attack": [a["acc_under_attack"] for a in at],
               "success": [a["success_rate"] for a in at]}
        out[name] = row
        L.append(f"| {name} | {len(ms_)} | {ms(row['valid_f1'])} | {ms(row['test_acc'])} | {ms(row['test_f1'])} | "
                 f"{ms(row['acc_under_attack'])} | {ms(row['success'])} |")
    write_json(out, f"{gc['out_dir']}/tuned_vs_default.json")
    open(p(f"{gc['out_dir']}/tuned_vs_default.md"), "w").write("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "run", "summary"])
    ap.add_argument("--local", default="/content/tuned")
    args = ap.parse_args()
    cfg = load_config()
    p(f"{cfg['grid']['out_dir']}/final").mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "run"):
        run(cfg, args.local)
    if args.stage in ("all", "summary"):
        summarise(cfg)


if __name__ == "__main__":
    main()
