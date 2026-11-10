"""Grid search over XFakeNet training hyperparameters (augmented corpus), selected on VALIDATION only.

Grid (configs/week1.yaml -> grid): encoder learning rate x batch size x dropout, each run trained for
`max_epochs` epochs with validation macro-F1 recorded after every epoch, so the number of epochs is
tuned too (lrs x batch sizes x dropouts x epochs configurations in total).

The test set is never evaluated during the search (--skip-test). Weights are kept on Colab's local
disk and deleted after each run; only the results go to Drive, one file per run, so the search can
be stopped and resumed (finished runs are skipped).

Output: results/grid/runs/<config>.json, results/grid/grid_results.csv, results/grid/grid_summary.md,
        results/grid/grid_best.json, results/grid/grid_heatmap.png
Usage : python scripts/20_grid_search.py [--stage run|summary|all] [--local /content/grid]
"""
from __future__ import annotations

import argparse
import itertools
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


def name_of(lr: float, bs: int, do: float) -> str:
    return f"lr{lr:g}_bs{bs}_do{do:g}"


def combos(gc: dict):
    return list(itertools.product(gc["lrs"], gc["batch_sizes"], gc["dropouts"]))


def run_all(cfg: dict, local: str) -> None:
    gc = cfg["grid"]
    out = p(f"{gc['out_dir']}/runs")
    out.mkdir(parents=True, exist_ok=True)
    todo = [(lr, bs, do) for lr, bs, do in combos(gc) if not (out / f"{name_of(lr, bs, do)}.json").exists()]
    print(f"[grid] {len(combos(gc))} configurations x {gc['max_epochs']} epochs; {len(todo)} still to run", flush=True)
    for k, (lr, bs, do) in enumerate(todo, 1):
        name = name_of(lr, bs, do)
        run_dir = Path(local) / name
        print(f"[grid] ({k}/{len(todo)}) {name}", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts/11_train_xfakenet.py"), "--data", "aug",
               "--run-name", f"grid_{name}", "--epochs", str(gc["max_epochs"]), "--lr", str(lr),
               "--batch-size", str(bs), "--dropout", str(do), "--skip-test", "--out-dir", str(run_dir)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            tail = (tail + [line])[-30:]
            if " done in " in line or "Error" in line or "Traceback" in line:
                print("   " + line.rstrip(), flush=True)
        if proc.wait() != 0 or not (run_dir / "metrics.json").exists():
            print("[grid] run FAILED; last output:\n" + "".join(tail), flush=True)
            shutil.rmtree(run_dir, ignore_errors=True)
            continue
        m = json.load(open(run_dir / "metrics.json"))
        write_json({"config": name, "lr": lr, "batch_size": bs, "dropout": do,
                    "best_epoch": m["best_epoch"], "valid_macro_f1": m["valid"]["macro_f1"],
                    "valid_accuracy": m["valid"]["accuracy"], "history": m["history"]},
                   f"{gc['out_dir']}/runs/{name}.json")
        shutil.rmtree(run_dir, ignore_errors=True)


def summarise(cfg: dict) -> dict:
    gc = cfg["grid"]
    rows = [json.load(open(f)) for f in sorted(p(f"{gc['out_dir']}/runs").glob("*.json"))]
    if not rows:
        print("[grid] no finished runs yet")
        return {}
    df = pd.DataFrame(rows).sort_values("valid_macro_f1", ascending=False).reset_index(drop=True)
    df.drop(columns=["history"]).to_csv(p(f"{gc['out_dir']}/grid_results.csv"), index=False)
    best = df.iloc[0]
    epochs = pd.DataFrame([{"config": r["config"], **h} for r in rows for h in r["history"]])
    best_json = {"config": best["config"], "lr": float(best["lr"]), "batch_size": int(best["batch_size"]),
                 "dropout": float(best["dropout"]), "best_epoch": int(best["best_epoch"]),
                 "valid_macro_f1": float(best["valid_macro_f1"]), "n_runs": len(df),
                 "n_total": len(combos(gc))}
    write_json(best_json, f"{gc['out_dir']}/grid_best.json")

    L = [f"Grid search: {len(df)} of {len(combos(gc))} configurations finished, each trained up to "
         f"{gc['max_epochs']} epochs ({len(epochs)} configuration x epoch points). Selected on validation "
         f"macro-F1; the test set was not used.", "",
         "| Rank | Encoder LR | Batch size | Dropout | Best epoch | Valid macro-F1 | Valid accuracy |",
         "|---|---|---|---|---|---|---|"]
    for i, r in df.iterrows():
        L.append(f"| {i + 1} | {r['lr']:g} | {int(r['batch_size'])} | {r['dropout']:g} | {int(r['best_epoch'])} | "
                 f"{r['valid_macro_f1']:.4f} | {r['valid_accuracy']:.4f} |")
    L += ["", "Average best validation macro-F1 for each value (how much each setting matters):", "",
          "| Hyperparameter | Value | Mean valid macro-F1 | Runs |", "|---|---|---|---|"]
    for col, lab in [("lr", "Encoder LR"), ("batch_size", "Batch size"), ("dropout", "Dropout")]:
        for v, g in df.groupby(col):
            L.append(f"| {lab} | {v:g} | {g['valid_macro_f1'].mean():.4f} | {len(g)} |")
    ep = epochs.groupby("epoch")["valid_macro_f1"].agg(["mean", "max"])
    L += ["", "| Epoch | Mean valid macro-F1 over configs | Best over configs |", "|---|---|---|"] + \
         [f"| {e} | {r['mean']:.4f} | {r['max']:.4f} |" for e, r in ep.iterrows()]
    spread = df["valid_macro_f1"].max() - df["valid_macro_f1"].min()
    L += ["", f"Best: encoder LR {best['lr']:g}, batch size {int(best['batch_size'])}, dropout {best['dropout']:g}, "
          f"{int(best['best_epoch'])} epochs (valid macro-F1 {best['valid_macro_f1']:.4f}). Spread across the grid: "
          f"{spread:.4f}."]
    open(p(f"{gc['out_dir']}/grid_summary.md"), "w").write("\n".join(L))
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bss, lrs, dos = sorted(df["batch_size"].unique()), sorted(df["lr"].unique()), sorted(df["dropout"].unique())
        fig, axes = plt.subplots(1, len(bss), figsize=(4.6 * len(bss), 3.6), squeeze=False, constrained_layout=True)
        vmin, vmax = df["valid_macro_f1"].min(), df["valid_macro_f1"].max()
        for k, (ax, bs) in enumerate(zip(axes[0], bss)):
            grid = np.full((len(dos), len(lrs)), np.nan)
            for _, r in df[df.batch_size == bs].iterrows():
                grid[dos.index(r["dropout"]), lrs.index(r["lr"])] = r["valid_macro_f1"]
            im = ax.imshow(grid, cmap="Oranges", vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(dos)):
                for j in range(len(lrs)):
                    if not np.isnan(grid[i, j]):
                        dark = (grid[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.6
                        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=10,
                                color="white" if dark else "black")
            ax.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs])
            ax.set_yticks(range(len(dos)), [f"{v:g}" for v in dos])
            ax.set_xlabel("Encoder learning rate")
            if k == 0:
                ax.set_ylabel("Dropout")
            ax.set_title(f"Batch size {bs}")
        fig.suptitle("Validation macro-F1 across the grid (best epoch per run)", fontsize=12)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        fig.savefig(p(f"{gc['out_dir']}/grid_heatmap.png"), dpi=150, bbox_inches="tight")
    except Exception as e:
        print("[grid] heatmap skipped:", e)
    return best_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "run", "summary"])
    ap.add_argument("--local", default="/content/grid", help="local folder for temporary weights")
    args = ap.parse_args()
    cfg = load_config()
    if args.stage in ("all", "run"):
        run_all(cfg, args.local)
    if args.stage in ("all", "summary"):
        summarise(cfg)


if __name__ == "__main__":
    main()
