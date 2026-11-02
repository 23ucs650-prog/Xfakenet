"""Week 4: final results tables (classification, robustness, ablations, C-score, faithfulness, attribution).

Seeds: runs trained with seed 42 (weeks 1-3) plus any extra-seed runs named <run>_s<seed>
(e.g. xfakenet_v2_s1). With 2+ seeds, tables show mean ± standard deviation.

Output: results/final/results_tables.md, results/final/results_tables.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402

FAMILIES = {  # name: (metrics of seed-42 run, glob for extra seeds, fresh-attack files)
    "Baseline (RoBERTa)": ("runs/baseline/metrics.json", "runs/baseline_s*/metrics.json",
                           ["data/augment/textfooler_test.jsonl", "data/augment/textfooler_test_baseline_s*.jsonl"]),
    "XFakeNet, no augmentation": ("runs/xfakenet_noaug/metrics.json", "runs/xfakenet_noaug_s*/metrics.json",
                                  ["data/augment/textfooler_test_xfakenet_noaug.jsonl",
                                   "data/augment/textfooler_test_xfakenet_noaug_s*.jsonl"]),
    "XFakeNet v2 (augmented)": ("runs/xfakenet_v2/metrics.json", "runs/xfakenet_v2_s*/metrics.json",
                                ["data/augment/textfooler_test_xfakenet_v2.jsonl",
                                 "data/augment/textfooler_test_xfakenet_v2_s*.jsonl"]),
    "XFakeNet v3 (2nd adversarial round)": ("runs/xfakenet_v3/metrics.json", "runs/xfakenet_v3_s*/metrics.json",
                                            ["data/augment/textfooler_test_xfakenet_v3.jsonl"]),
}


def jload(f):
    f = p(f)
    return json.load(open(f)) if f.exists() else None


def ms(xs: list[float], d: int = 3) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.{d}f}"
    return f"{np.mean(xs):.{d}f} ± {np.std(xs, ddof=1):.{d}f}"


def attack_stats(f: str) -> dict | None:
    f = p(f)
    if not f.exists():
        return None
    t = pd.DataFrame([json.loads(l) for l in open(f)])
    vc = t["status"].value_counts()
    s, fl, n = int(vc.get("success", 0)), int(vc.get("failed", 0)), len(t)
    return {"acc_under_attack": fl / n, "success_rate": s / max(1, s + fl)}


def main() -> None:
    cfg = load_config()
    out = {}
    L = ["# XFakeNet – final results tables", ""]

    # ---- T1 classification + T2 robustness
    t1 = ["## Table 1. Classification on the clean LIAR test set (1,283 statements, 3 classes)", "",
          "| Model | Seeds | Accuracy | Macro-F1 | F1 Fabricated | F1 Misleading | F1 Credible |",
          "|---|---|---|---|---|---|---|"]
    t2 = ["## Table 2. Robustness (each model attacked by its own TextFooler run, 499 test statements)", "",
          "| Model | Seeds | Accuracy under attack | Attack success rate | Paraphrased: prediction unchanged |",
          "|---|---|---|---|---|"]
    rob = jload(f"{cfg['week3']['out_dir']}/robustness.json") or {}
    for name, (m42, extra, atk) in FAMILIES.items():
        runs = [m for m in [jload(m42)] + [jload(f) for f in sorted(glob.glob(str(p(extra))))] if m]
        if not runs:
            continue
        get = lambda k, c=None: [r["test"]["per_class"][c]["f1-score"] if c else r["test"][k] for r in runs]  # noqa
        t1.append(f"| {name} | {len(runs)} | {ms(get('accuracy'))} | {ms(get('macro_f1'))} | "
                  f"{ms(get(None, 'Fabricated'))} | {ms(get(None, 'Misleading'))} | {ms(get(None, 'Credible'))} |")
        atks = [a for f in atk for g in sorted(glob.glob(str(p(f)))) if (a := attack_stats(g))]
        para = rob.get(name, {}).get("paraphrase", {}).get("prediction_unchanged")
        if atks:
            t2.append(f"| {name} | {len(atks)} | {ms([a['acc_under_attack'] for a in atks])} | "
                      f"{ms([a['success_rate'] for a in atks])} | {'' if para is None else f'{para*100:.0f}%'} |")
        out[name] = {"n_seeds": len(runs), "macro_f1": get("macro_f1"), "accuracy": get("accuracy"),
                     "acc_under_attack": [a["acc_under_attack"] for a in atks],
                     "attack_success": [a["success_rate"] for a in atks]}
    L += t1 + ["", "*Mean ± standard deviation over seeds when more than one seed was run.*", ""] + t2 + [""]

    # ---- T3 ablations
    L += ["## Table 3. Ablations", ""]
    a, b = out.get("XFakeNet, no augmentation"), out.get("XFakeNet v2 (augmented)")
    if a and b:
        L += ["**Adversarial + paraphrase augmentation** (XFakeNet without vs with):", "",
              "| | Without | With |", "|---|---|---|",
              f"| Clean macro-F1 | {ms(a['macro_f1'])} | {ms(b['macro_f1'])} |",
              f"| Accuracy under attack | {ms(a['acc_under_attack'])} | {ms(b['acc_under_attack'])} |",
              f"| Attack success rate | {ms(a['attack_success'])} | {ms(b['attack_success'])} |", ""]
    mres = jload(f"{cfg['week3']['out_dir']}/modality_results.json")
    if mres:
        au = mres["fake_vs_real_auc_5fold"]
        L += ["**C-score signal** (FakeNewsNet fake vs real, 5-fold ROC-AUC):", "",
              "| Features | AUC |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in au.items()] + [""]

    # ---- T4 C-score calibration
    c = jload(f"{cfg['cscore']['out_dir']}/cscore_results.json")
    if c:
        L += ["## Table 4. C-score calibration (aligned vs mismatched pairs, test split)", "",
              f"ROC-AUC {c['test_auc_aligned_vs_mismatched']:.3f}", "",
              "| Threshold | Value | Accuracy | Flags genuine pairs | Flags mismatched pairs |", "|---|---|---|---|---|"]
        for k, r in c["thresholds"].items():
            L.append(f"| {k} | {r['threshold']:.3f} | {r['accuracy']:.3f} | {r['flag_rate_aligned']*100:.0f}% | "
                     f"{r['flag_rate_mismatched']*100:.0f}% |")
        L.append("")

    # ---- T5 faithfulness
    fm = p(f"{cfg['week4']['out_dir']}/faithfulness.md")
    if fm.exists():
        L += ["## Table 5. Explanation faithfulness (deletion test) and agreement with SHAP", "",
              fm.read_text(), ""]

    fe = p(f"{cfg['week4']['out_dir']}/faithful_explanations.md")
    if fe.exists():
        L += ["## Table 5b. Faithful explanations: attention vs occlusion vs SHAP", "", fe.read_text(), ""]

    # ---- T6 deception attribution
    d = jload(f"{cfg['week4']['out_dir']}/deception_results.json")
    if d:
        L += ["## Table 6. Deception-pattern attribution", "",
              f"- Gemini labels: {d.get('n_labelled')} pairs; categories: {d.get('label_counts')}"]
        if "classifier_cv" in d:
            cv = d["classifier_cv"]
            L.append(f"- Classifier (CLIP features, {cv['folds']}-fold CV vs Gemini labels, {cv['n']} pairs): "
                     f"accuracy {cv['accuracy']:.3f}, macro-F1 {cv['macro_f1']:.3f} "
                     f"(majority-class accuracy {cv['majority_class_accuracy']:.3f})")
        if "handcheck" in d:
            h = d["handcheck"]
            L.append(f"- Hand-check of {h['n_checked']} pairs: Gemini correct {h['gemini_accuracy']*100:.0f}%, "
                     f"Cohen's kappa {h['gemini_kappa']}"
                     + (f"; classifier agrees with human labels {h['classifier_accuracy_vs_human']*100:.0f}% "
                        f"(n={h['classifier_n']})" if "classifier_accuracy_vs_human" in h else ""))
        else:
            L.append("- Hand-check: not uploaded yet")
        L.append("")

    md = "\n".join(L)
    open(p("results/final/results_tables.md"), "w").write(md)
    write_json(out, "results/final/results_tables.json")
    print(md)


if __name__ == "__main__":
    main()
