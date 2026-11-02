"""Compare models on clean, paraphrased and adversarial LIAR test text.

Models: baseline (week 1), XFakeNet without augmentation, XFakeNet v2 (augmented).

1. Clean test        : accuracy / macro-F1 on the 1,283 LIAR test statements.
2. Paraphrased test  : accuracy on the Gemini-paraphrased test statements vs their originals,
                       and how often the prediction stays the same.
3. TextFooler (fresh): each model attacked by its OWN TextFooler run on the same 499 test
                       statements. This is the fair robustness number.
4. TextFooler (transfer, reference only): accuracy on the adversarial texts that fooled the
                       baseline. Easier for other models, so it overstates robustness.

Output: results/week3/robustness.json, results/week3/robustness.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, write_json  # noqa: E402

MODELS = {  # name: (kind, model dir, fresh TextFooler test file)
    "Baseline (RoBERTa)": ("hf", "runs/baseline/best", "data/augment/textfooler_test.jsonl"),
    "XFakeNet, no augmentation": ("xfakenet", "runs/xfakenet_noaug",
                                  "data/augment/textfooler_test_xfakenet_noaug.jsonl"),
    "XFakeNet v2 (augmented)": ("xfakenet", "runs/xfakenet_v2", "data/augment/textfooler_test_xfakenet_v2.jsonl"),
}
METRICS = {"Baseline (RoBERTa)": "runs/baseline/metrics.json",
           "XFakeNet, no augmentation": "runs/xfakenet_noaug/metrics.json",
           "XFakeNet v2 (augmented)": "runs/xfakenet_v2/metrics.json"}


def jl(path):
    f = p(path)
    return pd.DataFrame([json.loads(line) for line in open(f)]) if f.exists() else pd.DataFrame()


def predictor(kind: str, model_dir: str, cfg: dict):
    if kind == "xfakenet":
        from xfakenet.model import Predictor
        return Predictor(p(model_dir)).predict_proba
    from xfakenet.textfooler import make_predict
    return make_predict(str(p(model_dir)), cfg["baseline"]["max_len"])


def main() -> None:
    cfg = load_config()
    out_dir = cfg["week3"]["out_dir"]
    para = jl(cfg["paraphrase"]["out_test"])
    para = para[para["status"] == "ok"].reset_index(drop=True) if len(para) else para
    base_adv = jl(cfg["textfooler"]["out_test"])
    base_adv = base_adv[(base_adv["status"] == "success") & (base_adv.get("tf_version", 0) == 2)] \
        if len(base_adv) else base_adv

    res = {}
    for name, (kind, mdir, tf_file) in MODELS.items():
        if not p(mdir).exists():
            print(f"[robust] {name}: model not found at {mdir}, skipped")
            continue
        r = {}
        m = json.load(open(p(METRICS[name])))
        r["clean_accuracy"], r["clean_macro_f1"] = m["test"]["accuracy"], m["test"]["macro_f1"]
        pred = predictor(kind, mdir, cfg)

        if len(para):
            po = pred(para["original"].tolist()).argmax(1)
            pp = pred(para["text"].tolist()).argmax(1)
            y = para["label_id"].to_numpy()
            r["paraphrase"] = {"n": len(para), "acc_original": round(accuracy_score(y, po), 4),
                               "acc_paraphrased": round(accuracy_score(y, pp), 4),
                               "prediction_unchanged": round(float((po == pp).mean()), 4)}

        t = jl(tf_file)
        if len(t):
            vc = t["status"].value_counts()
            s, f, n = int(vc.get("success", 0)), int(vc.get("failed", 0)), len(t)
            r["textfooler_fresh"] = {"n": n, "clean_acc_on_sample": round((s + f) / n, 4),
                                     "acc_under_attack": round(f / n, 4),
                                     "attack_success_rate": round(s / max(1, s + f), 4),
                                     "avg_words_changed": round(float(t.loc[t.status == "success",
                                                                          "words_changed"].mean()), 2)
                                     if s else None}
        if len(base_adv):
            pa = pred(base_adv["text"].tolist()).argmax(1)
            r["textfooler_transfer_from_baseline"] = {
                "n": len(base_adv), "acc": round(accuracy_score(base_adv["label_id"], pa), 4)}
        res[name] = r
        print(f"[robust] {name}: {r}", flush=True)

    write_json(res, f"{out_dir}/robustness.json")

    L = ["| Model | Clean acc | Clean macro-F1 | Paraphrased acc (orig → para) | Prediction unchanged "
         "| TextFooler: acc under attack | Attack success rate | Transfer acc* |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        pa = r.get("paraphrase", {})
        tf_ = r.get("textfooler_fresh", {})
        tr = r.get("textfooler_transfer_from_baseline", {})
        L.append(f"| {name} | {r['clean_accuracy']:.3f} | {r['clean_macro_f1']:.3f} | "
                 f"{pa.get('acc_original', float('nan')):.3f} → {pa.get('acc_paraphrased', float('nan')):.3f} | "
                 f"{pa.get('prediction_unchanged', float('nan')) * 100:.0f}% | "
                 f"{tf_.get('acc_under_attack', float('nan')):.3f} | "
                 f"{tf_.get('attack_success_rate', float('nan')) * 100:.0f}% | {tr.get('acc', float('nan')):.3f} |")
    L.append("")
    L.append("*Transfer = accuracy on the adversarial texts that fooled the baseline (reference only; "
             "the fair robustness number is the model's own TextFooler attack).")
    open(p(f"{out_dir}/robustness.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
