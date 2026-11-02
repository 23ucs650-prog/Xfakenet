"""The integrated XFakeNet pipeline: one call returns class, explanations, modality flag and pattern.

    pipe = XFakeNetPipeline.load()
    out = pipe("Headline or statement text", image="path/or/PIL.Image")   # image optional

Output keys:
  label, probs                      classifier decision (Fabricated / Misleading / Credible)
  highlighted, word_scores,         Module 1 explanations. Highlights and the top segment come from
  segments, top_segment             SHAP (faithful); the built-in attention versions are kept as
  attention_highlighted,            attention_highlighted / attention_top_segment for comparison.
  attention_top_segment
  modality_flag                     TEXT / IMAGE / CROSS-MODAL (TEXT when no image is given)
  cscore, cscore_threshold          Module 2 contradiction score (only with an image)
  deception_pattern, pattern_probs  one of six patterns (only when CROSS-MODAL)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import load_config, p


class XFakeNetPipeline:
    def __init__(self, model_dir, cscore_threshold, modality_path=None, deception_path=None,
                 clip_name="openai/clip-vit-base-patch32", shap_permutations: int = 10, threshold: float = 0.65):
        from .model import Predictor
        self.pred = Predictor(model_dir)
        self.thr = float(cscore_threshold)
        self.m, self.hl_thr = shap_permutations, threshold
        self.clip_name = clip_name
        self._clip = None
        self.modality = self._load(modality_path)
        self.deception = self._load(deception_path)

    @staticmethod
    def _load(path):
        if path and Path(path).exists():
            import joblib
            return joblib.load(path)
        return None

    @classmethod
    def load(cls, model_dir: str = "runs/xfakenet_v2"):
        cfg = load_config()
        thr = json.load(open(p(f"{cfg['cscore']['out_dir']}/cscore_results.json")))["chosen_threshold"]
        return cls(p(model_dir), thr, p(cfg["pipeline"]["modality_model"]), p(cfg["deception"]["model"]),
                   cfg["cscore"]["clip_model"], cfg["explain"]["permutations_gpu"],
                   cfg["xfakenet"]["highlight_threshold"])

    @property
    def clip(self):
        if self._clip is None:
            from .clip_utils import Clip
            self._clip = Clip(self.clip_name)
        return self._clip

    def __call__(self, text: str, image=None) -> dict:
        from .attribution import explain_batch, highlight
        e = self.pred.explain([text])[0]
        words, segs = [w["w"] for w in e["words"]], [s["text"] for s in e["segments"]]
        a = explain_batch(self.pred.predict_proba, [{"words": words, "segments": segs, "pred": e["pred"]}],
                          self.pred.meta["classes"], m=self.m, methods=("shap", "segment"))[0]
        shap = a.get("shap", [0.0] * len(words))
        seg_sc = a.get("segment_shap") or [s["weight"] for s in e["segments"]]
        top_seg = segs[int(np.argmax(seg_sc))] if segs else text
        out = {"label": e["pred"], "probs": e["probs"],
               "highlighted": highlight(words, shap, self.hl_thr),
               "word_scores": [{"w": w, "shap": round(float(x), 4)} for w, x in zip(words, shap)],
               "segments": [{"text": s, "shap": round(float(x), 4)} for s, x in zip(segs, seg_sc)],
               "top_segment": top_seg,
               "attention_highlighted": e["highlighted"], "attention_top_segment": e["top_segment"],
               "modality_flag": "TEXT", "cscore": None, "cscore_threshold": self.thr,
               "deception_pattern": None, "pattern_probs": None}
        if image is None:
            return out
        from .clip_utils import pair_features
        T, I = self.clip.embed([text], [image])
        c = float(1.0 - (T * I).sum())
        out["cscore"] = round(c, 4)
        if c > self.thr:
            out["modality_flag"] = "CROSS-MODAL"
            if self.deception is not None:
                pr = self.deception["model"].predict_proba(pair_features(T, I))[0]
                cls = list(self.deception["model"].classes_)
                out["pattern_probs"] = {k: round(float(v), 4) for k, v in zip(cls, pr)}
                out["deception_pattern"] = cls[int(np.argmax(pr))]
        elif self.modality is not None:
            m = self.modality
            zt = float(m["text"].decision_function(T)[0])
            zi = float(m["image"].decision_function(I)[0])
            z = m["fusion_scaler"].transform([[zt, zi]])[0]
            ct, ci = m["fusion_coef"][0] * z[0], m["fusion_coef"][1] * z[1]
            out["modality_flag"] = "TEXT" if abs(ct) >= abs(ci) else "IMAGE"
            out["modality_contrib"] = {"text": round(float(ct), 4), "image": round(float(ci), 4)}
        return out
