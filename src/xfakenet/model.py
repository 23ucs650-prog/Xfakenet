"""XFakeNet classifier: RoBERTa + jointly trained token- and segment-attention (Module 1 + 3).

    tokens --RoBERTa--> h_t
    token attention (within each segment):   a_t = softmax_t( v_t . tanh(W_t h_t) )
    segment vector:                          s_k = sum_t a_t h_t
    segment attention:                       alpha_k = softmax_k( v_s . tanh(W_s s_k) )
    statement vector:                        d = sum_k alpha_k s_k
    prediction:                              softmax(W_c d)

The classifier only sees d, so the attention weights are part of the decision
itself (explanation produced in the forward pass), not computed afterwards.
Token importance shown to the user = a_t * alpha_(segment of t).
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .explain_utils import word_explanation
from .segments import segment_spans


class XFakeNetModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_labels: int, max_segments: int = 8,
                 att_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.K = max_segments
        self.tok_proj, self.tok_v = nn.Linear(d, att_dim), nn.Linear(att_dim, 1, bias=False)
        self.seg_proj, self.seg_v = nn.Linear(d, att_dim), nn.Linear(att_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(d, num_labels)

    def forward(self, input_ids, attention_mask, seg_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # B L D
        b, l, _ = h.shape
        e = self.tok_v(torch.tanh(self.tok_proj(h))).squeeze(-1).float()                       # B L
        ks = torch.arange(self.K, device=h.device)
        member = seg_ids.unsqueeze(1) == ks.view(1, -1, 1)                                       # B K L
        scores = e.unsqueeze(1).expand(b, self.K, l).masked_fill(~member, -1e4)
        a = torch.softmax(scores, -1) * member                                                   # B K L (empty segment -> 0)
        s = torch.bmm(a.to(h.dtype), h)                                                          # B K D
        valid = member.any(-1)                                                                   # B K
        u = self.seg_v(torch.tanh(self.seg_proj(s))).squeeze(-1).float().masked_fill(~valid, -1e4)
        alpha = torch.softmax(u, -1) * valid                                                     # B K
        doc = torch.bmm(alpha.unsqueeze(1).to(s.dtype), s).squeeze(1)                            # B D
        logits = self.cls(self.drop(doc)).float()
        token_imp = (a * alpha.unsqueeze(-1)).sum(1)                                             # B L
        return {"logits": logits, "token_att": token_imp, "seg_att": alpha}


def encode(tok, texts: Sequence[str], max_len: int, max_segments: int, device=None):
    """Tokenise and assign every token to a segment (special/pad tokens get -1)."""
    enc = tok(list(texts), truncation=True, max_length=max_len, padding=True,
              return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    seg_ids = torch.full(enc["input_ids"].shape, -1, dtype=torch.long)
    spans_all = []
    for i, text in enumerate(texts):
        spans = segment_spans(text, max_segments)
        starts = [a for a, _ in spans]
        spans_all.append(spans)
        for j, (a, z) in enumerate(offsets[i].tolist()):
            if z > a:
                seg_ids[i, j] = max(0, bisect_right(starts, a) - 1)
    enc["seg_ids"] = seg_ids
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc, offsets, spans_all


def build(encoder_name: str, num_labels: int, max_segments: int, att_dim: int) -> XFakeNetModel:
    from transformers import AutoModel
    return XFakeNetModel(AutoModel.from_pretrained(encoder_name), num_labels, max_segments, att_dim)


def save(model: XFakeNetModel, tok, out_dir: str | Path, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "xfakenet.pt")
    model.encoder.config.save_pretrained(out / "encoder_config")
    tok.save_pretrained(out)
    json.dump(meta, open(out / "xfakenet_meta.json", "w"), indent=2)


def load(model_dir: str | Path, device: str = "cpu"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    d = Path(model_dir)
    meta = json.load(open(d / "xfakenet_meta.json"))
    enc = AutoModel.from_config(AutoConfig.from_pretrained(d / "encoder_config"))
    model = XFakeNetModel(enc, len(meta["classes"]), meta["max_segments"], meta["att_dim"])
    model.load_state_dict(torch.load(d / "xfakenet.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(d), meta


class Predictor:
    """Batch prediction + explanations for a saved XFakeNet model."""

    def __init__(self, model_dir: str | Path, batch_size: int = 64, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok, self.meta = load(model_dir, self.device)
        self.bs = batch_size

    @torch.no_grad()
    def _run(self, texts: Sequence[str]):
        for s in range(0, len(texts), self.bs):
            batch = list(texts[s:s + self.bs])
            enc, offsets, spans = encode(self.tok, batch, self.meta["max_len"], self.meta["max_segments"],
                                         self.device)
            with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                out = self.model(**enc)
            yield batch, enc, offsets, spans, out

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        res = [torch.softmax(o["logits"], -1).cpu().numpy() for *_, o in self._run(texts)]
        return np.concatenate(res) if res else np.zeros((0, len(self.meta["classes"])))

    def explain(self, texts: Sequence[str], threshold: float | None = None) -> list[dict]:
        thr = self.meta.get("highlight_threshold", 0.65) if threshold is None else threshold
        out = []
        for batch, enc, offsets, spans, o in self._run(texts):
            probs = torch.softmax(o["logits"], -1).cpu().numpy()
            tok_att = o["token_att"].float().cpu().numpy()
            seg_att = o["seg_att"].float().cpu().numpy()
            seg_ids = enc["seg_ids"].cpu().numpy()
            offs = offsets.tolist()
            for i, text in enumerate(batch):
                out.append(word_explanation(text, offs[i], seg_ids[i], tok_att[i], seg_att[i], spans[i],
                                            probs[i], self.meta["classes"], thr))
        return out

