"""CLIP ViT-B/32 helpers shared by the C-score, modality flag, deception classifier and pipeline."""
from __future__ import annotations

import numpy as np


class Clip:
    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def embed(self, titles: list[str], images: list, batch_size: int = 64):
        """Normalised (text, image) embeddings. `images` are PIL images or file paths."""
        from PIL import Image
        T, I = [], []
        for s in range(0, len(titles), batch_size):
            ims = [Image.open(x).convert("RGB") if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__")
                   else x.convert("RGB") for x in images[s:s + batch_size]]
            inp = self.proc(text=list(titles[s:s + batch_size]), images=ims, return_tensors="pt",
                            padding=True, truncation=True, max_length=77).to(self.device)
            with self.torch.no_grad():
                out = self.model(**inp)
            T.append(self.torch.nn.functional.normalize(out.text_embeds, dim=-1).cpu().numpy())
            I.append(self.torch.nn.functional.normalize(out.image_embeds, dim=-1).cpu().numpy())
        return np.concatenate(T), np.concatenate(I)


def pair_features(T: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Features for the deception-pattern classifier: text, image, their product, and the C-score."""
    c = 1.0 - (T * I).sum(1, keepdims=True)
    return np.hstack([T, I, T * I, c])
