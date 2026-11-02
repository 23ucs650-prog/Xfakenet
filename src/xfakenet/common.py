"""Shared helpers: config loading, label mapping, seeding, text cleanup."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    return load_yaml("configs/week1.yaml")


def load_mapping() -> dict:
    return load_yaml("configs/label_mapping.yaml")


def class_names() -> list[str]:
    return load_mapping()["classes"]


def label2id() -> dict[str, int]:
    return {c: i for i, c in enumerate(class_names())}


def p(rel: str | Path) -> Path:
    """Resolve a repo-relative path."""
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return _WS.sub(" ", s).strip()


def truncate_words(text: str, max_words: int) -> str:
    """Keep whole leading sentences up to max_words; hard-cut if the first sentence is longer."""
    text = clean_text(text)
    out, n = [], 0
    for sent in _SENT.split(text):
        w = sent.split()
        if n + len(w) > max_words:
            if not out:
                out.append(" ".join(w[:max_words]))
            break
        out.append(sent)
        n += len(w)
    return " ".join(out)


def write_json(obj, path: str | Path) -> None:
    with open(p(path), "w") as f:
        json.dump(obj, f, indent=2, default=str)
