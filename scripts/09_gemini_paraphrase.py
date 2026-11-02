"""Generate LLM paraphrases of LIAR statements with the Gemini API (augmentation data).

- train: a class-stratified sample of the TRAIN split, sized so paraphrases are
  15% of the final corpus (70% original / 15% TextFooler / 15% paraphrase).
- test : a separate sample of the TEST split, used only in week 4 to measure
  robustness to paraphrased text. Never used for training.

Each paraphrase keeps the original gold label. A paraphrase is rejected if it
changes any number, is identical to the original, or is far longer/shorter.

API key: Colab secret GEMINI_API_KEY, or environment variable GEMINI_API_KEY.
Free key: https://aistudio.google.com/apikey

Resumable: ids already in the output file are skipped.

Usage: python scripts/09_gemini_paraphrase.py --split train
       python scripts/09_gemini_paraphrase.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import clean_text, load_config, p  # noqa: E402

PROMPT = """You rewrite short political statements for a research dataset on robust fake-news detection.

Paraphrase each statement below so it reads like a different person said it:
- Keep exactly the same meaning, claim, people, places and every number.
- Do not correct, soften, exaggerate or comment on the claim, even if it is false.
- Change wording and sentence structure; keep roughly the same length.

Return a JSON array with one object per statement, in the same order:
[{{"i": <index>, "paraphrase": "<text>"}}, ...]

Statements:
{items}"""

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(s: str) -> list[str]:
    return sorted(x.replace(",", "") for x in NUM.findall(s))


def check(orig: str, para: str) -> str | None:
    """Return a rejection reason, or None if the paraphrase is acceptable."""
    if not para or not para.strip():
        return "empty"
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    if norm(orig) == norm(para):
        return "identical"
    if numbers(orig) != numbers(para):
        return "numbers_changed"
    r = len(para.split()) / max(1, len(orig.split()))
    if r < 0.5 or r > 2.0:
        return "length"
    return None


def get_key() -> str:
    # The notebook puts the key in the environment; Colab's secret API only works in the kernel itself.
    k = os.environ.get("GEMINI_API_KEY", "")
    if not k:
        sys.exit("[paraphrase] No GEMINI_API_KEY in the environment. Run the key cell in the notebook first.")
    return k


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    frac = min(1.0, n / len(df))
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).head(n)


class Gemini:
    def __init__(self, models: list[str], temperature: float):
        from google import genai
        from google.genai import types
        self.client, self.types = genai.Client(api_key=get_key()), types
        self.models, self.temperature, self.model = models, temperature, None

    def call(self, prompt: str) -> str:
        cfg = self.types.GenerateContentConfig(response_mime_type="application/json",
                                               temperature=self.temperature)
        candidates = [self.model] if self.model else self.models
        last = None
        for m in candidates:
            for attempt in range(6):
                try:
                    r = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                    if self.model is None:
                        self.model = m
                        print(f"[paraphrase] using model {m}", flush=True)
                    return r.text
                except Exception as e:
                    last, msg = e, str(e)
                    if "404" in msg or "not found" in msg.lower() or "NOT_FOUND" in msg:
                        break  # try the next model name
                    wait = min(60, 5 * 2 ** attempt)
                    print(f"[paraphrase] {m}: {msg[:120]} ... retry in {wait}s", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"All Gemini calls failed: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    pc = cfg["paraphrase"]
    df = pd.read_csv(p(f"{cfg['paths']['processed']}/corpus_{args.split}.csv"))
    if args.n:
        n = args.n
    elif args.split == "train":
        o = cfg["augmentation"]
        n = round(len(df) * o["paraphrase_share"] / o["original_share"])
    else:
        n = pc["test_n"]
    sub = stratified(df, n, cfg["seed"])

    out = p(pc["out_train"] if args.split == "train" else pc["out_test"])
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in open(out)}
    todo = sub[~sub["id"].astype(str).isin(done)]
    print(f"[paraphrase] split={args.split} target={len(sub)} done={len(done)} todo={len(todo)}", flush=True)

    gem = Gemini(pc["models"], pc["temperature"])
    stats = {"ok": 0}
    bs = pc["batch_size"]
    with open(out, "a") as f:
        for s in range(0, len(todo), bs):
            b = todo.iloc[s:s + bs]
            items = "\n".join(f"{i}. {t}" for i, t in enumerate(b["text"]))
            try:
                raw = gem.call(PROMPT.format(items=items))
                got = {int(o["i"]): clean_text(o.get("paraphrase", "")) for o in json.loads(raw)}
            except Exception as e:
                print(f"[paraphrase] batch {s // bs} failed: {e!r}"[:200], flush=True)
                continue  # not written -> retried on the next run
            for i, row in enumerate(b.itertuples(index=False)):
                para = got.get(i, "")
                why = check(row.text, para)
                rec = {"id": str(row.id), "label": row.label, "label_id": int(row.label_id),
                       "source": row.source, "original": row.text, "text": para,
                       "status": "ok" if why is None else f"rejected:{why}", "model": gem.model}
                stats[rec["status"]] = stats.get(rec["status"], 0) + 1
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[paraphrase] {min(s + bs, len(todo))}/{len(todo)} {stats}", flush=True)
            time.sleep(pc["sleep_s"])
    print(f"[paraphrase] finished {stats}")


if __name__ == "__main__":
    main()
