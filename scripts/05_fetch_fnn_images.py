"""Download FakeNewsNet article images and count usable headline-image pairs.

For each sampled article we fetch the page (live, then the Wayback Machine if the
live link is dead), take og:image / twitter:image (or the first large <img>),
download it and check it opens and is at least min_image_side px on each side.

Resumable: already-processed ids in the manifest are skipped, so you can
kill and restart it freely.

Input : data/raw/FakeNewsNet/dataset/{politifact,gossipcop}_{fake,real}.csv
Output: data/images/<id>.jpg
        data/processed/fnn_manifest.jsonl     (one row per article, with status)
        data/processed/fnn_pairs.csv          (usable pairs only: id,title,label,image_path,source_url)
        data/processed/fnn_image_summary.json (the counts for the week-1 report)

Usage : python scripts/05_fetch_fnn_images.py [--limit N] [--summary-only]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import clean_text, load_config, p, write_json  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) XFakeNet-research/0.1"}
_lock = threading.Lock()


def load_fnn(d: Path) -> pd.DataFrame:
    frames = []
    for site in ["politifact", "gossipcop"]:
        for lab in ["fake", "real"]:
            f = d / f"{site}_{lab}.csv"
            if f.exists():
                df = pd.read_csv(f, dtype=str, keep_default_na=False)
                df["site"], df["fnn_label"] = site, lab
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No FakeNewsNet CSVs in {d}. Run scripts/01_download_data.sh first.")
    df = pd.concat(frames, ignore_index=True)
    df["title"] = df["title"].map(clean_text)
    df = df[(df["news_url"].str.len() > 0) & (df["title"].str.len() > 0)]
    return df.drop_duplicates(subset=["id"])


def sample_balanced(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced over fake/real and over sites, as far as the data allows."""
    groups = df.groupby(["site", "fnn_label"])
    per = max(1, n // groups.ngroups)
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in groups]
    out = pd.concat(parts)
    if len(out) < n:  # top up from whatever is left
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def norm_url(u: str) -> str:
    u = u.strip()
    return u if u.startswith(("http://", "https://")) else "http://" + u


def get(url: str, timeout: int) -> requests.Response | None:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def wayback(url: str, timeout: int) -> str | None:
    r = get(f"https://archive.org/wayback/available?url={url}&timestamp=20180101", timeout)
    if r is None:
        return None
    try:
        snap = r.json()["archived_snapshots"]["closest"]
        return snap["url"] if snap.get("available") else None
    except (KeyError, ValueError):
        return None


def find_image_url(html: str, base: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for attr, key in [("property", "og:image"), ("name", "twitter:image"),
                      ("property", "og:image:url"), ("name", "twitter:image:src")]:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            return urljoin(base, tag["content"].strip())
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if any(s in src.lower() for s in ("logo", "icon", "sprite", "pixel", ".svg", "avatar")):
            continue
        return urljoin(base, src)
    return None


def process(row: dict, cfg: dict) -> dict:
    from PIL import Image  # local import keeps worker startup cheap

    fc = cfg["fakenewsnet"]
    url = norm_url(row["news_url"])
    rec = {"id": row["id"], "title": row["title"], "fnn_label": row["fnn_label"],
           "site": row["site"], "news_url": url, "status": None, "via": None}

    page, via = get(url, fc["timeout"]), "live"
    if page is None and fc["use_wayback"]:
        snap = wayback(url, fc["timeout"])
        page, via = (get(snap, fc["timeout"]), "wayback") if snap else (None, None)
    if page is None:
        rec["status"] = "page_dead"
        return rec
    rec["via"] = via

    img_url = find_image_url(page.text, page.url)
    if not img_url:
        rec["status"] = "no_image_tag"
        return rec
    rec["image_url"] = img_url

    r = get(img_url, fc["timeout"])
    if r is None and via == "live" and fc["use_wayback"]:
        snap = wayback(img_url, fc["timeout"])
        r = get(snap, fc["timeout"]) if snap else None
    if r is None:
        rec["status"] = "image_dead"
        return rec
    try:
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        rec["status"] = "image_corrupt"
        return rec
    if min(im.size) < fc["min_image_side"]:
        rec["status"] = "image_too_small"
        return rec
    out = p(f"{cfg['paths']['images']}/{row['id']}.jpg")
    im.save(out, "JPEG", quality=90)
    rec.update(status="ok", image_path=str(out.relative_to(p("."))), width=im.size[0], height=im.size[1])
    return rec


def summarise(cfg: dict) -> dict:
    d = cfg["paths"]["processed"]
    man = p(f"{d}/fnn_manifest.jsonl")
    rows = [json.loads(line) for line in open(man)] if man.exists() else []
    if not rows:
        print("[fnn] manifest empty")
        return {}
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    ok["label"] = ok["fnn_label"].map({"fake": "Fabricated", "real": "Credible"})
    ok[["id", "title", "fnn_label", "label", "site", "image_path", "news_url"]] \
        .to_csv(p(f"{d}/fnn_pairs.csv"), index=False)
    s = {
        "attempted": len(df),
        "usable_pairs": len(ok),
        "usable_rate": round(len(ok) / len(df), 3),
        "status_counts": df["status"].value_counts().to_dict(),
        "usable_by_label": ok["fnn_label"].value_counts().to_dict(),
        "usable_by_site": ok["site"].value_counts().to_dict(),
        "usable_via": ok["via"].value_counts().to_dict(),
    }
    write_json(s, f"{d}/fnn_image_summary.json")
    print(json.dumps(s, indent=2))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="override sample_size")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.summary_only:
        summarise(cfg)
        return

    df = load_fnn(p(cfg["fakenewsnet"]["dir"]))
    print(f"[fnn] {len(df)} articles with URL+title | "
          f"{df.groupby(['site','fnn_label']).size().to_dict()}")
    sample = sample_balanced(df, args.limit or cfg["fakenewsnet"]["sample_size"], cfg["seed"])

    man = p(f"{cfg['paths']['processed']}/fnn_manifest.jsonl")
    done = {json.loads(line)["id"] for line in open(man)} if man.exists() else set()
    todo = [r for r in sample.to_dict("records") if r["id"] not in done]
    print(f"[fnn] sample={len(sample)} already_done={len(done)} todo={len(todo)}")

    n = 0
    with ThreadPoolExecutor(cfg["fakenewsnet"]["workers"]) as ex, open(man, "a") as fout:
        futs = [ex.submit(process, r, cfg) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with _lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            n += 1
            if n % 50 == 0:
                print(f"[fnn] {n}/{len(todo)}", flush=True)
    summarise(cfg)


if __name__ == "__main__":
    main()
