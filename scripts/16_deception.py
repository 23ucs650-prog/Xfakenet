"""Week 4: six-category deception-pattern attribution for image-text pairs.

Stages (run in order; each is skipped if its inputs are missing):
  label : Gemini (multimodal) labels each pair with one of six deception patterns or
          "consistent". Pairs = all 389 real FakeNewsNet pairs + 200 shuffled (mismatched) pairs.
  train : a small classifier (logistic regression on CLIP text/image features + C-score)
          learns the six patterns from Gemini's labels; 5-fold cross-validated.
          This classifier is what the pipeline uses for CROSS-MODAL items (no API at run time).
  page  : builds results/week4/handcheck.html with 100 pairs for you to verify by hand
          (works offline in any browser; download your answers as CSV at the end).
  score : after you upload the CSV, measures Gemini's and the classifier's agreement with you.

Usage: python scripts/16_deception.py --stage all
       python scripts/16_deception.py --stage score
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xfakenet.common import load_config, p, set_seed, write_json  # noqa: E402

CATS = {
    "out_of_context": "Genuine photo, but from a different event, time or story than the headline",
    "misattributed_entity": "Shows a different person, group or organisation than the headline names",
    "wrong_location_or_event": "Shows a different place or kind of event than the headline describes",
    "exaggeration": "Image and headline differ sharply in drama or emotion (one sensationalises the other)",
    "manipulated_or_synthetic": "Looks edited, composited, a meme, a screenshot with added text, or AI-generated",
    "irrelevant_stock": "Generic stock or decorative image with no real connection to the specific claim",
    "consistent": "Image plausibly matches the headline (no deception pattern)",
}
PATTERNS = [c for c in CATS if c != "consistent"]

PROMPT = """You are annotating news headline + image pairs for research on image-text deception.
For each item, choose exactly ONE category:
{cats}

Judge only the relationship between the headline and the image (not whether the claim is true).
Return a JSON array, one object per item, same order:
[{{"i": <index>, "category": "<category key>", "confidence": <0-1>, "reason": "<max 15 words>"}}, ...]
"""


def items(cfg: dict) -> pd.DataFrame:
    """Real pairs + a sample of shuffled pairs (from the C-score step)."""
    dc = cfg["deception"]
    al = pd.read_csv(p(f"{cfg['paths']['processed']}/fnn_pairs.csv"))
    al = al[al["image_path"].map(lambda x: p(x).exists())]
    al = al.assign(pair_type="aligned")[["id", "title", "image_path", "fnn_label", "site", "pair_type"]]
    parts = [al]
    cp = p(f"{cfg['cscore']['out_dir']}/cscore_pairs.csv")
    if cp.exists():
        mis = pd.read_csv(cp)
        mis = mis[mis["pair_type"] == "mismatched"]
        mis = mis.sample(min(len(mis), dc["n_mismatched"]), random_state=cfg["seed"])
        parts.append(mis[["id", "title", "image_path", "fnn_label", "site", "pair_type"]])
    df = pd.concat(parts, ignore_index=True)
    df["key"] = df["id"].astype(str) + "|" + df["pair_type"]
    return df.reset_index(drop=True)


def jpeg_bytes(path, side: int) -> bytes:
    from PIL import Image
    im = Image.open(p(path)).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# ---------------- stage: label ----------------
def stage_label(cfg: dict, df: pd.DataFrame) -> None:
    dc = cfg["deception"]
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("[deception] no GEMINI_API_KEY in the environment; labelling skipped")
        return
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    out = p(dc["labels"])
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = df[~df["key"].isin(done)]
    print(f"[deception] {len(df)} pairs, {len(done)} labelled, {len(todo)} to go", flush=True)
    cats = "\n".join(f'- "{k}": {v}' for k, v in CATS.items())
    model = None
    gcfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    bs = dc["batch_size"]
    with open(out, "a") as f:
        for s in range(0, len(todo), bs):
            b = todo.iloc[s:s + bs]
            contents = [PROMPT.format(cats=cats)]
            for i, r in enumerate(b.itertuples(index=False)):
                contents += [f"Item {i}. Headline: {r.title}",
                             types.Part.from_bytes(data=jpeg_bytes(r.image_path, dc["image_side"]),
                                                   mime_type="image/jpeg")]
            got = None
            for m in ([model] if model else cfg["paraphrase"]["models"]):
                for attempt in range(5):
                    try:
                        resp = client.models.generate_content(model=m, contents=contents, config=gcfg)
                        got = {int(o["i"]): o for o in json.loads(resp.text)}
                        model = m
                        break
                    except Exception as e:
                        msg = str(e)
                        if "404" in msg or "NOT_FOUND" in msg or "402" in msg or "credits" in msg.lower():
                            print(f"[deception] {m}: {msg[:120]}", flush=True)
                            break
                        time.sleep(min(60, 5 * 2 ** attempt))
                if got is not None:
                    break
            if got is None:
                print(f"[deception] batch {s // bs} failed; will retry on the next run", flush=True)
                continue
            for i, r in enumerate(b.itertuples(index=False)):
                o = got.get(i, {})
                cat = str(o.get("category", "")).strip()
                f.write(json.dumps({"key": r.key, "category": cat if cat in CATS else "invalid",
                                    "confidence": o.get("confidence"), "reason": o.get("reason", ""),
                                    "model": model}) + "\n")
            f.flush()
            print(f"[deception] labelled {min(s + bs, len(todo))}/{len(todo)}", flush=True)
            time.sleep(dc["sleep_s"])


def load_labels(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    f = p(cfg["deception"]["labels"])
    if not f.exists():
        return pd.DataFrame()
    lab = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates("key", keep="last")
    return df.merge(lab, on="key", how="inner")


# ---------------- stage: train ----------------
def stage_train(cfg: dict, df: pd.DataFrame) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xfakenet.clip_utils import Clip, pair_features

    lab = load_labels(cfg, df)
    if not len(lab):
        print("[deception] no labels yet; training skipped")
        return {}
    res = {"n_labelled": len(lab),
           "label_counts": lab["category"].value_counts().to_dict(),
           "label_counts_by_pair_type": lab.groupby(["pair_type", "category"]).size().unstack(fill_value=0)
           .to_dict(orient="index")}

    feats_f = p(cfg["deception"]["features"])
    T, I = Clip(cfg["cscore"]["clip_model"]).embed(lab["title"].tolist(), [p(x) for x in lab["image_path"]])
    np.savez(feats_f, key=lab["key"].to_numpy(), T=T, I=I)
    X = pair_features(T, I)

    pat = lab["category"].isin(PATTERNS).to_numpy()
    y = lab.loc[pat, "category"].to_numpy()
    counts = pd.Series(y).value_counts()
    keep_cls = counts[counts >= cfg["deception"]["min_per_class"]].index.tolist()
    m = np.isin(y, keep_cls)
    Xp, yp = X[pat][m], y[m]
    res["classifier_classes"] = keep_cls
    res["dropped_rare_classes"] = counts[counts < cfg["deception"]["min_per_class"]].to_dict()
    if len(keep_cls) < 2:
        print("[deception] fewer than 2 pattern classes with enough examples; classifier skipped")
        write_json(res, f"{cfg['week4']['out_dir']}/deception_results.json")
        return res
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["deception"]["C"], max_iter=5000,
                                                             class_weight="balanced"))
    n_splits = int(min(5, counts[keep_cls].min()))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=cfg["seed"])
    oof = cross_val_predict(clf, Xp, yp, cv=cv)
    majority = pd.Series(yp).value_counts(normalize=True).iloc[0]
    res["classifier_cv"] = {"n": int(len(yp)), "folds": n_splits,
                            "accuracy": round(float(accuracy_score(yp, oof)), 4),
                            "macro_f1": round(float(f1_score(yp, oof, average="macro")), 4),
                            "majority_class_accuracy": round(float(majority), 4)}
    keys = lab.loc[pat, "key"].to_numpy()[m]
    pd.DataFrame({"key": keys, "gemini_category": yp, "oof_pred": oof}).to_csv(
        p(cfg["deception"]["oof"]), index=False)
    clf.fit(Xp, yp)
    joblib.dump({"model": clf, "classes": keep_cls}, p(cfg["deception"]["model"]))
    print(f"[deception] classifier CV: {res['classifier_cv']}", flush=True)
    write_json(res, f"{cfg['week4']['out_dir']}/deception_results.json")
    return res


# ---------------- stage: page ----------------
PAGE_CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1c1c1a;--muted:#66655f;--line:#e2dfd6;--acc:#b45309;--ok:#2f7d4f;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#21201d;--ink:#eceae4;--muted:#a3a198;--line:#34332e;--acc:#f59e0b;--ok:#6cc58f;--bad:#f2867e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
.wrap{max-width:860px;margin:0 auto}h1{font-size:20px;margin:0}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:8px}
.prog{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}.prog i{display:block;height:100%;background:var(--acc)}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer}
main{padding:16px}details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:16px}
details li{margin:2px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px;
display:grid;grid-template-columns:220px 1fr;gap:14px}.card.done{border-color:var(--ok)}
.card img{width:100%;border-radius:6px;object-fit:cover;max-height:220px}
.n{font-size:12px;color:var(--muted)}.hl{font-weight:600;margin:2px 0 8px}.gem{font-size:14px;margin-bottom:8px}
.gem b{color:var(--acc)}.why{color:var(--muted);font-size:13px}label{margin-right:14px;cursor:pointer}
select{font:inherit;margin-top:6px;padding:4px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
@media (max-width:600px){.card{grid-template-columns:1fr}}
"""

PAGE_JS = """
const KEY='xfn-handcheck-v1';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){S={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}upd()}
function upd(){const n=ITEMS.filter(it=>S[it.key]&&(S[it.key].v==='correct'||(S[it.key].v==='wrong'&&S[it.key].c))).length;
document.getElementById('cnt').textContent=n+' / '+ITEMS.length+' checked';document.getElementById('pbar').style.width=(100*n/ITEMS.length)+'%';
ITEMS.forEach(it=>{const el=document.getElementById('c-'+it.idx);const s=S[it.key];el.classList.toggle('done',!!s&&(s.v==='correct'||(s.v==='wrong'&&!!s.c)))})}
function setV(k,v){S[k]=S[k]||{};S[k].v=v;if(v==='correct')S[k].c='';document.getElementById('s-'+k).disabled=(v!=='wrong');save()}
function setC(k,c){S[k]=S[k]||{v:'wrong'};S[k].c=c;save()}
function dl(){const rows=[['key','gemini_category','verdict','human_category']];
ITEMS.forEach(it=>{const s=S[it.key]||{};rows.push([it.key,it.gem,s.v||'',s.v==='correct'?it.gem:(s.c||'')])});
const csv=rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='handcheck_results.csv';a.click()}
window.addEventListener('DOMContentLoaded',()=>{ITEMS.forEach(it=>{const s=S[it.key];if(!s)return;
const r=document.querySelector('input[name="v-'+it.idx+'"][value="'+s.v+'"]');if(r)r.checked=true;
const sel=document.getElementById('s-'+it.key);sel.disabled=s.v!=='wrong';if(s.c)sel.value=s.c});upd()});
"""


def stage_page(cfg: dict, df: pd.DataFrame) -> None:
    lab = load_labels(cfg, df)
    lab = lab[lab["category"].isin(CATS)]
    if not len(lab):
        print("[deception] no labels yet; hand-check page skipped")
        return
    n = cfg["deception"]["handcheck_n"]
    rng = np.random.default_rng(cfg["seed"])
    # at least a few of every category, the rest proportional
    picks = []
    for c, g in lab.groupby("category"):
        picks += g.sample(min(len(g), 5), random_state=cfg["seed"]).index.tolist()
    rest = lab.drop(picks)
    picks += rest.sample(min(len(rest), max(0, n - len(picks))), random_state=cfg["seed"]).index.tolist()
    sample = lab.loc[picks].sample(frac=1.0, random_state=cfg["seed"]).head(n).reset_index(drop=True)
    sample[["key", "category"]].to_csv(p(cfg["deception"]["handcheck_sample"]), index=False)

    js_items, cards = [], []
    opts = "".join(f'<option value="{k}">{k.replace("_", " ")}</option>' for k in CATS)
    for i, r in sample.iterrows():
        b64 = base64.b64encode(jpeg_bytes(r.image_path, 360)).decode()
        k = html.escape(r.key, quote=True)
        js_items.append({"idx": i, "key": r.key, "gem": r.category})
        cards.append(
            f'<div class="card" id="c-{i}"><img alt="" src="data:image/jpeg;base64,{b64}">'
            f'<div><div class="n">#{i + 1}</div><div class="hl">{html.escape(str(r.title))}</div>'
            f'<div class="gem">Gemini: <b>{r.category.replace("_", " ")}</b> '
            f'<span class="why">— {html.escape(str(r.reason))}</span></div>'
            f'<label><input type="radio" name="v-{i}" value="correct" onchange="setV(\'{k}\',\'correct\')"> Correct</label>'
            f'<label><input type="radio" name="v-{i}" value="wrong" onchange="setV(\'{k}\',\'wrong\')"> Wrong, should be:</label>'
            f'<br><select id="s-{k}" disabled onchange="setC(\'{k}\',this.value)"><option value="">choose…</option>{opts}</select>'
            f'</div></div>')
    legend = "".join(f"<li><b>{k.replace('_', ' ')}</b>: {html.escape(v)}</li>" for k, v in CATS.items())
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Deception hand-check</title><style>{PAGE_CSS}</style>"
            f"</head><body><header><div class='wrap'><h1>Hand-check Gemini's deception labels</h1><div class='bar'>"
            f"<div class='prog'><i id='pbar' style='width:0'></i></div><span id='cnt'></span>"
            f"<button onclick='dl()'>Download results (CSV)</button></div></div></header><main class='wrap'>"
            f"<details open><summary><b>How to check</b> (your answers are saved in this browser as you go)</summary>"
            f"<p>For each pair, look at the headline and image and decide whether Gemini's category is right. "
            f"If not, pick the right one. When finished, click <b>Download results (CSV)</b> and upload that file "
            f"in the Week 4 notebook.</p><ul>{legend}</ul></details>{''.join(cards)}</main>"
            f"<script>const ITEMS={json.dumps(js_items)};{PAGE_JS}</script></body></html>")
    out = p(f"{cfg['week4']['out_dir']}/handcheck.html")
    out.write_text(page, encoding="utf-8")
    print(f"[deception] hand-check page: {out} ({len(sample)} pairs)")


# ---------------- stage: score ----------------
def stage_score(cfg: dict) -> None:
    from sklearn.metrics import cohen_kappa_score
    f = p(cfg["deception"]["handcheck_results"])
    if not f.exists():
        print("[deception] no hand-check results uploaded yet; scoring skipped")
        return
    h = pd.read_csv(f)
    h = h[h["verdict"].isin(["correct", "wrong"]) & h["human_category"].fillna("").astype(str).str.len().gt(0)]
    resf = p(f"{cfg['week4']['out_dir']}/deception_results.json")
    res = json.load(open(resf)) if resf.exists() else {}
    hc = {"n_checked": int(len(h)),
          "gemini_accuracy": round(float((h["verdict"] == "correct").mean()), 4) if len(h) else None,
          "gemini_kappa": round(float(cohen_kappa_score(h["gemini_category"], h["human_category"])), 4)
          if len(h) > 1 else None,
          "pattern_vs_consistent_agreement": round(float(
              ((h["gemini_category"] == "consistent") == (h["human_category"] == "consistent")).mean()), 4)
          if len(h) else None}
    oof = p(cfg["deception"]["oof"])
    if oof.exists():
        o = pd.read_csv(oof).merge(h[["key", "human_category"]], on="key")
        if len(o):
            hc["classifier_accuracy_vs_human"] = round(float((o["oof_pred"] == o["human_category"]).mean()), 4)
            hc["classifier_n"] = int(len(o))
    res["handcheck"] = hc
    write_json(res, resf)
    print(f"[deception] hand-check: {hc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "label", "train", "page", "score"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    df = items(cfg)
    if args.stage in ("all", "label"):
        stage_label(cfg, df)
    if args.stage in ("all", "train"):
        stage_train(cfg, df)
    if args.stage in ("all", "page"):
        stage_page(cfg, df)
    if args.stage in ("all", "score"):
        stage_score(cfg)


if __name__ == "__main__":
    main()
