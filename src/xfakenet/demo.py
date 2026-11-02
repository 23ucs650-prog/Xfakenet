"""Gradio demo for XFakeNet.

    from xfakenet.demo import build_app
    build_app(XFakeNetPipeline.load(), examples).launch(share=True)

Every colour is set inline on every element, so Gradio's theme (light or dark) cannot
override it, and the app forces Gradio's light theme. The render functions are plain
Python (no Gradio) so they can be tested anywhere.
"""
from __future__ import annotations

import html

INK, MUTED, FAINT, LINE, CARD = "#1c1917", "#57534e", "#a8a29e", "#e7e5e4", "#ffffff"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
CLASS_INFO = {
    "Fabricated": ("#b91c1c", "#fee2e2", "False claim with no factual basis"),
    "Misleading": ("#c2410c", "#ffedd5", "Contains some truth but distorts or omits context"),
    "Credible": ("#15803d", "#dcfce7", "Accurate or substantially accurate"),
}
FLAG_INFO = {
    "TEXT": ("#44403c", "#f5f5f4", "The evidence comes mainly from the text."),
    "IMAGE": ("#075985", "#e0f2fe", "The image carries more of the evidence than the text."),
    "CROSS-MODAL": ("#92400e", "#fef3c7", "The image and the headline contradict each other."),
}
PATTERN_INFO = {
    "out_of_context": "Genuine photo from a different event, time or story",
    "misattributed_entity": "Image shows a different person or organisation than named",
    "wrong_location_or_event": "Image shows a different place or kind of event",
    "exaggeration": "Image and headline differ sharply in drama or emotion",
    "manipulated_or_synthetic": "Image looks edited, composited, a meme or AI-generated",
    "irrelevant_stock": "Generic stock image with no real link to the claim",
}


def _e(s) -> str:
    return html.escape(str(s))


def _card(inner: str) -> str:
    return (f'<div style="font-family:{FONT};color:{INK};background:{CARD};border:1px solid {LINE};'
            f'border-radius:14px;padding:20px 22px;box-shadow:0 1px 2px rgba(0,0,0,.04)">{inner}</div>')


def _label(text: str, top: int = 0) -> str:
    return (f'<div style="font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;'
            f'color:{MUTED};margin:{top}px 0 8px">{_e(text)}</div>')


def render_verdict(out: dict) -> str:
    color, tint, desc = CLASS_INFO.get(out["label"], (INK, "#f5f5f4", ""))
    rows = ""
    for k, v in sorted(out["probs"].items(), key=lambda kv: -kv[1]):
        c = CLASS_INFO.get(k, (FAINT, "", ""))[0]
        bold = "700" if k == out["label"] else "500"
        rows += (f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0">'
                 f'<span style="width:92px;font-size:14px;font-weight:{bold};color:{INK}">{_e(k)}</span>'
                 f'<span style="flex:1;height:10px;background:#f0eeeb;border-radius:5px;overflow:hidden;display:block">'
                 f'<span style="display:block;height:100%;width:{v*100:.1f}%;background:{c};border-radius:5px"></span></span>'
                 f'<span style="width:44px;text-align:right;font-size:14px;font-weight:{bold};color:{INK}">{v*100:.0f}%</span></div>')
    head = (f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">'
            f'<span style="font-size:28px;font-weight:700;color:{color};line-height:1.2">{_e(out["label"])}</span>'
            f'<span style="font-size:12px;font-weight:600;color:{color};background:{tint};padding:3px 10px;'
            f'border-radius:999px">{out["probs"][out["label"]]*100:.0f}% confident</span></div>')
    return _card(_label("Verdict") + head +
                 f'<div style="font-size:14px;color:{MUTED};margin-bottom:14px">{_e(desc)}</div>' + rows)


def render_modality(out: dict) -> str:
    flag = out["modality_flag"]
    fc, ft, fdesc = FLAG_INFO.get(flag, (INK, "#f5f5f4", ""))
    pill = (f'<span style="display:inline-block;font-size:13px;font-weight:700;color:{fc};background:{ft};'
            f'padding:4px 12px;border-radius:999px;letter-spacing:.03em">{_e(flag)}</span>')
    if out.get("cscore") is None:
        body = (f'<div style="font-size:14px;color:{MUTED};margin-top:10px">No image given, so the analysis is text '
                f'only. Add an image to check whether it fits the headline.</div>')
    else:
        c, t = out["cscore"], out["cscore_threshold"]
        stat = lambda name, val, col=INK: (  # noqa: E731
            f'<div><div style="font-size:12px;color:{MUTED}">{_e(name)}</div>'
            f'<div style="font-size:22px;font-weight:700;color:{col}">{_e(val)}</div></div>')
        verdict = "Contradict" if c > t else "Consistent"
        body = (f'<div style="font-size:14px;color:{MUTED};margin-top:10px">{_e(fdesc)}</div>'
                f'<div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:14px">'
                f'{stat("C-score", f"{c:.3f}")}{stat("Threshold", f"{t:.3f}")}'
                f'{stat("Image vs headline", verdict, "#92400e" if c > t else "#15803d")}</div>')
        pat = out.get("deception_pattern")
        if pat:
            body += (_label("Likely deception pattern", 18) +
                     f'<div style="font-size:15px;color:{INK}"><span style="font-weight:700;color:{INK}">'
                     f'{_e(pat.replace("_", " ").capitalize())}</span>'
                     f'<span style="color:{MUTED}"> · {_e(PATTERN_INFO.get(pat, ""))}</span></div>')
    return _card(_label("Modality") + pill + body)


def render_explanation(out: dict) -> str:
    ws = out.get("word_scores", [])
    top = max([w["shap"] for w in ws if w["shap"] > 0], default=0) or 1.0
    hl = set(out.get("highlighted", []))
    chips = []
    for w in ws:
        a = max(0.0, w["shap"] / top)
        is_hl = w["w"] in hl
        style = (f'display:inline-block;margin:3px 2px;padding:2px 7px;border-radius:6px;font-size:16px;'
                 f'line-height:1.6;color:{INK};background:rgba(234,88,12,{a*0.45:.2f});'
                 f'border:1.5px solid {"#c2410c" if is_hl else "transparent"};font-weight:{"700" if is_hl else "400"}')
        chips.append(f'<span title="SHAP {w["shap"]:+.3f}" style="{style}">{_e(w["w"])}</span>')
    att = ", ".join(_e(w) for w in out.get("attention_highlighted", [])) or "none"
    seg = (f'<div style="font-size:16px;color:{INK};background:#fff7ed;border-left:4px solid #ea580c;'
           f'padding:10px 14px;border-radius:8px">{_e(out["top_segment"])}</div>')
    note = (f'<div style="font-size:12.5px;color:{MUTED};margin-top:14px;line-height:1.5">Darker = pushed harder toward '
            f'the verdict (SHAP). Outlined words are highlighted (at least 65% of the strongest). For comparison, the '
            f'model\'s built-in attention highlighted: {att}.</div>')
    return _card(_label("Why: words that pushed the model to this verdict") + f'<div>{"".join(chips)}</div>' +
                 _label("Most suspicious part", 18) + seg + note)


def render_placeholder(msg: str) -> str:
    return _card(f'<div style="font-size:14px;color:{MUTED}">{_e(msg)}</div>')


FORCE_LIGHT = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""


def build_app(pipe, examples: list | None = None):
    import gradio as gr

    def run(text, image):
        text = (text or "").strip()
        if not text:
            return render_placeholder("Type a statement or headline first."), "", ""
        out = pipe(text, image=image)
        return render_verdict(out), render_explanation(out), render_modality(out)

    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="stone",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    look = {"theme": theme, "js": FORCE_LIGHT}
    new_api = int(gr.__version__.split(".")[0]) >= 6      # Gradio 6 moved theme/js from Blocks() to launch()
    with gr.Blocks(title="XFakeNet", **({} if new_api else look)) as app:
        gr.HTML(f'<div style="font-family:{FONT};padding:8px 2px 4px">'
                f'<div style="font-size:30px;font-weight:800;color:{INK}">XFakeNet</div>'
                f'<div style="font-size:15px;color:{MUTED};margin-top:4px">Explainable fake-news detection: a verdict, '
                f'the words behind it, and whether the image fits the headline. Research prototype trained on LIAR '
                f'(US political statements), not a fact-checker.</div></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                text = gr.Textbox(label="Statement or headline", lines=3,
                                  placeholder="e.g. Says the unemployment rate doubled under the governor.")
                image = gr.Image(label="Image (optional)", type="pil", height=260)
                btn = gr.Button("Analyse", variant="primary", size="lg")
            with gr.Column(scale=6):
                verdict = gr.HTML(render_placeholder("The verdict will appear here."))
                modality = gr.HTML()
        explanation = gr.HTML()
        btn.click(run, [text, image], [verdict, explanation, modality])
        text.submit(run, [text, image], [verdict, explanation, modality])
        if examples:
            gr.Examples(examples=examples, inputs=[text, image], label="Examples (click one, then Analyse)")
    app.xfn_launch_kwargs = look if new_api else {}
    return app


def launch(app, **kwargs):
    """app.launch() with the theme passed where this Gradio version expects it."""
    extra = getattr(app, "xfn_launch_kwargs", {})
    try:
        return app.launch(**extra, **kwargs)
    except TypeError:          # a Gradio that does not take theme/js in launch()
        return app.launch(**kwargs)
