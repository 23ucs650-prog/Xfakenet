"""XFakeNet demo on Hugging Face Spaces (CPU). Models are uploaded by notebooks/XFakeNet_Deploy_HF_Space.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from xfakenet.demo import build_app, launch  # noqa: E402
from xfakenet.pipeline import XFakeNetPipeline  # noqa: E402

EXAMPLES = [
    ["Says the unemployment rate doubled under the governor's watch.", None],
    ["You can buy lobster with food stamps.", None],
    ["There is no clinical evidence that vaccinating healthcare workers protects patients.", None],
    ["Virginia ranked near the bottom of the nation 50 years ago in per capita income but is in the top 10 today.", None],
]

pipe = XFakeNetPipeline.load()
pipe("Warm-up statement.")   # load the model before the first visitor
launch(build_app(pipe, EXAMPLES))
