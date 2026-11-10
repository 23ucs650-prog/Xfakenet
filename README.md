# XFakeNet: a cross-modal explainable fake news detector



XFakeNet classifies a statement as **Fabricated, Misleading or Credible**, shows **which words and which clause** drove
the verdict, and, given an image, checks whether the **image contradicts the headline**.



## Results at a glance

LIAR test set (1,283 statements), mean ± std over 3 random seeds. Full tables: [`results/results_tables.md`](results/results_tables.md).

| Model | Macro-F1 (clean) | Accuracy under TextFooler attack | Attack success rate |
|---|---|---|---|
| Baseline (RoBERTa-base) | 0.473 ± 0.007 | 0.068 ± 0.006 | 0.857 ± 0.011 |
| XFakeNet, no augmentation | 0.460 ± 0.015 | 0.057 ± 0.015 | 0.880 ± 0.027 |
| **XFakeNet v2 (augmented)** | 0.463 ± 0.003 | **0.124 ± 0.028** | **0.730 ± 0.062** |

- **Explanations are faithful:** removing the top 3 SHAP words drops the predicted-class probability by 0.264 vs 0.044
  for random words (beats random on 88% of statements). The model's built-in attention is not faithful (0.073, 53%).
  See [`results/faithful_explanations.md`](results/faithful_explanations.md).
- **Image–headline contradiction (C-score):** separates genuine from swapped image pairs with ROC-AUC 0.940 at a
  calibrated threshold of 0.753; it does not by itself separate fake from real FakeNewsNet articles (AUC 0.43).
- **Augmentation** (15% TextFooler + 15% Gemini paraphrases) roughly doubles accuracy under attack, consistently across seeds.
- **Hyperparameter grid search** (encoder LR {1,2,3}e-5 × batch size {16,32} × dropout {0.1,0.3} × up to 4 epochs,
  selected on validation only): all 12 configurations lie within 0.018 validation macro-F1; the default settings rank
  2nd of 12. Retrained with 3 seeds, the best configuration does not beat the defaults on clean test data
  (macro-F1 0.451 ± 0.017 vs 0.463 ± 0.003) but is slightly more robust (accuracy under attack 0.138 ± 0.005 vs
  0.124 ± 0.028), so the defaults were kept. See [`results/grid/grid_summary.md`](results/grid/grid_summary.md) and
  [`results/grid/tuned_vs_default.md`](results/grid/tuned_vs_default.md).

## How it works

1. **Module 3, classifier:** RoBERTa-base with jointly trained token and clause attention, class-weighted loss,
   trained on 70% LIAR + 15% TextFooler + 15% paraphrased statements.
2. **Module 1, explanations:** SHAP values per word (permutation sampling) and exact Shapley values per clause;
   words ≥ 0.65 of the strongest are highlighted.
3. **Module 2, contradiction:** CLIP ViT-B/32, C = 1 − cos(text, image), threshold 0.753; a modality flag
   (TEXT / IMAGE / CROSS-MODAL) and a deception-pattern classifier for contradicting pairs.

`src/xfakenet/pipeline.py` returns all of this for one input:

```python
from xfakenet.pipeline import XFakeNetPipeline
pipe = XFakeNetPipeline.load()                       # needs the trained models in runs/
out = pipe("Says the unemployment rate doubled under the governor.", image="photo.jpg")  # image optional
out["label"], out["highlighted"], out["top_segment"], out["modality_flag"], out["cscore"]
```

## Reproduce

Everything was run on Google Colab (free T4 GPU) with data saved to Google Drive. Run the notebooks in order;
each step is resumable, so if Colab disconnects, `Runtime → Run all` again continues where it stopped.

| Notebook | What it does | Time on T4 |
|---|---|---|
| `notebooks/XFakeNet_Week1_Colab.ipynb` | Download LIAR and FakeNewsNet, 3-class mapping, image crawl, baseline | ~1.5 h |
| `notebooks/XFakeNet_Week2_Colab.ipynb` | CLIP C-score + calibration, Gemini paraphrases, model v1, TextFooler | ~2 h |
| `notebooks/XFakeNet_Week3_Colab.ipynb` | Augmented training, XFakeNet models, robustness, explanations, modality flag | ~1 h |
| `notebooks/XFakeNet_Week4_Colab.ipynb` | Faithfulness test, deception patterns, pipeline, 3 seeds, final tables | ~2.5 h |
| `notebooks/XFakeNet_Week4b_Explanations.ipynb` | SHAP vs attention faithfulness; pipeline switched to SHAP | ~15 min |
| `notebooks/XFakeNet_GridSearch_Colab.ipynb` | Grid search (12 configs) + best config with 3 seeds and TextFooler attack | ~3.5 h |
| `notebooks/XFakeNet_Demo.ipynb` | Interactive Gradio demo with a temporary public link | ~3 min |
| `notebooks/XFakeNet_Deploy_HF_Space.ipynb` | Publishes the demo as a permanent Hugging Face Space | ~10 min |

Paraphrasing needs a free Gemini API key (Colab secret `GEMINI_API_KEY`); deployment needs a Hugging Face write
token (`HF_TOKEN`).

## Repository layout

```
configs/        week1.yaml (all settings), label_mapping.yaml (LIAR -> 3 classes)
docs/           label_mapping.md (why 3 classes)
scripts/        01-19, numbered in run order (data, training, attacks, evaluation, tables)
src/xfakenet/   model, segmentation, TextFooler, SHAP attribution, CLIP utilities, pipeline, demo
notebooks/      the Colab notebooks above (each embeds the code it runs)
results/        all results tables, figures and example pages
space/          the Hugging Face Space app
data/, runs/    datasets and trained models (not in git; created by the notebooks)
```

## Data

- **LIAR** (Wang, 2017): 12.8k PolitiFact statements. Six ratings mapped to three classes: pants-fire/false →
  Fabricated, barely-true/half-true → Misleading, mostly-true/true → Credible.
- **FakeNewsNet** (Shu et al., 2020): 1,000 articles crawled, 389 usable headline–image pairs.
- **TSHP-17** (Rashkin et al., 2017): tried for Satire/Propaganda; dropped because the model learned website style
  (see `docs/label_mapping.md`).

## Limitations

Text-only accuracy on LIAR is limited (~0.47 macro-F1) because truth depends on facts outside the statement; the
augmented model is still fooled by 73% of attacks; SHAP explanations are computed after the prediction; the
C-score detects swapped images but is not a fake-news signal on FakeNewsNet; only two of six deception patterns
occur often enough to learn. Details in the report.
