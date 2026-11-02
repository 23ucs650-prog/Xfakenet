# XFakeNet – final results tables

## Table 1. Classification on the clean LIAR test set (1,283 statements, 3 classes)

| Model | Seeds | Accuracy | Macro-F1 | F1 Fabricated | F1 Misleading | F1 Credible |
|---|---|---|---|---|---|---|
| Baseline (RoBERTa) | 3 | 0.483 ± 0.002 | 0.473 ± 0.007 | 0.428 ± 0.025 | 0.437 ± 0.040 | 0.554 ± 0.015 |
| XFakeNet, no augmentation | 3 | 0.468 ± 0.014 | 0.460 ± 0.015 | 0.438 ± 0.026 | 0.413 ± 0.050 | 0.530 ± 0.016 |
| XFakeNet v2 (augmented) | 3 | 0.465 ± 0.002 | 0.463 ± 0.003 | 0.441 ± 0.010 | 0.443 ± 0.026 | 0.504 ± 0.023 |

*Mean ± standard deviation over seeds when more than one seed was run.*

## Table 2. Robustness (each model attacked by its own TextFooler run, 499 test statements)

| Model | Seeds | Accuracy under attack | Attack success rate | Paraphrased: prediction unchanged |
|---|---|---|---|---|
| Baseline (RoBERTa) | 3 | 0.068 ± 0.006 | 0.857 ± 0.011 | 76% |
| XFakeNet, no augmentation | 3 | 0.057 ± 0.015 | 0.880 ± 0.027 | 76% |
| XFakeNet v2 (augmented) | 3 | 0.124 ± 0.028 | 0.730 ± 0.062 | 71% |

## Table 3. Ablations

**Adversarial + paraphrase augmentation** (XFakeNet without vs with):

| | Without | With |
|---|---|---|
| Clean macro-F1 | 0.460 ± 0.015 | 0.463 ± 0.003 |
| Accuracy under attack | 0.057 ± 0.015 | 0.124 ± 0.028 |
| Attack success rate | 0.880 ± 0.027 | 0.730 ± 0.062 |

**C-score signal** (FakeNewsNet fake vs real, 5-fold ROC-AUC):

| Features | AUC |
|---|---|
| text_only | 0.784 |
| image_only | 0.660 |
| cscore_only | 0.429 |
| text+image | 0.784 |
| text+image+cscore | 0.783 |

## Table 4. C-score calibration (aligned vs mismatched pairs, test split)

ROC-AUC 0.940

| Threshold | Value | Accuracy | Flags genuine pairs | Flags mismatched pairs |
|---|---|---|---|---|
| youden_on_valid | 0.753 | 0.879 | 18% | 94% |
| p95_aligned_valid | 0.811 | 0.782 | 5% | 62% |
| proposal_0.62 | 0.620 | 0.523 | 95% | 100% |

## Table 5. Explanation faithfulness (deletion test) and agreement with SHAP

| Model | Removed | Drop (top) | Drop (random) | Flip rate top / random | Top hurts more than random |
|---|---|---|---|---|---|
| XFakeNet v2 (augmented) | k=1 words | 0.028 | 0.017 | 17% / 14% | 50.8% |
| XFakeNet v2 (augmented) | k=3 words | 0.073 | 0.046 | 30% / 25% | 52.5% |
| XFakeNet v2 (augmented) | highlighted words | 0.074 | 0.050 | 27% / 23% | 53.9% |
| XFakeNet v2 (augmented) | top segment | 0.112 | 0.099 | – | – |
| XFakeNet, no augmentation | k=1 words | 0.037 | 0.026 | 17% / 13% | 54.9% |
| XFakeNet, no augmentation | k=3 words | 0.096 | 0.075 | 30% / 27% | 55.3% |
| XFakeNet, no augmentation | highlighted words | 0.092 | 0.077 | 30% / 26% | 55.4% |
| XFakeNet, no augmentation | top segment | 0.105 | 0.086 | – | – |

| Model | Statements (2+ segments) | Attention top segment = SHAP top segment | Chance | Spearman (3+ segments) |
|---|---|---|---|---|
| XFakeNet v2 (augmented) | 542 | 43.9% | 43.2% | 0.058 (n=192) |
| XFakeNet, no augmentation | 542 | 48.0% | 43.2% | 0.044 (n=192) |

## Table 5b. Faithful explanations: attention vs occlusion vs SHAP

Faithfulness of word and segment explanations, XFakeNet v2 (1283 LIAR test statements, GPU, 10 permutations). Drop = fall in the predicted-class probability.

| Test | Attention (built-in) | Occlusion | SHAP | Random |
|---|---|---|---|---|
| Remove top word ↑ | 0.028 (beats random 52%) | 0.204 (beats random 100%) | 0.113 (beats random 76%) | 0.013 |
| Remove top 3 words ↑ | 0.073 (beats random 53%) | 0.303 (beats random 93%) | 0.264 (beats random 88%) | 0.044 |
| Keep only top 3 words ↓ | 0.240 (beats random 54%) | 0.186 (beats random 63%) | 0.048 (beats random 81%) | 0.277 |
| Remove top segment ↑ | 0.127 (beats random 29%) | – | 0.263 (beats random 52%) | 0.127 |

↑ higher is better, ↓ lower is better. Occlusion at "remove top word" is the test itself (circular).
Attention and SHAP pick the same top segment in 43.9% of 542 multi-segment statements.

## Table 6. Deception-pattern attribution

- Gemini labels: 589 pairs; categories: {'consistent': 330, 'misattributed_entity': 127, 'irrelevant_stock': 117, 'manipulated_or_synthetic': 8, 'out_of_context': 5, 'exaggeration': 1, 'wrong_location_or_event': 1}
- Classifier (CLIP features, 5-fold CV vs Gemini labels, 257 pairs): accuracy 0.813, macro-F1 0.459 (majority-class accuracy 0.494)
- Hand-check: not uploaded yet
