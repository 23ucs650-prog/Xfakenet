# XFakeNet – Week 2 results

## Module 2: CLIP contradiction score (ViT-B/32, FakeNewsNet)

- Pairs: 389 aligned (real headline + own image), 389 mismatched (same headline + another article's image, same site)
- Mean C-score: aligned 0.701 ± 0.059, mismatched 0.823 ± 0.044
- **Separation (test ROC-AUC): 0.940**

| Threshold (fitted on validation) | Value | Test accuracy | F1 (mismatch) | Flags aligned | Flags mismatched |
|---|---|---|---|---|---|
| youden_on_valid | 0.753 | 0.879 | 0.887 | 18% | 94% |
| p95_aligned_valid | 0.811 | 0.782 | 0.739 | 5% | 62% |
| proposal_0.62 | 0.620 | 0.523 | 0.677 | 95% | 100% |

- Chosen threshold: **0.753** (Youden's J on validation)
- Fake vs real articles (own images): mean C-score 0.697 vs 0.706, AUC 0.429

## Gemini paraphrases

| Split | Generated | Accepted | Rejected (reasons) | Model |
|---|---|---|---|---|
| train | 2197 | 2091 (95%) | {'numbers_changed': 98, 'length': 8} | gemini-3.5-flash-lite |
| test | 499 | 477 (96%) | {'numbers_changed': 21, 'length': 1} | gemini-3.5-flash-lite |

- *Says Thom Tillis refuses to say what we should do to combat ISIS.* → *Claims Thom Tillis declines to state our strategy for fighting ISIS.*
- *Before Act 10, the total cost of public sector employees employment was 29 percent higher than that in the private sector. After the act, its still 22 percent higher.* → *Prior to Act 10, public sector worker compensation expenses exceeded private sector costs by 29 percent. Following the legislation, the gap remains at 22 percent.*

## Model v1 (RoBERTa-base, class-weighted CE, LIAR test set)

| Model | LR | Epochs | Accuracy | Macro-F1 | F1 Fabricated | F1 Misleading | F1 Credible |
|---|---|---|---|---|---|---|---|
| Baseline (week 1) | 2e-05 | 4 | 0.482 | 0.470 | 0.399 | 0.452 | 0.559 |
| Model v1 | 2e-05 | 5 | 0.469 | 0.457 | 0.388 | 0.444 | 0.540 |

LR sweep (validation macro-F1): 1e-05 → 0.500, 2e-05 → 0.501, 3e-05 → 0.495

## TextFooler adversarial examples (victim: week-1 baseline)

- Train: 2197 successful adversarial examples from 3930 attacked statements ({'success': 2197, 'skipped': 1340, 'failed': 393}); synonyms: counter-fitted
- Average words changed: 2.0, meaning similarity 0.92
- Successful examples by class: {'Misleading': 809, 'Credible': 781, 'Fabricated': 607}
- Example: *Says 25 percent of U.S. Muslims agreed that violence against Americans here in the United States is justified as a part of the global jihad.* → *Says 25 percent of U.S. Muslims agreed that onslaught against Americans here in the United States is justified as a part of the global jihad.*
- **Test robustness (week-1 baseline):** clean accuracy 0.485 on 499 statements → accuracy under attack 0.068; attack success rate 86%
