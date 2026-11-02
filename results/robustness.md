| Model | Clean acc | Clean macro-F1 | Paraphrased acc (orig → para) | Prediction unchanged | TextFooler: acc under attack | Attack success rate | Transfer acc* |
|---|---|---|---|---|---|---|---|
| Baseline (RoBERTa) | 0.482 | 0.470 | 0.482 → 0.465 | 76% | 0.068 | 86% | 0.000 |
| XFakeNet, no augmentation | 0.483 | 0.476 | 0.472 → 0.436 | 76% | 0.056 | 88% | 0.500 |
| XFakeNet v2 (augmented) | 0.462 | 0.460 | 0.447 → 0.434 | 71% | 0.100 | 78% | 0.572 |

*Transfer = accuracy on the adversarial texts that fooled the baseline (reference only; the fair robustness number is the model's own TextFooler attack).