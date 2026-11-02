# XFakeNet – Week 3 results

## Augmented training set

- 14540 training statements: liar 10252 (70.5%), textfooler 2197 (15.1%), paraphrase 2091 (14.4%)

## Clean LIAR test set (1,283 statements)

| Model | Accuracy | Macro-F1 | F1 Fabricated | F1 Misleading | F1 Credible |
|---|---|---|---|---|---|
| Baseline (RoBERTa, week 1) | 0.482 | 0.470 | 0.399 | 0.452 | 0.559 |
| XFakeNet, no augmentation | 0.483 | 0.476 | 0.422 | 0.469 | 0.537 |
| XFakeNet v2 (augmented) | 0.462 | 0.460 | 0.430 | 0.462 | 0.489 |

## Robustness (LIAR test)

| Model | Clean acc | Clean macro-F1 | Paraphrased acc (orig → para) | Prediction unchanged | TextFooler: acc under attack | Attack success rate | Transfer acc* |
|---|---|---|---|---|---|---|---|
| Baseline (RoBERTa) | 0.482 | 0.470 | 0.482 → 0.465 | 76% | 0.068 | 86% | 0.000 |
| XFakeNet, no augmentation | 0.483 | 0.476 | 0.472 → 0.436 | 76% | 0.056 | 88% | 0.500 |
| XFakeNet v2 (augmented) | 0.462 | 0.460 | 0.447 → 0.434 | 71% | 0.100 | 78% | 0.572 |

*Transfer = accuracy on the adversarial texts that fooled the baseline (reference only; the fair robustness number is the model's own TextFooler attack).

## Module 1: explanations (XFakeNet v2, LIAR test)

- Segments per statement: 1.62 on average; 42.2% of statements have more than one segment
- Most suspicious segment's attention weight (multi-segment statements): 0.537
- Highlighted words (importance ≥ 0.65): 2.95 per statement (16.4% of words)
- 40 colour-coded examples: explanations.html

- *America is No. 1 in wind power.* → **Credible**; highlighted: No., power.; most suspicious segment: "1 in wind power."
- *Before I was state treasurer, my Rhode Island business helped create over 1,000 jobs.* → **Credible**; highlighted: treasurer,; most suspicious segment: "Before I was state treasurer"
- *"Fifty-six percent decline in overall crime. A 73 percent decline in motor-vehicle theft. A 67 percent decline in robbery. A 66 percent decline in murder. This is way beyond what happened in the nation during this period of time.* → **Credible**; highlighted: "Fifty-six, motor-vehicle; most suspicious segment: "A 73 percent decline in motor-vehicle theft."

## Modality flag (FakeNewsNet, 389 items)

- Flags: {'TEXT': 277, 'CROSS-MODAL': 65, 'IMAGE': 47} (CROSS-MODAL = C-score > 0.7534)
- Fake vs real AUC (5-fold): text 0.7836, image 0.6598, C-score 0.4292, text+image 0.7837, text+image+C-score 0.7827
