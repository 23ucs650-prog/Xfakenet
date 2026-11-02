# XFakeNet – Week 1 results

## Dataset (3-class corpus, LIAR)

| Split | Fabricated | Misleading | Credible | Total |
|---|---|---|---|---|
| train | 2832 | 3774 | 3646 | 10252 |
| valid | 379 | 485 | 420 | 1284 |
| test | 342 | 481 | 460 | 1283 |

## Baseline: RoBERTa-base, class-weighted cross-entropy (test set)

- **Accuracy:** 0.482
- **Macro-F1:** 0.470

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Fabricated | 0.454 | 0.357 | 0.399 | 342 |
| Misleading | 0.462 | 0.443 | 0.452 | 481 |
| Credible | 0.512 | 0.615 | 0.559 | 460 |

## FakeNewsNet image-text pairs

- Articles tried: 1000
- **Usable headline+image pairs: 389** (39%)
- By label: {'real': 196, 'fake': 193}
- Failure reasons: {'page_dead': 488, 'ok': 389, 'no_image_tag': 60, 'image_too_small': 35, 'image_dead': 23, 'image_corrupt': 5}