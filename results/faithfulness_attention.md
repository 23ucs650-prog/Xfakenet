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