Grid search: 12 of 12 configurations finished, each trained up to 4 epochs (48 configuration x epoch points). Selected on validation macro-F1; the test set was not used.

| Rank | Encoder LR | Batch size | Dropout | Best epoch | Valid macro-F1 | Valid accuracy |
|---|---|---|---|---|---|---|
| 1 | 3e-05 | 16 | 0.3 | 2 | 0.5028 | 0.5039 |
| 2 | 2e-05 | 32 | 0.1 | 3 | 0.5022 | 0.5039 |
| 3 | 3e-05 | 32 | 0.1 | 2 | 0.5022 | 0.5039 |
| 4 | 1e-05 | 32 | 0.3 | 2 | 0.4998 | 0.5086 |
| 5 | 3e-05 | 32 | 0.3 | 2 | 0.4979 | 0.4977 |
| 6 | 2e-05 | 32 | 0.3 | 3 | 0.4963 | 0.4961 |
| 7 | 2e-05 | 16 | 0.1 | 2 | 0.4947 | 0.4977 |
| 8 | 2e-05 | 16 | 0.3 | 3 | 0.4946 | 0.4961 |
| 9 | 1e-05 | 16 | 0.3 | 2 | 0.4942 | 0.4930 |
| 10 | 1e-05 | 32 | 0.1 | 4 | 0.4919 | 0.4922 |
| 11 | 3e-05 | 16 | 0.1 | 2 | 0.4894 | 0.4891 |
| 12 | 1e-05 | 16 | 0.1 | 2 | 0.4850 | 0.4844 |

Average best validation macro-F1 for each value (how much each setting matters):

| Hyperparameter | Value | Mean valid macro-F1 | Runs |
|---|---|---|---|
| Encoder LR | 1e-05 | 0.4927 | 4 |
| Encoder LR | 2e-05 | 0.4969 | 4 |
| Encoder LR | 3e-05 | 0.4981 | 4 |
| Batch size | 16 | 0.4934 | 6 |
| Batch size | 32 | 0.4984 | 6 |
| Dropout | 0.1 | 0.4942 | 6 |
| Dropout | 0.3 | 0.4976 | 6 |

| Epoch | Mean valid macro-F1 over configs | Best over configs |
|---|---|---|
| 1 | 0.4640 | 0.4886 |
| 2 | 0.4926 | 0.5028 |
| 3 | 0.4871 | 0.5022 |
| 4 | 0.4787 | 0.4969 |

Best: encoder LR 3e-05, batch size 16, dropout 0.3, 2 epochs (valid macro-F1 0.5028). Spread across the grid: 0.0178.