Default settings: encoder LR 2e-05, batch size 32, dropout 0.1, 4 epochs. Tuned: encoder LR 3e-05, batch size 16, dropout 0.3, up to 4 epochs (best epoch by validation).

| Model | Seeds | Valid macro-F1 | Test accuracy | Test macro-F1 | Accuracy under attack | Attack success rate |
|---|---|---|---|---|---|---|
| Default settings (XFakeNet v2) | 3 | 0.497 ± 0.012 | 0.465 ± 0.002 | 0.463 ± 0.003 | 0.124 ± 0.028 | 0.730 ± 0.062 |
| Tuned (grid-search best) | 3 | 0.495 ± 0.007 | 0.456 ± 0.020 | 0.451 ± 0.017 | 0.138 ± 0.005 | 0.697 ± 0.018 |