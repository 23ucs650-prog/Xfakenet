Faithfulness of word and segment explanations, XFakeNet v2 (1283 LIAR test statements, GPU, 10 permutations). Drop = fall in the predicted-class probability.

| Test | Attention (built-in) | Occlusion | SHAP | Random |
|---|---|---|---|---|
| Remove top word ↑ | 0.028 (beats random 52%) | 0.204 (beats random 100%) | 0.113 (beats random 76%) | 0.013 |
| Remove top 3 words ↑ | 0.073 (beats random 53%) | 0.303 (beats random 93%) | 0.264 (beats random 88%) | 0.044 |
| Keep only top 3 words ↓ | 0.240 (beats random 54%) | 0.186 (beats random 63%) | 0.048 (beats random 81%) | 0.277 |
| Remove top segment ↑ | 0.127 (beats random 29%) | – | 0.263 (beats random 52%) | 0.127 |

↑ higher is better, ↓ lower is better. Occlusion at "remove top word" is the test itself (circular).
Attention and SHAP pick the same top segment in 43.9% of 542 multi-segment statements.