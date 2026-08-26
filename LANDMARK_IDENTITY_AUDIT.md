
# Landmark Identity Audit

This no-training audit evaluates fixed five-fold OOF predictions. It does not
modify predictions, select checkpoints, search thresholds, or claim a causal
method gain.

| Pair group | N | Swap-sensitive pairs | Oracle-swap MRE sensitivity |
| --- | ---: | ---: | ---: |
| A4C horizontal | 368 | 32.07% | 29.99 px |
| A4C vertical | 368 | 0.27% | 0.01 px |
| PSAX | 86 | 20.93% | 5.44 px |
| Other-task controls | 4,914 | 4.46% | 0.27 px |

Swapping endpoint identities leaves pair-derived measurement values unchanged.
The diagnostic therefore demonstrates that endpoint correspondence affects MRE
independently of pair-derived measurements. It establishes the representation
problem, not the amount of performance improvement caused by the final method.
