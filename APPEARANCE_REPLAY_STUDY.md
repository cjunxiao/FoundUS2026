
# Task-Conditioned Appearance Replay Study

This controlled five-fold study is reported separately from the submitted
distillation system. Organizer-provided landmark annotations supply geometry;
same-task donor images supply bounded appearance statistics only.

The matched labeled-donor control and unlabeled-donor condition use the same
folds, content-image schedule, adapter, optimizer, epoch count, and random seed.
Only donor source changes.

| Condition | Task-macro MRE | Measurement-proxy MAE | Combined proxy |
| --- | ---: | ---: | ---: |
| Matched labeled-donor control | 23.1183 | 17.0117 | 20.0650 |
| Unlabeled-donor adaptation | 23.0815 | 16.9964 | 20.0390 |
| Delta | -0.0367 | -0.0153 | -0.0260 |

The combined proxy improved in all five folds. The effect is consistent but
small, and no statistical-significance claim is made. These controlled-study
metrics are not the organizer's official Average MRE/Average MAE and are not
combined with official results in one comparison.
