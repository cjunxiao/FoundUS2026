
# Stable Landmark Identity and Unlabeled Adaptation

Official method and reproducibility materials for team `cjunxiao`, paper
`MWM95`, and FoundUS/FUB 2026 CodaBench submission `879954`.

Current public snapshot: [`postcompetition-v1.1`](https://github.com/cjunxiao/FoundUS2026/tree/postcompetition-v1.1).

The final challenge system is a single nine-task student model. Five fused
teachers are used only during training to construct reliability-weighted soft
heatmap targets from the organizer-provided unlabeled pool. Runtime inference
uses one dual-encoder graph, one checkpoint, and one image view.

## Results

| Evaluation | MRE | MAE |
| --- | ---: | ---: |
| Official validation | 24.7986 | 25.9233 |
| Hidden test | 24.9356 | 26.8758 |

- Leaderboard A / Method 1: preliminary rank `12/31`, score `0.28363`
- Leaderboard B / Method 2: preliminary rank `15/31`, score `0.27864`
- PSAX MRE: `39.311`, preliminary rank 1 in the PSAX MRE metric dimension

Rankings are preliminary pending final eligibility verification. Leaderboard B
averages were recomputed from organizer-published task values rounded to three
decimals; exact hidden aggregates are reported above.

## Method

1. Stable landmark identity preserves fixed heatmap-channel semantics through
   supervised fitting, teacher aggregation, student distillation, and output
   conversion.
2. ConvNeXt and USFM provide complementary local and ultrasound-domain dense
   representations.
3. Five fused teachers generate mean soft heatmaps and landmark reliability
   from coordinate dispersion.
4. Reliability-weighted distillation transfers teacher knowledge into one
   deployable student.

The official unlabeled pool supports distillation for eight task routes. Femur
retains its supervised route within the same nine-task runtime graph.

## Repository Layout

- `METHOD_OVERVIEW.md`: final method and nine-task configuration
- `FINAL_DISTILLATION.md`: reliability-weighted soft-heatmap distillation
- `REPRODUCIBILITY_QUICKSTART.md`: Docker execution, source verification, and training-stage map
- `final_inference/`: readable, runtime-equivalent inference implementation
- `exact_submission_source/`: immutable submitted source names and OCI labels
- `training/`: provenance-complete training snapshots and prerequisites
- `analyses/`: controlled appearance and landmark-identity studies
- `evidence/`: aggregate metrics, official results, QC, and runtime records
- `paper/`: camera-ready paper, supplement, and compact LaTeX source archive
- `MODEL_LICENSE_BOUNDARY.md`: team-code, pretrained-model, weight, and data license scope
- `THIRD_PARTY_NOTICES.md`: external initializations, licenses, and citations

## Executable Reference

The immutable public Docker is the executable reference. The deployment
checkpoint is identified by SHA256 in `final_inference/model_manifest.json` but
is not committed to this repository. Local reconstruction requires the
SHA-locked checkpoint and organizer-provided data. Raw challenge data, model
weights, teacher banks, caches, per-case predictions, and organizer image
overlays are intentionally excluded.

See [`REPRODUCIBILITY_QUICKSTART.md`](REPRODUCIBILITY_QUICKSTART.md) for the
container command, input/output contract, source-level verification, and the
prerequisites for each optimization stage.

## Runtime Verification

The published container processed 619 validation images on an NVIDIA RTX
A6000 in 35.36 seconds: 17.51 images/s and 57.12 ms/image amortized over the
complete run, with 1.63 GB peak CUDA reserved memory.

## License and Attribution

Repository-authored code is provided under the Apache License 2.0. This license
does not relicense third-party software, pretrained resources, derived model
weights, or organizer-provided data. USFM and weights derived from it remain
subject to CC BY-NC 4.0. See
[`MODEL_LICENSE_BOUNDARY.md`](MODEL_LICENSE_BOUNDARY.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before reuse.
