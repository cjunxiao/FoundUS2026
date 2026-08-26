
# Reproducibility Quickstart

This guide separates executable final inference from the training-stage source
snapshots. The immutable public Docker is the executable reference for final
CodaBench submission `879954`.

## 1. Final inference

Prerequisites: Docker with NVIDIA Container Toolkit, an NVIDIA GPU compatible
with CUDA 12.1, and organizer-format input data.

```bash
docker pull docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
docker run --rm --gpus all --shm-size=2g \
  -e GU_INPUT_DIR=/input \
  -e GU_OUTPUT_DIR=/output \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
```

The input directory contains `test_metadata.csv` and task folders such as
`A4C_test`, `AOP_test`, and `PSAX_test`. The container writes
`regression_predictions.json` to the output directory. Runtime uses one model
graph, one checkpoint, and one image view.

## 2. Source-level verification

`final_inference/` is a readable, runtime-equivalent implementation. The
deployment checkpoint is not duplicated in this compact package. Its required
SHA256 is recorded in `final_inference/model_manifest.json`:

```text
73086754757a70be87319d636ab4b837aeeefbed88e2189add40484cfb7b8da1
```

With the SHA-locked checkpoint available as `final_inference/best_model.pth`,
the directory can be built using its Dockerfile. `exact_submission_source/`
preserves the original submitted symbols and OCI labels.

## 3. Training-stage map

| Stage | Source | Resolved configuration | Required upstream artifacts |
| --- | --- | --- | --- |
| Supervised ConvNeXt | `training/supervised_convnext/src/train.py` | `training/supervised_convnext/configs/fold*.json` | Labeled data, ConvNeXt initialization, grouped folds |
| USFM heatmap branch | `training/usfm_heatmap/src/train.py` | `training/usfm_heatmap/configs/fold*.json` | Labeled data, `USFM_latest.pth` |
| Dense fusion | `training/dense_fusion/src/train.py` | `training/dense_fusion/configs/fold*.json` | Matching ConvNeXt and USFM fold checkpoints |
| Teacher targets | `training/unlabeled_distillation/src/build_teacher_bank.py` | `training/unlabeled_distillation/configs/teacher_bank.json` | Five fused teachers, official unlabeled pool |
| Final student | `training/unlabeled_distillation/src/train.py` | `evidence/final_submission/distillation_config.resolved.json` | Teacher bank, selected student initialization |

After organizer data and the upstream checkpoints are restored according to
`provenance/LEGACY_PATH_MAP.tsv`, each stage follows the common invocation
pattern:

```bash
python <stage-script> --config <resolved-config>
```

The snapshots document the optimization logic and exact configurations; they
are not a redistribution of organizer data, pretrained weights, fold
checkpoints, or the 12 GB teacher bank.

## 4. Aggregate verification

- Official results: `evidence/official_results/`
- Final configuration and coverage: `evidence/final_submission/`
- Runtime profile: `evidence/runtime_profile/`
- Controlled studies: `evidence/controlled_studies/`
- Component mapping: `readable_reference/COMPONENT_MAP.md`

For the complete post-competition ZIP, run `sha256sum -c MANIFEST.sha256` from
its root. The Docker digest, checkpoint SHA256, output contract, and official
metrics are also recorded in machine-readable manifests.
