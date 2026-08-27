# FoundUS 2026 Source Code

Complete training and inference code for paper `MWM95` and submission `879954`.

## Contents

- `training/`: supervised ConvNeXt, USFM heatmaps, dense fusion, fusion-scale
  calibration, teacher-target generation, warm-start, and final distillation
- `final_inference/`: the complete readable inference implementation
- `THIRD_PARTY_NOTICES.md`: pretrained-resource and license scope

The challenge data, pretrained weights, intermediate checkpoints, and final
checkpoint are not redistributed. Required paths are listed in the JSON
configs. The immutable submitted Docker is the executable inference reference.

## Inference

```bash
docker pull docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
docker run --rm --gpus all --shm-size=2g \
  -e GU_INPUT_DIR=/input -e GU_OUTPUT_DIR=/output \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
```

Input contains `test_metadata.csv` and task image folders. Output is
`regression_predictions.json`.

## Training

See `training/README.md`. Each stage accepts a JSON config and writes its own
checkpoints. Five fused teachers build soft heatmap targets for eight unlabeled
task routes; the final runtime uses one student checkpoint.

Repository: https://github.com/cjunxiao/FoundUS2026 (`postcompetition-v1.3.1`)
