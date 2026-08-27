# FoundUS 2026 Source Code

Code for paper `MWM95` and final CodaBench submission `879954`.

The submitted runtime is one ConvNeXt-USFM student, one checkpoint, and one
image view. Five fused teachers are used only to build unlabeled training
targets.

## Files

- `final_inference/`: readable runtime-equivalent inference source
- `training_reference.py`: core identity and distillation operations
- `THIRD_PARTY_NOTICES.md`: pretrained-resource and license scope

## Run the submitted image

```bash
docker pull docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
docker run --rm --gpus all --shm-size=2g \
  -e GU_INPUT_DIR=/input -e GU_OUTPUT_DIR=/output \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e
```

Input contains `test_metadata.csv` and task image folders. Output is
`regression_predictions.json`.

The Docker is the executable reference. Building `final_inference/` locally
also requires `best_model.pth` with SHA256 `73086754757a70be87319d636ab4b837aeeefbed88e2189add40484cfb7b8da1`; weights and
challenge data are not redistributed.

## Reported result

- Hidden test: MRE `24.9356`, MAE `26.8758`
- Leaderboard A: preliminary `12/31`, score `0.28363`
- PSAX MRE: `39.311`, preliminary rank 1 in that metric dimension

Repository: https://github.com/cjunxiao/FoundUS2026 (`postcompetition-v1.2`)
