# Complete Training Guide

This guide rebuilds the method from supervised folds to the final deployable
student. Run all commands from the repository root. The challenge data and
pretrained weights are not redistributed.

## 1. Environment

Use Linux, Python 3.11, an NVIDIA GPU, and CUDA 12.1. Teacher-bank generation
requires CUDA. The published run used PyTorch 2.3.1 and bfloat16 AMP.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
```

The complete run trains 15 fold models before distillation and stores dense
teacher heatmaps for 182,870 images. Plan for multiple GPU-days and up to
approximately 480 GB of working storage. Fold jobs may run on separate GPUs,
but each fold must use its matching config.

## 2. Required Inputs

Create this layout:

```text
data/
  labeled_csv/
    A4C_train.csv
    AOP_train.csv
    FA_train.csv
    fetal_femur_train.csv
    FUGC_train.csv
    HC_train.csv
    IVC_train.csv
    PLAX_train.csv
    PSAX_train.csv
  grouped_5fold_manifest.csv
  official_unlabeled_manifest.csv
weights/
  convnext_small.in12k_ft_in1k.safetensors
  USFM_latest.pth
```

Paths stored in CSV files may be repository-relative or absolute. The
organizer data remain subject to the organizer's access terms.

### Labeled CSV contract

Each task CSV contains one row per image and these columns:

```text
image_path,source_image_path,task_id,num_classes,point_1_xy,...,point_N_xy
```

- `source_image_path`: readable image file used by training.
- `image_path`: stable organizer-facing image identifier.
- `task_id`: `A4C`, `AOP`, `FA`, `fetal_femur`, `FUGC`, `HC`, `IVC`, `PLAX`, or
  `PSAX`.
- `num_classes`: number of landmark points for that task.
- `point_K_xy`: JSON text such as `[123.5, 87.0]`, in original-image pixels.

The required point counts are:

| Task | Points |
| --- | ---: |
| A4C | 16 |
| AOP | 4 |
| FA | 4 |
| fetal_femur | 2 |
| FUGC | 2 |
| HC | 4 |
| IVC | 2 |
| PLAX | 22 |
| PSAX | 4 |

A4C and PSAX CSVs must already use the canonical endpoint channel order
described in the paper. The same ordering is used by labels, heatmaps,
teachers, and the student.

### Five-fold manifest contract

`grouped_5fold_manifest.csv` contains:

```text
row_id,task_id,source_image_path,group_id,grouping_evidence,fold,
train_eligible,validation_eligible,excluded_reason
```

`row_id` is computed by `stable_row_id()` in
`supervised_convnext/dependencies/fivefold_trainer/protocol.py`. Groups must not
cross folds. The submitted protocol also records the organizer-listed Femur
exclusions and sequence-level grouping. This organizer-derived split manifest
is a required data-preparation input and is not redistributed with the code.

### Unlabeled manifest contract

`official_unlabeled_manifest.csv` contains exactly one row per unique readable
image:

```text
source_image_path,task_id,sha256
```

It contains 182,870 unique images over eight routes: A4C, AOP, FA, FUGC, HC,
IVC, PLAX, and PSAX. `sha256` is the image-file SHA256 and is also the stable
sample identifier. Femur retains its supervised route.

### Pretrained weights

- ConvNeXt Small: `timm/convnext_small.in12k_ft_in1k`.
- USFM BEiT-B/16: official `USFM_latest.pth`; expected submitted-run SHA256:
  `d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17`.

See `THIRD_PARTY_NOTICES.md` for license boundaries.

## 3. Lock Local Inputs

The supplied JSON files record the submitted run. Before a local run, update
their paths and hashes from the files on disk:

```bash
python training/update_pipeline_configs.py
```

Rerun this command after each major stage. It connects newly produced
checkpoints to the next stage and records their actual SHA256 values. It also
uses the same local unlabeled manifest for warm-start and full distillation,
avoiding dependence on historical manifest filenames.

Use command-line options when inputs are elsewhere:

```bash
python training/update_pipeline_configs.py \
  --labeled-csv-dir /data/foundus/labeled_csv \
  --fold-manifest /data/foundus/grouped_5fold_manifest.csv \
  --unlabeled-manifest /data/foundus/official_unlabeled_manifest.csv \
  --convnext-weights /models/convnext_small.in12k_ft_in1k.safetensors \
  --usfm-weights /models/USFM_latest.pth
```

## 4. Preflight

Run one forward/backward step for each supervised branch before full training:

```bash
python training/supervised_convnext/src/train.py \
  --config training/supervised_convnext/configs/fold0.json --smoke-forward
python training/usfm_heatmap/src/train.py \
  --config training/usfm_heatmap/configs/fold0.json --smoke-forward
```

The commands must finish with finite gradients. Smoke outputs use separate
`-smoke-forward` directories.

## 5. Train the Ten Supervised Experts

Train five ConvNeXt folds and five USFM folds:

```bash
for fold in 0 1 2 3 4; do
  python training/supervised_convnext/src/train.py \
    --config training/supervised_convnext/configs/fold${fold}.json
done

for fold in 0 1 2 3 4; do
  python training/usfm_heatmap/src/train.py \
    --config training/usfm_heatmap/configs/fold${fold}.json
done
```

The selected checkpoint from each run is:

```text
outputs/<stage>/foldK/checkpoints/best_final_proxy.pt
```

Selection uses the prespecified composite validation criterion. To resume an
interrupted supervised or USFM fold, pass its
`checkpoints/resume_last.pt` through `--resume-checkpoint`.

Update the dense-fusion configs with the ten selected checkpoints:

```bash
python training/update_pipeline_configs.py
```

## 6. Train Five Fused Teachers

```bash
for fold in 0 1 2 3 4; do
  python training/dense_fusion/src/train.py \
    --config training/dense_fusion/configs/fold${fold}.json
done
```

Each selected fused teacher is written to:

```text
outputs/dense_fusion/foldK/checkpoints/best_final_proxy.pt
```

The fold runs also write selected-checkpoint OOF heatmap caches required by
fusion-scale calibration.

## 7. Calibrate Task Fusion Scales

```bash
python training/calibrate_fusion.py \
  --config training/configs/fusion_scales.json
python training/update_pipeline_configs.py
```

The calibration output is
`outputs/fusion_scale_calibration/summary.json`. The updater copies its
`locked_task_scales` into both distillation configs and records the five fused
teacher checkpoint hashes.

## 8. Warm-Start the Student

Warm-start must run before teacher-bank generation because it creates the
fixed split consumed by the full-bank stage:

```bash
python training/unlabeled_distillation/src/train_warmstart.py \
  --config training/unlabeled_distillation/configs/warmstart.json
python training/update_pipeline_configs.py
```

Required outputs are:

```text
outputs/unlabeled_distillation/warmstart/stage0_unlabeled_split.csv
outputs/unlabeled_distillation/warmstart/stage0_task_private_lora_correction.pt
```

## 9. Build the Five-Teacher Target Bank

```bash
python training/unlabeled_distillation/src/build_teacher_bank.py \
  --config training/unlabeled_distillation/configs/final.json
```

This CUDA-only stage evaluates every eligible unlabeled image with all five
fused teachers. It stores mean soft heatmaps and coordinate dispersion under
`outputs/unlabeled_distillation/teacher_bank/teacher_bank/`. Continue only when
`bank_summary.json` and `qc_summary.json` report a passing, complete bank.

## 10. Full Reliability-Weighted Distillation

```bash
python training/unlabeled_distillation/src/train.py \
  --config training/unlabeled_distillation/configs/final.json
```

The trainable result is:

```text
outputs/unlabeled_distillation/final_student/
  final_distillation_task_private_lora_correction.pt
```

This file contains the task-specific LoRA and contextual-correction state. The
selected fold-1 fused teacher remains the frozen anchor. Check
`metrics_summary.json` and `qc_summary.json` before export.

## 11. Export the Complete Deployment Model

Merge the frozen anchor and distilled trainable state into the full checkpoint:

```bash
python training/update_pipeline_configs.py
python training/export_final_model.py \
  --config training/unlabeled_distillation/configs/final.json \
  --output-dir outputs/deployment
```

The export performs strict state-key, tensor-shape, and deployment-load checks.
It creates a self-contained Docker build context:

```text
outputs/deployment/
  best_model.pth
  model_manifest.json
  model.py
  predict.py
  Dockerfile
  requirements.txt
  THIRD_PARTY_NOTICES.md
  export_summary.json
```

`best_model.pth` contains the complete `model_state`, `task_configs`, and
calibrated `task_scales`; it is the final model, not only the 0.97M adapted
parameters.

## 12. Build and Run the Reproduced Model

```bash
docker build -t foundus2026-local outputs/deployment
docker run --rm --gpus all --shm-size=2g \
  -e GU_INPUT_DIR=/input -e GU_OUTPUT_DIR=/output \
  -v /path/to/test_input:/input:ro \
  -v /path/to/predictions:/output \
  foundus2026-local
```

The input root contains `test_metadata.csv` and task image directories such as
`A4C_test/`. The output is `regression_predictions.json`.

For a direct local model check, run from the export directory so the checkpoint
and generated manifest are found together:

```bash
cd outputs/deployment
FOUNDUS_DEVICE=cuda python -c "from model import Model; Model()"
```

## 13. Reproduction Boundary

The code implements the complete optimization and export path. Bitwise equality
with the submitted checkpoint additionally requires the same organizer data,
canonical CSVs, grouped fold manifest, pretrained weights, software stack, and
GPU execution conditions. A fresh retraining may produce a different checkpoint
SHA256 while remaining method-equivalent. The immutable submitted Docker digest
and checkpoint SHA256 in `final_inference/model_manifest.json` identify the
official scored artifact; locally generated manifests identify reproductions.
