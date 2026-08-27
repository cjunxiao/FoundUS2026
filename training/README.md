# Training

The training chain is organized by method stage:

1. `supervised_convnext/`: five-fold ConvNeXt heatmap training with stable
   A4C and PSAX channel identity.
2. `usfm_heatmap/`: five-fold USFM task-specific heatmap training.
3. `dense_fusion/`: five-fold supervised ConvNeXt-USFM residual fusion.
4. `calibrate_fusion.py`: task-wise residual fusion-scale selection.
5. `unlabeled_distillation/src/build_teacher_bank.py`: five-teacher mean heatmaps
   and coordinate dispersion for the official unlabeled pool.
6. `unlabeled_distillation/src/train_warmstart.py`: bounded warm-start on the
   configured subset.
7. `unlabeled_distillation/src/train.py`: full reliability-weighted student
   distillation.

Run these commands from `training/`:

```bash
python supervised_convnext/src/train.py --config supervised_convnext/configs/fold0.json
python usfm_heatmap/src/train.py --config usfm_heatmap/configs/fold0.json
python dense_fusion/src/train.py --config dense_fusion/configs/fold0.json
python calibrate_fusion.py --config configs/fusion_scales.json
python unlabeled_distillation/src/build_teacher_bank.py --config unlabeled_distillation/configs/final.json
python unlabeled_distillation/src/train_warmstart.py --config unlabeled_distillation/configs/warmstart.json
python unlabeled_distillation/src/train.py --config unlabeled_distillation/configs/final.json
```

Configs use paths relative to the source-code root. Supply organizer data and
the SHA-locked ConvNeXt, USFM, and intermediate fold checkpoints at those
paths. Femur retains its supervised route in the final student.
