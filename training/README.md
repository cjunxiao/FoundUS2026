# Training

The complete procedure is documented in
[`TRAINING_GUIDE.md`](TRAINING_GUIDE.md). It covers:

1. five-fold supervised ConvNeXt and USFM training;
2. five-fold dense fusion and task-scale calibration;
3. warm-start, teacher-bank generation, and full unlabeled distillation;
4. export of the full `best_model.pth` deployment checkpoint; and
5. local inference and Docker verification.

`update_pipeline_configs.py` records the SHA256 and paths produced by each
stage. `export_final_model.py` merges the selected fused anchor with the final
LoRA/correction state into the complete inference checkpoint.
