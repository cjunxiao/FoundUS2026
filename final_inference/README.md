
# Final Single-Model Inference

This directory provides a readable, runtime-equivalent implementation
corresponding to final CodaBench submission `879954`. The immutable submitted
symbols and OCI labels are preserved separately in `../exact_submission_source/`.

Runtime executes one task-routed ConvNeXt-USFM graph, one checkpoint, and one
image view. It performs full-frame dense heatmap prediction, top-k coordinate
expectation, inverse letterbox restoration, and one deterministic A4C/PSAX
output-order conversion.

The immutable public Docker is the executable reference. The 834 MB deployment
checkpoint is omitted from this compact source package, so local reconstruction
requires the SHA-locked checkpoint recorded in `model_manifest.json`. The
published artifact was verified on an NVIDIA RTX A6000 with PyTorch 2.3.1 and
CUDA 12.1 and processed all 619 official-validation images successfully.
