
# Training Source Snapshot Status

The `training/` directories are compact provenance snapshots of the
source files and resolved configurations used during method development. They
are included so reviewers can inspect the optimization logic and trace the
reported evidence to immutable historical paths.

They are **not standalone package entry points**. Re-execution requires:

- organizer-provided labeled and unlabeled data;
- the disclosed ConvNeXt and USFM pretrained weights;
- fold-specific teacher checkpoints and the final student initialization;
- the historical module layout listed in `provenance/LEGACY_PATH_MAP.tsv`.

`final_inference/` provides readable runtime-equivalent source, while
`exact_submission_source/` preserves the submitted names and OCI labels. The
immutable public Docker is the executable reference. Because the checkpoint is
not duplicated here, local reconstruction requires the SHA-locked checkpoint.

No claim of package-level end-to-end retraining is made. Aggregate training
configs, metrics, coverage records, and hashes are supplied under `evidence/`.
