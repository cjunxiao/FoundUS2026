
# Readable Component Map

The public `final_inference/` directory uses formal method names while retaining
the exact executable logic. `exact_submission_source/` preserves the immutable
submitted source names and OCI labels for provenance.

| Formal component | Readable source | Exact-source symbol or path |
| --- | --- | --- |
| Final single-model student | `final_inference/model.py::FinalStudentNetwork` | `Exp301Network` |
| Stable endpoint conversion | `final_inference/model.py::sort_official_vertical` | same function |
| Task-routed inference | `final_inference/model.py::Model` | `Model` |
| Container entry point | `final_inference/predict.py` | `exact_submission_source/predict.py` |
| Immutable artifact identity | `final_inference/model_manifest.json` | Docker digest and checkpoint SHA256 |

Only public-facing identifiers and nonfunctional OCI labels differ. Runtime
tensor operations, checkpoint keys, decoding, and output serialization are
unchanged. The top-level `checkpoint_sha256` compatibility field mirrors the
nested manifest value expected by the submitted source.
