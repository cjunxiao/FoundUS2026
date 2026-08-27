from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    Path(__file__).resolve().parents[1] / "dependencies/student/model.py"
)
EXPECTED_SHA256 = "478ee1350b8ff5d95462dcd9cd3ee0259671276007198887c67b6c6f48f30cce"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


spec = importlib.util.spec_from_file_location("final_student_model_for_final_distillation", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

DecoupledTaskPrivateFoundationCorrection = (
    module.DecoupledTaskPrivateFoundationCorrection
)
frozen_anchor_state_sha256 = module.frozen_anchor_state_sha256

FunctionalCompressionStudent = DecoupledTaskPrivateFoundationCorrection
