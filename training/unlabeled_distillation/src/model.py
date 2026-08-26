from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    PROJECT_ROOT
    / "2-code/299-decoupled-task-private-foundation-correction/src/model.py"
)
EXPECTED_SHA256 = "478ee1350b8ff5d95462dcd9cd3ee0259671276007198887c67b6c6f48f30cce"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


actual = _sha256(SOURCE)
if actual != EXPECTED_SHA256:
    raise RuntimeError(
        f"Locked Exp299 model changed: expected={EXPECTED_SHA256}, actual={actual}"
    )
spec = importlib.util.spec_from_file_location("exp299_model_for_exp301", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

DecoupledTaskPrivateFoundationCorrection = (
    module.DecoupledTaskPrivateFoundationCorrection
)
frozen_anchor_state_sha256 = module.frozen_anchor_state_sha256
