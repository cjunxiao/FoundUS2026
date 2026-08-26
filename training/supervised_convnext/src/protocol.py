from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP157_PROTOCOL = PROJECT_ROOT / "2-code/157-active-vertical-exp152-5fold/src/protocol.py"


def _load_protocol():
    name = "exp157_protocol_for_exp161"
    spec = importlib.util.spec_from_file_location(name, EXP157_PROTOCOL)
    if spec is None or spec.loader is None:
        raise ImportError(EXP157_PROTOCOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
Exp148Dataset = _protocol.Exp148Dataset
DeterministicTaskUniformBatchSampler = _protocol.DeterministicTaskUniformBatchSampler
SequentialTaskBatchSampler = _protocol.SequentialTaskBatchSampler
collate_exp148 = _protocol.collate_exp148
load_exp148_fold = _protocol.load_exp148_fold
phase_frame = _protocol.phase_frame
resolve_path = _protocol.resolve_path
sha256_file = _protocol.sha256_file
stable_row_id = _protocol.stable_row_id
task_uniform_expected_draws = _protocol.task_uniform_expected_draws
worker_init_fn = _protocol.worker_init_fn
