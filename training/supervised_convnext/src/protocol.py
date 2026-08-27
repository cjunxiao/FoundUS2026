from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIVEFOLD_TRAINER_PROTOCOL = Path(__file__).resolve().parents[1] / "dependencies/fivefold_trainer/protocol.py"


def _load_protocol():
    name = "fivefold_trainer_protocol_for_supervised_convnext"
    spec = importlib.util.spec_from_file_location(name, FIVEFOLD_TRAINER_PROTOCOL)
    if spec is None or spec.loader is None:
        raise ImportError(FIVEFOLD_TRAINER_PROTOCOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
GroupedFoldDataset = _protocol.GroupedFoldDataset
DeterministicTaskUniformBatchSampler = _protocol.DeterministicTaskUniformBatchSampler
SequentialTaskBatchSampler = _protocol.SequentialTaskBatchSampler
collate_grouped_fold = _protocol.collate_grouped_fold
load_grouped_fold_fold = _protocol.load_grouped_fold_fold
phase_frame = _protocol.phase_frame
resolve_path = _protocol.resolve_path
sha256_file = _protocol.sha256_file
stable_row_id = _protocol.stable_row_id
task_uniform_expected_draws = _protocol.task_uniform_expected_draws
worker_init_fn = _protocol.worker_init_fn
