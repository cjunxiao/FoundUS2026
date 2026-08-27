from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT_ROOT / "training/supervised_convnext/dependencies/fivefold_trainer/protocol.py"


def _load():
    name = "fivefold_trainer_protocol_for_dense_fusion"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_source = _load()
GroupedFoldDataset = _source.GroupedFoldDataset
DeterministicTaskUniformBatchSampler = _source.DeterministicTaskUniformBatchSampler
SequentialTaskBatchSampler = _source.SequentialTaskBatchSampler
collate_grouped_fold = _source.collate_grouped_fold
load_grouped_fold_fold = _source.load_grouped_fold_fold
phase_frame = _source.phase_frame
resolve_path = _source.resolve_path
sha256_file = _source.sha256_file
stable_row_id = _source.stable_row_id
task_uniform_expected_draws = _source.task_uniform_expected_draws
worker_init_fn = _source.worker_init_fn

