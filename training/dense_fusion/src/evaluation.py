from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT_ROOT / "training/supervised_convnext/src/evaluation.py"


def _load():
    name = "supervised_convnext_evaluation_for_dense_fusion"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_source = _load()
evaluate = _source.evaluate
measurement_values = _source.measurement_values
canvas_to_original = _source.canvas_to_original

