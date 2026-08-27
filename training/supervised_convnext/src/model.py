from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_IDENTITY_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "dependencies/canonical_identity/model.py"
)
MODEL_NAME = "SupervisedConvNeXt-Stable-Internal-PSAXPairHead"


def _load_canonical_identity_model():
    name = "canonical_identity_model_for_supervised_convnext"
    spec = importlib.util.spec_from_file_location(name, CANONICAL_IDENTITY_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CANONICAL_IDENTITY_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_canonical_identity = _load_canonical_identity_model()
A4C_PAIRS = _canonical_identity.A4C_PAIRS
PSAX_PAIRS = _canonical_identity.PSAX_PAIRS
OFFICIAL_VERTICAL_TASKS = _canonical_identity.OFFICIAL_VERTICAL_TASKS
PSAXPairAwareHead = _canonical_identity.PSAXPairAwareHead


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    return _canonical_identity.build_model(settings, task_configs)


def canonicalize_internal_points(points: torch.Tensor, task_id: str) -> torch.Tensor:
    return _canonical_identity.canonicalize_internal_points(points, task_id)


def canonicalize_training_target(
    target: dict[str, torch.Tensor],
    task_id: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    return _canonical_identity.canonicalize_training_target(target, task_id, settings)


def sort_official_vertical(
    points: torch.Tensor,
    task_id: str,
    settings: dict[str, Any] | None = None,
) -> torch.Tensor:
    return _canonical_identity.sort_official_vertical(points, task_id, settings)


def forward_train(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return _canonical_identity.forward_train(model, images, task_id, settings)


def forward_inference(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    return _canonical_identity.forward_inference(
        model,
        images,
        task_id,
        phase_or_settings,
        settings,
    )


def compute_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    phase: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _canonical_identity.compute_loss(output, target, task_id, phase, settings)


def configure_trainable(model: nn.Module, scope: str) -> dict[str, int]:
    return _canonical_identity.configure_trainable(model, scope)


def optimizer_groups(model: nn.Module, phase: dict[str, Any]) -> list[dict[str, Any]]:
    return _canonical_identity.optimizer_groups(model, phase)


def set_train_mode(model: nn.Module, scope: str) -> None:
    _canonical_identity.set_train_mode(model, scope)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return _canonical_identity.trainable_parameter_names(model)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return _canonical_identity.trainable_parameters(model)


def aggregate_internal_then_official(
    fold_predictions: torch.Tensor,
    task_id: str,
    settings: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median stable channels first, then apply one public-boundary conversion."""
    if fold_predictions.ndim != 4:
        raise ValueError(
            "Expected fold predictions shaped [fold,batch,point,xy], got "
            f"{tuple(fold_predictions.shape)}"
        )
    internal = fold_predictions.median(dim=0).values
    official = sort_official_vertical(internal, task_id, settings)
    return internal, official


def endpoint_identity_policy(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = _canonical_identity.endpoint_identity_policy(settings)
    policy.update(
        {
            "validation_primary_prediction": "fixed_internal",
            "validation_primary_target": "canonicalized_fixed_internal",
            "validation_secondary_prediction": "official_vertical_diagnostic",
            "validation_secondary_target": "active_official_vertical",
            "checkpoint_selection": "internal_final_proxy",
            "ensemble_order": "median_fixed_internal_channels_then_single_official_conversion",
        }
    )
    return policy
