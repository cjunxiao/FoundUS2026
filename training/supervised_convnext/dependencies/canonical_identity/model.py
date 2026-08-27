from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PSAX_PAIR_HEAD_MODEL_PATH = Path(__file__).resolve().parents[1] / "psax_pair_head/model.py"
MODEL_NAME = "CanonicalIdentity-Canonical-Internal-PSAXPairHead-Official-Vertical-Output"

A4C_PAIRS = tuple((index, index + 1) for index in range(0, 16, 2))
PSAX_PAIRS = ((0, 1), (2, 3))
OFFICIAL_VERTICAL_TASKS = frozenset({"A4C", "PSAX"})


def _load_psax_pair_head_model():
    name = "psax_pair_head_model_for_canonical_identity"
    spec = importlib.util.spec_from_file_location(name, PSAX_PAIR_HEAD_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PSAX_PAIR_HEAD_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_psax_pair_head = _load_psax_pair_head_model()
PSAXPairAwareHead = _psax_pair_head.PSAXPairAwareHead


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    variant = str(settings.get("model_variant", "psax_pair"))
    if variant == "psax_pair":
        return _psax_pair_head.build_model(settings, task_configs)
    if variant == "canonical_baseline_heatmap":
        baseline_settings = {
            "encoder": settings["encoder"],
            "encoder_weights": settings.get("encoder_weights"),
            "heatmap_size": settings["heatmap_size"],
            "head_hidden_channels": settings.get("head_hidden_channels"),
        }
        return _psax_pair_head._baseline_heatmap.build_model(baseline_settings, task_configs)
    raise ValueError(f"Unknown CanonicalIdentity model_variant: {variant}")


def _swap_pair_where(
    values: torch.Tensor,
    first: int,
    second: int,
    swap: torch.Tensor,
) -> torch.Tensor:
    output = values.clone()
    first_value = values[:, first].clone()
    second_value = values[:, second].clone()
    view = (values.shape[0],) + (1,) * (values.ndim - 2)
    selector = swap.reshape(view)
    output[:, first] = torch.where(selector, second_value, first_value)
    output[:, second] = torch.where(selector, first_value, second_value)
    return output


def canonicalize_internal_points(points: torch.Tensor, task_id: str) -> torch.Tensor:
    """Map official endpoint order to the fixed identities used by PSAXPairHead."""
    task = str(task_id)
    output = points
    if task == "A4C":
        if points.shape[1] != 16:
            raise ValueError(f"A4C requires 16 points, got {points.shape[1]}.")
        for pair_number, (first, second) in enumerate(A4C_PAIRS, start=1):
            if pair_number % 2 == 0:
                swap = output[:, first, 0] > output[:, second, 0]
            else:
                swap = output[:, first, 1] > output[:, second, 1]
            output = _swap_pair_where(output, first, second, swap)
    elif task == "PSAX":
        if points.shape[1] != 4:
            raise ValueError(f"PSAX requires 4 points, got {points.shape[1]}.")
        for first, second in PSAX_PAIRS:
            swap = output[:, first, 0] < output[:, second, 0]
            output = _swap_pair_where(output, first, second, swap)
    return output


def canonical_order_indices(points: torch.Tensor, task_id: str) -> torch.Tensor:
    batch, count = points.shape[:2]
    labels = torch.arange(count, device=points.device).expand(batch, count)
    task = str(task_id)
    output = labels
    if task == "A4C":
        for pair_number, (first, second) in enumerate(A4C_PAIRS, start=1):
            if pair_number % 2 == 0:
                swap = points[:, first, 0] > points[:, second, 0]
            else:
                swap = points[:, first, 1] > points[:, second, 1]
            output = _swap_pair_where(output, first, second, swap)
    elif task == "PSAX":
        for first, second in PSAX_PAIRS:
            swap = points[:, first, 0] < points[:, second, 0]
            output = _swap_pair_where(output, first, second, swap)
    return output


def _gather_point_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    shape = indices.shape + (1,) * (values.ndim - 2)
    gather_index = indices.reshape(shape).expand_as(values)
    return torch.gather(values, 1, gather_index)


def canonicalize_training_target(
    target: dict[str, torch.Tensor],
    task_id: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    internal_tasks = {
        str(value)
        for value in (settings or {}).get(
            "internal_identity_tasks",
            sorted(OFFICIAL_VERTICAL_TASKS),
        )
    }
    if str(task_id) not in internal_tasks:
        return target
    points = target["points_norm"]
    indices = canonical_order_indices(points, task_id)
    output = dict(target)
    for name in ("points_norm", "points_original", "heatmap"):
        if name in output:
            output[name] = _gather_point_axis(output[name], indices)
    return output


def sort_official_vertical(
    points: torch.Tensor,
    task_id: str,
    settings: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Convert internal identities to the official odd-even vertical order."""
    task = str(task_id)
    output_sort_tasks = {
        str(value)
        for value in (settings or {}).get(
            "official_output_sort_tasks",
            sorted(OFFICIAL_VERTICAL_TASKS),
        )
    }
    if task not in output_sort_tasks:
        return points
    pairs = A4C_PAIRS if task == "A4C" else PSAX_PAIRS
    output = points
    for first, second in pairs:
        swap = output[:, first, 1] > output[:, second, 1]
        output = _swap_pair_where(output, first, second, swap)
    return output


def forward_train(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return _psax_pair_head.forward_train(model, images, task_id, settings)


def forward_inference(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if settings is None:
        settings = phase_or_settings
    output = model(images, task_id=str(task_id))
    endpoint = _psax_pair_head.decode_topk(
        output["heatmap_logits"],
        int(settings.get("decode_topk", 25)),
        float(settings.get("decode_topk_beta", 1.0)),
    )
    official = sort_official_vertical(endpoint, task_id, settings)
    output["internal_coords_norm"] = endpoint
    output["coords_norm"] = official
    # The historical structured scorer is intentionally disabled. The primary
    # endpoint heatmaps are converted deterministically at the model boundary.
    output["structured_coords_norm"] = official
    return output


def compute_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    phase: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    internal_target = canonicalize_training_target(target, task_id, settings)
    if str(task_id) == "PSAX" and "midpoint_logits" not in output:
        endpoint = F.mse_loss(
            torch.sigmoid(output["heatmap_logits"]), internal_target["heatmap"]
        )
        zero = endpoint.detach() * 0.0
        return endpoint, {
            "total_loss": endpoint,
            "heatmap_loss": endpoint,
            "midpoint_loss": zero,
            "tube_loss": zero,
            "direction_loss": zero,
        }
    return _psax_pair_head.compute_loss(output, internal_target, task_id, phase, settings)


def configure_trainable(model: nn.Module, scope: str) -> dict[str, int]:
    return _psax_pair_head.configure_trainable(model, scope)


def optimizer_groups(model: nn.Module, phase: dict[str, Any]) -> list[dict[str, Any]]:
    return _psax_pair_head.optimizer_groups(model, phase)


def set_train_mode(model: nn.Module, scope: str) -> None:
    _psax_pair_head.set_train_mode(model, scope)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return _psax_pair_head.trainable_parameter_names(model)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def endpoint_identity_policy(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    internal_tasks = sorted(
        str(value)
        for value in (settings or {}).get(
            "internal_identity_tasks",
            sorted(OFFICIAL_VERTICAL_TASKS),
        )
    )
    output_sort_tasks = sorted(
        str(value)
        for value in (settings or {}).get(
            "official_output_sort_tasks",
            sorted(OFFICIAL_VERTICAL_TASKS),
        )
    )
    return {
        "training_input_labels": "active official vertical-order CSV",
        "internal_identity_tasks": internal_tasks,
        "official_output_sort_tasks": output_sort_tasks,
        "training_internal_identity": {
            "A4C": {
                "pairs_1_3_5_7": "first endpoint has smaller y",
                "pairs_2_4_6_8": "first endpoint has smaller x",
            },
            "PSAX": {"pairs_1_2": "first endpoint has larger x"},
        },
        "inference_output_identity": {
            "A4C": "each odd-even pair sorted by y ascending",
            "PSAX": "each odd-even pair sorted by y ascending",
        },
        "official_csv_modified": False,
        "structured_psax_scorer_enabled": False,
    }
