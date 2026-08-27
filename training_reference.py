"""Core training operations used by the final FoundUS student."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


A4C_PAIRS = tuple((index, index + 1) for index in range(0, 16, 2))
PSAX_PAIRS = ((0, 1), (2, 3))
RELIABILITY_FLOOR = 0.10
RELIABILITY_SCALE_QUANTILE = 0.75
COORDINATE_LOSS_WEIGHT = 0.05
REGULARIZATION_WEIGHT = 1e-5
DECODE_TOPK = 25


def _swap_pair_where(
    values: torch.Tensor,
    first: int,
    second: int,
    swap: torch.Tensor,
) -> torch.Tensor:
    output = values.clone()
    shape = (values.shape[0],) + (1,) * (values.ndim - 2)
    selector = swap.reshape(shape)
    output[:, first] = torch.where(selector, values[:, second], values[:, first])
    output[:, second] = torch.where(selector, values[:, first], values[:, second])
    return output


def canonicalize_landmarks(points: torch.Tensor, task: str) -> torch.Tensor:
    """Apply the fixed A4C/PSAX channel identity before target generation."""
    output = points
    if task == "A4C":
        if points.shape[1] != 16:
            raise ValueError("A4C requires 16 points")
        for pair_number, (first, second) in enumerate(A4C_PAIRS, start=1):
            axis = 0 if pair_number % 2 == 0 else 1
            output = _swap_pair_where(
                output, first, second, output[:, first, axis] > output[:, second, axis]
            )
    elif task == "PSAX":
        if points.shape[1] != 4:
            raise ValueError("PSAX requires 4 points")
        for first, second in PSAX_PAIRS:
            output = _swap_pair_where(
                output, first, second, output[:, first, 0] < output[:, second, 0]
            )
    return output


def organizer_order(points: torch.Tensor, task: str) -> torch.Tensor:
    """Convert stable internal channels once at the organizer output boundary."""
    if task not in {"A4C", "PSAX"}:
        return points
    output = points
    for first in range(0, points.shape[1], 2):
        output = _swap_pair_where(
            output,
            first,
            first + 1,
            output[:, first, 1] > output[:, first + 1, 1],
        )
    return output


def teacher_consensus(
    probabilities: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return five-teacher mean heatmaps and coordinate dispersion.

    `probabilities` has shape [teachers, batch, landmarks, height, width].
    `coordinates` has shape [teachers, batch, landmarks, 2].
    """
    mean_probability = probabilities.mean(0)
    mean_probability = mean_probability / mean_probability.sum(
        (-2, -1), keepdim=True
    ).clamp_min(1e-12)
    coordinate_mean = coordinates.mean(0)
    variance = (coordinates.square().mean(0) - coordinate_mean.square()).clamp_min(0)
    dispersion = variance.sum(-1).sqrt()
    return mean_probability, dispersion


def reliability_weights(
    dispersion: torch.Tensor,
    floor: float = RELIABILITY_FLOOR,
    scale_quantile: float = RELIABILITY_SCALE_QUANTILE,
) -> torch.Tensor:
    scale = torch.quantile(dispersion.float(), scale_quantile).clamp_min(1e-4)
    return floor + (1.0 - floor) / (1.0 + (dispersion / scale).square())


def topk_coordinates(probability: torch.Tensor, topk: int = DECODE_TOPK) -> torch.Tensor:
    _, _, height, width = probability.shape
    flat = probability.flatten(2)
    values, indices = torch.topk(flat, min(topk, flat.shape[-1]), dim=-1)
    weights = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
    x = (indices % width).to(probability.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(probability.dtype)
    return torch.stack(((weights * x).sum(-1), (weights * y).sum(-1)), dim=-1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-12)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    reliability: torch.Tensor,
    residual_logits: torch.Tensor,
    lora_up_parameters: Iterable[torch.Tensor] = (),
) -> dict[str, torch.Tensor]:
    """Reliability-weighted heatmap and coordinate objective."""
    target = teacher_probability / teacher_probability.sum(
        (-2, -1), keepdim=True
    ).clamp_min(1e-12)
    log_probability = F.log_softmax(student_logits.float().flatten(2), dim=-1)
    target_flat = target.float().flatten(2).clamp_min(1e-12)
    kl_point = (target_flat * (target_flat.log() - log_probability)).sum(-1)

    student_coordinate = topk_coordinates(log_probability.exp().reshape_as(target))
    teacher_coordinate = topk_coordinates(target)
    coordinate_point = F.smooth_l1_loss(
        student_coordinate, teacher_coordinate, beta=1.0, reduction="none"
    ).mean(-1)

    kl = _weighted_mean(kl_point, reliability)
    coordinate = _weighted_mean(coordinate_point, reliability)
    correction_l2 = residual_logits.float().square().mean()
    lora_terms = [parameter.float().square().mean() for parameter in lora_up_parameters]
    lora_l2 = torch.stack(lora_terms).mean() if lora_terms else correction_l2 * 0.0
    total = (
        kl
        + COORDINATE_LOSS_WEIGHT * coordinate
        + REGULARIZATION_WEIGHT * correction_l2
        + REGULARIZATION_WEIGHT * lora_l2
    )
    return {"loss": total, "kl": kl, "coordinate": coordinate}
