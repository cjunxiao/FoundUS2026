from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP56_MODEL_PATH = PROJECT_ROOT / "2-code/56-convnext-small-heatmap/src/model.py"
MODEL_NAME = "Exp152-Exp56-PSAX-Pair-Aware-Heatmap"
PSAX_PAIRS = ((0, 1), (2, 3))


def _load_exp56_model():
    name = "exp56_model_for_exp152"
    spec = importlib.util.spec_from_file_location(name, EXP56_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(EXP56_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_exp56 = _load_exp56_model()


class ConvGroupNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        groups = min(int(groups), int(out_channels))
        while out_channels % groups and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)


class PSAXPairAwareHead(nn.Module):
    """PSAX-only dense head; all outputs retain full spatial evidence."""

    def __init__(
        self,
        in_channels: int,
        heatmap_size: int,
        hidden_channels: int | None,
        norm_groups: int,
    ):
        super().__init__()
        hidden = int(hidden_channels or max(in_channels // 2, 192))
        mid = max(hidden // 2, 96)
        self.heatmap_size = int(heatmap_size)
        self.trunk = nn.Sequential(
            ConvGroupNormAct(in_channels, hidden, norm_groups),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGroupNormAct(hidden, mid, norm_groups),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGroupNormAct(mid, mid, norm_groups),
        )
        self.endpoint_logits = nn.Conv2d(mid, 4, 1)
        self.midpoint_logits = nn.Conv2d(mid, 2, 1)
        self.tube_logits = nn.Conv2d(mid, 2, 1)
        self.direction_field = nn.Conv2d(mid, 4, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        if hidden.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            hidden = F.interpolate(
                hidden,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        batch, _, height, width = hidden.shape
        return {
            "heatmap_logits": self.endpoint_logits(hidden),
            "midpoint_logits": self.midpoint_logits(hidden),
            "tube_logits": self.tube_logits(hidden),
            "direction_field": torch.tanh(self.direction_field(hidden)).reshape(
                batch, 2, 2, height, width
            ),
        }


class Exp152Model(nn.Module):
    def __init__(self, settings: dict[str, Any], task_configs: list[dict[str, Any]]):
        super().__init__()
        baseline_settings = {
            "encoder": settings["encoder"],
            "encoder_weights": settings.get("encoder_weights"),
            "heatmap_size": settings["heatmap_size"],
            "head_hidden_channels": settings.get("head_hidden_channels"),
        }
        baseline = _exp56.build_model(baseline_settings, task_configs)
        if "PSAX" not in baseline.heads:
            raise ValueError("Exp152 requires a PSAX task head.")
        self.encoder = baseline.encoder
        self.heads = baseline.heads
        self.heads["PSAX"] = PSAXPairAwareHead(
            self.encoder.out_channels,
            int(settings["heatmap_size"]),
            settings.get("head_hidden_channels"),
            int(settings.get("psax_norm_groups", 32)),
        )

    def forward(self, image: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        if str(task_id) == "PSAX":
            output = self.heads["PSAX"](features)
            output["features"] = features
            return output
        return {
            "heatmap_logits": self.heads[str(task_id)](features),
            "features": features,
        }


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    return Exp152Model(settings, task_configs)


def soft_argmax(logits: torch.Tensor, beta: float) -> torch.Tensor:
    batch, points, height, width = logits.shape
    probability = torch.softmax(logits.reshape(batch, points, -1) * float(beta), dim=-1)
    probability = probability.reshape(batch, points, height, width)
    y_axis = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
    x_axis = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x = (probability * xx[None, None]).sum(dim=(-2, -1))
    y = (probability * yy[None, None]).sum(dim=(-2, -1))
    return torch.stack([x, y], dim=-1)


def decode_topk(logits: torch.Tensor, topk: int, beta: float) -> torch.Tensor:
    batch, points, height, width = logits.shape
    values, indices = torch.topk(
        logits.reshape(batch, points, -1),
        k=min(max(int(topk), 1), height * width),
        dim=-1,
    )
    weights = torch.softmax(values * float(beta), dim=-1)
    x = (indices % width).to(logits.dtype) / max(float(width - 1), 1.0)
    y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    y = y / max(float(height - 1), 1.0)
    return torch.stack([(weights * x).sum(-1), (weights * y).sum(-1)], dim=-1)


def _sample_map(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample [B,C,H,W] at normalized [B,N,2] points."""
    grid = points.mul(2.0).sub(1.0).unsqueeze(2)
    sampled = F.grid_sample(values, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


def _local_peak_candidates(
    logits: torch.Tensor,
    count: int,
    nms_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, points, height, width = logits.shape
    kernel = max(int(nms_kernel), 1)
    if kernel % 2 == 0:
        kernel += 1
    maxima = F.max_pool2d(logits, kernel, stride=1, padding=kernel // 2)
    suppressed = logits.masked_fill(logits < maxima, -torch.inf)
    values, indices = torch.topk(
        suppressed.reshape(batch, points, -1),
        k=min(max(int(count), 1), height * width),
        dim=-1,
    )
    x = (indices % width).to(logits.dtype) / max(float(width - 1), 1.0)
    y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    y = y / max(float(height - 1), 1.0)
    return torch.stack([x, y], dim=-1), values


def decode_psax_structured(
    output: dict[str, torch.Tensor],
    settings: dict[str, Any],
) -> torch.Tensor:
    logits = output["heatmap_logits"]
    candidates, candidate_logits = _local_peak_candidates(
        logits,
        int(settings.get("psax_peak_candidates", 8)),
        int(settings.get("psax_peak_nms_kernel", 5)),
    )
    midpoint_logits = output["midpoint_logits"]
    tube_logits = output["tube_logits"]
    direction = output["direction_field"]
    batch, _, candidate_count, _ = candidates.shape
    decoded = logits.new_zeros((batch, 4, 2))
    line_t = torch.linspace(0.0, 1.0, 9, device=logits.device, dtype=logits.dtype)
    min_length = float(settings.get("psax_min_pair_length_norm", 0.03))
    max_length = float(settings.get("psax_max_pair_length_norm", 0.90))

    for pair_index, (first_index, second_index) in enumerate(PSAX_PAIRS):
        first = candidates[:, first_index, :, None, :].expand(-1, -1, candidate_count, -1)
        second = candidates[:, second_index, None, :, :].expand(-1, candidate_count, -1, -1)
        first = first.reshape(batch, -1, 2)
        second = second.reshape(batch, -1, 2)
        vector = second - first
        length = torch.linalg.norm(vector, dim=-1)
        midpoint = 0.5 * (first + second)

        first_scores = torch.log_softmax(candidate_logits[:, first_index], dim=-1)
        second_scores = torch.log_softmax(candidate_logits[:, second_index], dim=-1)
        endpoint_score = (
            first_scores[:, :, None] + second_scores[:, None, :]
        ).reshape(batch, -1)
        midpoint_score = F.logsigmoid(
            _sample_map(midpoint_logits[:, pair_index : pair_index + 1], midpoint).squeeze(-1)
        )

        line_points = first[:, :, None, :] + line_t[None, None, :, None] * vector[:, :, None, :]
        flattened_line = line_points.reshape(batch, -1, 2)
        tube_samples = _sample_map(
            tube_logits[:, pair_index : pair_index + 1], flattened_line
        ).reshape(batch, -1, line_t.numel())
        tube_score = torch.log(torch.sigmoid(tube_samples).mean(-1).clamp_min(1e-6))

        direction_map = direction[:, pair_index]
        direction_samples = _sample_map(direction_map, flattened_line).reshape(
            batch, -1, line_t.numel(), 2
        )
        predicted_direction = F.normalize(direction_samples.mean(2), dim=-1, eps=1e-6)
        candidate_direction = F.normalize(vector, dim=-1, eps=1e-6)
        direction_score = (predicted_direction * candidate_direction).sum(-1)

        score = (
            endpoint_score
            + float(settings.get("psax_score_midpoint_weight", 0.5)) * midpoint_score
            + float(settings.get("psax_score_tube_weight", 0.5)) * tube_score
            + float(settings.get("psax_score_direction_weight", 0.25)) * direction_score
        )
        valid = (
            (first[..., 0] > second[..., 0])
            & (length >= min_length)
            & (length <= max_length)
        )
        valid_score = score.masked_fill(~valid, -torch.inf)
        best = valid_score.argmax(dim=-1)
        no_valid = ~valid.any(dim=-1)
        if no_valid.any():
            best = torch.where(no_valid, score.argmax(dim=-1), best)
        gather_index = best[:, None, None].expand(-1, 1, 2)
        decoded[:, first_index] = first.gather(1, gather_index).squeeze(1)
        decoded[:, second_index] = second.gather(1, gather_index).squeeze(1)
    return decoded.clamp(0.0, 1.0)


def forward_train(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    output = model(images, task_id=str(task_id))
    output["coords_norm"] = soft_argmax(
        output["heatmap_logits"], float(settings.get("train_softargmax_beta", 4.0))
    )
    return output


def forward_inference(
    model: nn.Module,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    # Generic official inference passes an unused phase before settings.
    if settings is None:
        settings = phase_or_settings
    output = model(images, task_id=str(task_id))
    endpoint = decode_topk(
        output["heatmap_logits"],
        int(settings.get("decode_topk", 25)),
        float(settings.get("decode_topk_beta", 1.0)),
    )
    output["coords_norm"] = endpoint
    output["structured_coords_norm"] = (
        decode_psax_structured(output, settings) if str(task_id) == "PSAX" else endpoint
    )
    return output


def _gaussian_targets(points: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    y_axis = torch.arange(size, device=points.device, dtype=points.dtype)
    x_axis = torch.arange(size, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x = points[..., 0] * float(size - 1)
    y = points[..., 1] * float(size - 1)
    distance = (xx - x[..., None, None]).square() + (yy - y[..., None, None]).square()
    return torch.exp(-distance / (2.0 * float(sigma) ** 2))


def _pair_targets(
    points: torch.Tensor,
    size: int,
    midpoint_sigma: float,
    tube_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = points[:, (0, 2)]
    second = points[:, (1, 3)]
    midpoint = 0.5 * (first + second)
    midpoint_target = _gaussian_targets(midpoint, size, midpoint_sigma)

    axis = torch.linspace(0.0, 1.0, size, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pixels = torch.stack([xx, yy], dim=-1)[None, None]
    segment = second - first
    relative = pixels - first[:, :, None, None, :]
    denominator = segment.square().sum(-1).clamp_min(1e-8)
    projection = (
        (relative * segment[:, :, None, None, :]).sum(-1)
        / denominator[:, :, None, None]
    ).clamp(0.0, 1.0)
    nearest = first[:, :, None, None, :] + projection[..., None] * segment[:, :, None, None, :]
    distance_px = torch.linalg.norm(pixels - nearest, dim=-1) * float(size - 1)
    tube_target = torch.exp(-distance_px.square() / (2.0 * float(tube_sigma) ** 2))
    direction_target = F.normalize(segment, dim=-1, eps=1e-6)
    return midpoint_target, tube_target, direction_target


def _dense_auxiliary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    settings: dict[str, Any],
) -> torch.Tensor:
    batch, channels, height, width = logits.shape
    distribution = target.reshape(batch, channels, -1)
    distribution = distribution / distribution.sum(-1, keepdim=True).clamp_min(1e-8)
    spatial_ce = -(
        distribution * torch.log_softmax(logits.reshape(batch, channels, -1), dim=-1)
    ).sum(-1).mean()
    spatial_ce = spatial_ce / float(height * width)
    weight = 1.0 + float(settings.get("psax_aux_foreground_weight", 20.0)) * target
    weighted_mse = (weight * (torch.sigmoid(logits) - target).square()).mean()
    return (
        float(settings.get("psax_aux_spatial_ce_weight", 1.0)) * spatial_ce
        + float(settings.get("psax_aux_weighted_mse_weight", 0.2)) * weighted_mse
    )


def compute_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    phase: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del phase
    endpoint = F.mse_loss(torch.sigmoid(output["heatmap_logits"]), target["heatmap"])
    zero = endpoint.detach() * 0.0
    if str(task_id) != "PSAX":
        return endpoint, {
            "total_loss": endpoint,
            "heatmap_loss": endpoint,
            "midpoint_loss": zero,
            "tube_loss": zero,
            "direction_loss": zero,
        }
    size = int(output["heatmap_logits"].shape[-1])
    midpoint_target, tube_target, direction_target = _pair_targets(
        target["points_norm"],
        size,
        float(settings.get("psax_midpoint_sigma", 2.2)),
        float(settings.get("psax_tube_sigma", 1.8)),
    )
    midpoint = _dense_auxiliary_loss(output["midpoint_logits"], midpoint_target, settings)
    tube = _dense_auxiliary_loss(output["tube_logits"], tube_target, settings)
    predicted_direction = F.normalize(output["direction_field"], dim=2, eps=1e-6)
    cosine = (
        predicted_direction
        * direction_target[:, :, :, None, None]
    ).sum(dim=2)
    direction = ((1.0 - cosine) * tube_target).mean()
    total = (
        endpoint
        + float(settings.get("psax_midpoint_loss_weight", 0.1)) * midpoint
        + float(settings.get("psax_tube_loss_weight", 0.1)) * tube
        + float(settings.get("psax_direction_loss_weight", 0.05)) * direction
    )
    return total, {
        "total_loss": total,
        "heatmap_loss": endpoint,
        "midpoint_loss": midpoint,
        "tube_loss": tube,
        "direction_loss": direction,
    }


def configure_trainable(model: nn.Module, scope: str) -> dict[str, int]:
    if str(scope) != "all":
        raise ValueError("Exp152 Exp56 stage only supports train_scope=all.")
    for parameter in model.parameters():
        parameter.requires_grad = True
    return {
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "frozen_parameters": 0,
    }


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def optimizer_groups(model: nn.Module, phase: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "params": _trainable(model.encoder.parameters()),
            "lr": float(phase["encoder_lr"]),
            "name": "encoder",
        },
        {
            "params": _trainable(model.heads.parameters()),
            "lr": float(phase["head_lr"]),
            "name": "heads",
        },
    ]


def set_train_mode(model: nn.Module, scope: str) -> None:
    if str(scope) != "all":
        raise ValueError("Exp152 Exp56 stage only supports train_scope=all.")
    model.train()


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
