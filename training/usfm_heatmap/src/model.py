from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from usfm_backbone import PROJECT_ROOT, USFMBackbone


MODEL_NAME = "USFMHeatmap-USFM-ViTB16-TaskPrivate-ViTPose-Heatmap"
IDENTITY_SOURCE = PROJECT_ROOT / "training/supervised_convnext/dependencies/canonical_identity/model.py"


def _load_identity_helpers():
    name = "canonical_identity_identity_for_usfm_heatmap"
    spec = importlib.util.spec_from_file_location(name, IDENTITY_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(IDENTITY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_identity = _load_identity_helpers()
canonicalize_training_target = _identity.canonicalize_training_target
canonicalize_internal_points = _identity.canonicalize_internal_points
sort_official_vertical = _identity.sort_official_vertical


class ViTPoseHeatmapDecoder(nn.Module):
    def __init__(self, in_channels: int, channels: int, norm_groups: int):
        super().__init__()
        groups = min(int(norm_groups), int(channels))
        while channels % groups and groups > 1:
            groups -= 1
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, channels, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TaskPrivateViTPoseHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        norm_groups: int,
        num_classes: int,
    ):
        super().__init__()
        self.decoder = ViTPoseHeatmapDecoder(in_channels, channels, norm_groups)
        self.output = nn.Conv2d(channels, num_classes, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.decoder(value)
        return self.output(feature), feature


class USFMViTPose(nn.Module):
    def __init__(self, settings: dict[str, Any], task_configs: list[dict[str, Any]]):
        super().__init__()
        self.encoder = USFMBackbone(settings)
        channels = int(settings.get("decoder_channels", 256))
        self.decoder = nn.Identity()
        self.heads = nn.ModuleDict(
            {
                str(item["task_id"]): TaskPrivateViTPoseHead(
                    self.encoder.out_channels,
                    channels,
                    int(settings.get("decoder_norm_groups", 32)),
                    int(item["num_classes"]),
                )
                for item in task_configs
            }
        )

    def forward(self, image: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        logits, feature = self.heads[str(task_id)](self.encoder(image))
        return {"heatmap_logits": logits, "features": feature}


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    if int(settings.get("input_size", 256)) != 256:
        raise ValueError("USFMHeatmap fixes input_size=256 for a 16x16 token grid and 64x64 heatmaps.")
    if int(settings.get("heatmap_size", 64)) != 64:
        raise ValueError("USFMHeatmap fixes heatmap_size=64.")
    return USFMViTPose(settings, task_configs)


def _soft_argmax(logits: torch.Tensor, beta: float) -> torch.Tensor:
    batch, points, height, width = logits.shape
    probability = F.softmax(logits.flatten(2) * float(beta), dim=-1).reshape_as(logits)
    ys = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack(
        [(probability * xx).sum((-2, -1)), (probability * yy).sum((-2, -1))],
        dim=-1,
    )


def decode_topk(logits: torch.Tensor, topk: int, beta: float) -> torch.Tensor:
    batch, points, height, width = logits.shape
    values, indices = torch.topk(logits.flatten(2), min(max(int(topk), 1), height * width), dim=-1)
    weights = torch.softmax(values * float(beta), dim=-1)
    x = (indices % width).to(logits.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    return torch.stack(
        [(weights * x).sum(-1) / max(width - 1, 1), (weights * y).sum(-1) / max(height - 1, 1)],
        dim=-1,
    ).reshape(batch, points, 2)


def forward_train(
    network: USFMViTPose,
    images: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    output = network(images, task_id)
    output["coords_norm"] = _soft_argmax(
        output["heatmap_logits"], float(settings.get("train_softargmax_beta", 4.0))
    )
    return output


def forward_inference(
    network: USFMViTPose,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if settings is None:
        settings = phase_or_settings
    output = network(images, task_id)
    internal = decode_topk(
        output["heatmap_logits"],
        int(settings.get("decode_topk", 25)),
        float(settings.get("decode_topk_beta", 1.0)),
    )
    output["internal_coords_norm"] = internal
    output["coords_norm"] = sort_official_vertical(internal, task_id, settings)
    return output


def _length(points: torch.Tensor, first: int, second: int) -> torch.Tensor:
    return torch.linalg.norm(points[:, second] - points[:, first], dim=-1).clamp_min(1e-6)


def _measurement_loss(prediction: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    pairs = {
        "A4C": list(zip(range(0, 16, 2), range(1, 16, 2))),
        "FUGC": [(0, 1)],
        "IVC": [(0, 1)],
        "PLAX": list(zip(range(0, 22, 2), range(1, 22, 2))),
        "PSAX": [(0, 1), (2, 3)],
        "fetal_femur": [(0, 1)],
    }
    if task in pairs:
        pred = torch.stack([_length(prediction, a, b) for a, b in pairs[task]], -1)
        truth = torch.stack([_length(target, a, b) for a, b in pairs[task]], -1)
        return F.smooth_l1_loss(pred / truth.detach(), torch.ones_like(truth), beta=0.02)
    if task in {"HC", "FA"}:
        pred = torch.stack([_length(prediction, 0, 1), _length(prediction, 2, 3)], -1)
        truth = torch.stack([_length(target, 0, 1), _length(target, 2, 3)], -1)
        return F.smooth_l1_loss(pred / truth.detach(), torch.ones_like(truth), beta=0.02)
    if task == "AOP":
        first_pred = F.normalize(prediction[:, 1] - prediction[:, 0], dim=-1, eps=1e-6)
        second_pred = F.normalize(prediction[:, 3] - prediction[:, 0], dim=-1, eps=1e-6)
        first_gt = F.normalize(target[:, 1] - target[:, 0], dim=-1, eps=1e-6)
        second_gt = F.normalize(target[:, 3] - target[:, 0], dim=-1, eps=1e-6)
        angle = F.smooth_l1_loss(
            (first_pred * second_pred).sum(-1), (first_gt * second_gt).sum(-1), beta=0.02
        )
        hsd = F.smooth_l1_loss(
            _length(prediction, 0, 2) / _length(target, 0, 2).detach(),
            torch.ones_like(_length(target, 0, 2)),
            beta=0.02,
        )
        return angle + hsd
    return prediction.sum() * 0.0


def task_relevant_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = torch.sigmoid(logits)
    pixel_weight = 1.0 + float(settings.get("heatmap_foreground_weight", 20.0)) * target
    weighted_mse = (
        pixel_weight * (prediction - target).square()
    ).sum() / pixel_weight.sum().clamp_min(1.0)
    target_probability = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    temperature = float(settings.get("heatmap_ce_temperature", 1.0))
    log_probability = F.log_softmax(logits.flatten(2) / temperature, dim=-1).reshape_as(logits)
    spatial_ce = -(target_probability * log_probability).sum(dim=(-2, -1)).mean()
    spatial_ce = spatial_ce * temperature**2
    sigma = float(settings.get("heatmap_sigma", 1.8))
    radius = float(settings.get("heatmap_support_radius", 5.0))
    threshold = math.exp(-(radius**2) / (2.0 * sigma**2))
    support = target >= threshold
    probability = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
    support_mass = (probability * support.to(probability.dtype)).sum(dim=(-2, -1))
    support_mass_loss = -support_mass.clamp_min(1e-8).log().mean()
    heatmap = (
        float(settings.get("heatmap_mse_weight", 0.2)) * weighted_mse
        + float(settings.get("heatmap_ce_weight", 1.0)) * spatial_ce
    )
    return heatmap, {
        "heatmap_loss": heatmap,
        "heatmap_weighted_mse_loss": weighted_mse,
        "heatmap_ce_loss": spatial_ce,
        "support_mass_loss": support_mass_loss,
        "gt_support_mass": support_mass.mean(),
    }


def compute_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    phase: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    internal_target = canonicalize_training_target(target, task_id, settings)
    logits = output["heatmap_logits"]
    heatmap, heatmap_parts = task_relevant_heatmap_loss(
        logits, internal_target["heatmap"], settings
    )
    points = internal_target["points_norm"]
    coordinates = _soft_argmax(logits, float(settings.get("train_softargmax_beta", 4.0)))
    coordinate = F.smooth_l1_loss(coordinates, points, beta=0.02)
    measurement = _measurement_loss(coordinates, points, str(task_id))
    measurement_weight = float(settings.get("_measurement_weight_current", phase.get("measurement_weight", 0.0)))
    total = (
        float(phase.get("heatmap_weight", 1.0)) * heatmap
        + float(settings.get("heatmap_support_mass_weight", 0.1))
        * heatmap_parts["support_mass_loss"]
        + float(phase.get("coordinate_weight", 0.0)) * coordinate
        + measurement_weight * measurement
    )
    return total, {
        "total_loss": total,
        **heatmap_parts,
        "coordinate_loss": coordinate,
        "measurement_loss": measurement,
    }


def configure_trainable(network: USFMViTPose, scope: str) -> dict[str, int]:
    for parameter in network.parameters():
        parameter.requires_grad = False
    scope = str(scope)
    if scope == "decoder_only":
        count = 0
    elif scope.startswith("last"):
        count = int(re.sub(r"\D", "", scope))
    elif scope == "all":
        count = len(network.encoder.backbone.blocks)
        for parameter in network.encoder.backbone.patch_embed.parameters():
            parameter.requires_grad = True
        network.encoder.backbone.cls_token.requires_grad = True
        for parameter in network.encoder.backbone.rel_pos_bias.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown USFMHeatmap train scope: {scope}")
    network.encoder.set_trainable_last_blocks(count)
    if scope == "all":
        for parameter in network.encoder.parameters():
            parameter.requires_grad = True
    for module in (network.decoder, network.heads):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return {
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in network.parameters() if not p.requires_grad),
        "trainable_backbone_blocks": count,
    }


def optimizer_groups(network: USFMViTPose, phase: dict[str, Any]) -> list[dict[str, Any]]:
    encoder_parameters = [p for p in network.encoder.parameters() if p.requires_grad]
    head_parameters = [
        p for module in (network.decoder, network.heads) for p in module.parameters() if p.requires_grad
    ]
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": float(phase.get("encoder_lr", 0.0)), "name": "encoder"})
    groups.append({"params": head_parameters, "lr": float(phase["head_lr"]), "name": "head"})
    return groups


def set_train_mode(network: USFMViTPose, scope: str) -> None:
    network.train()
    if str(scope) == "decoder_only":
        network.encoder.eval()


def trainable_parameter_names(network: nn.Module) -> list[str]:
    return [name for name, parameter in network.named_parameters() if parameter.requires_grad]


def trainable_parameters(network: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in network.parameters() if parameter.requires_grad)


def model_metadata(network: USFMViTPose) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "backbone": "USFM BEiT ViT-B/16",
        "decoder": "nine task-private two-stage ViTPose deconvolution heads",
        "encoder_load_info": network.encoder.load_info,
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
    }
