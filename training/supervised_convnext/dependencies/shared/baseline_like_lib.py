from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


RACE_SHARED_DIR = Path(__file__).resolve().parents[2] / "11-e2e-roialign-cascade" / "src"
if str(RACE_SHARED_DIR) not in sys.path:
    sys.path.append(str(RACE_SHARED_DIR))

from foundus_race_lib import (  # noqa: E402
    ConvNormAct,
    DINOv2FeatureBackbone,
    HeatmapDecoderHead,
    decode_argmax_norm,
    sample_feature_at_points,
)


class BasicDINOHeatmapModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        task_configs: list[dict[str, Any]],
        input_size: int,
        heatmap_size: int,
        head_hidden_channels: int | None = None,
    ):
        super().__init__()
        pretrained = encoder_weights is not None and str(encoder_weights).lower() not in {"", "none", "false"}
        self.encoder = DINOv2FeatureBackbone(encoder_name, pretrained=pretrained, img_size=int(input_size))
        self.heads = nn.ModuleDict()
        for config in task_configs:
            self.heads[str(config["task_id"])] = HeatmapDecoderHead(
                self.encoder.out_channels,
                int(config["num_classes"]),
                int(heatmap_size),
                head_hidden_channels,
            )

    def forward(self, image: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        logits = self.heads[str(task_id)](features)
        return {"heatmap_logits": logits, "features": features}


class ConvNeXtFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        timm = __import__("timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(-1,))
        self.out_channels = int(self.backbone.feature_info.channels()[-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)[-1]


class ConvNeXtHeatmapModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        task_configs: list[dict[str, Any]],
        heatmap_size: int,
        head_hidden_channels: int | None = None,
    ):
        super().__init__()
        pretrained = encoder_weights is not None and str(encoder_weights).lower() not in {"", "none", "false"}
        self.encoder = ConvNeXtFeatureBackbone(encoder_name, pretrained=pretrained)
        self.heads = nn.ModuleDict()
        for config in task_configs:
            self.heads[str(config["task_id"])] = HeatmapDecoderHead(
                self.encoder.out_channels,
                int(config["num_classes"]),
                int(heatmap_size),
                head_hidden_channels,
            )

    def forward(self, image: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        logits = self.heads[str(task_id)](features)
        return {"heatmap_logits": logits, "features": features}


class DynamicDINOFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, img_size: int):
        super().__init__()
        timm = __import__("timm")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=int(img_size),
            dynamic_img_size=True,
        )
        self.out_channels = int(self.backbone.num_features)

    def forward_tokens(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(image)
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                return feats["x_norm_patchtokens"]
            if "x_prenorm" in feats:
                return feats["x_prenorm"][:, 1:, :]
            raise RuntimeError(f"Unsupported DINOv2 feature dict keys: {sorted(feats.keys())}")
        if isinstance(feats, torch.Tensor):
            return feats[:, 1:, :]
        raise RuntimeError(f"Unexpected DINOv2 feature type: {type(feats)!r}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.forward_tokens(image)
        batch, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square.")
        return patch_tokens.transpose(1, 2).reshape(batch, channels, side, side)


def decode_heatmap(outputs: dict[str, torch.Tensor], settings: dict[str, Any], task_id: str) -> torch.Tensor:
    return decode_argmax_norm(outputs["heatmap_logits"])


def heatmap_optimizer_groups(model: nn.Module, base_lr: float, settings: dict[str, Any]) -> list[dict[str, Any]]:
    encoder_lr = base_lr * float(settings.get("encoder_learning_rate_multiplier", 0.2))
    head_lr = base_lr * float(settings.get("head_learning_rate_multiplier", 10.0))
    head_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    return [
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ]


def expectation_from_logits_1d(logits: torch.Tensor) -> torch.Tensor:
    bins = logits.shape[-1]
    prob = F.softmax(logits, dim=-1)
    coords = torch.linspace(0.0, 1.0, bins, device=logits.device, dtype=logits.dtype)
    return (prob * coords).sum(dim=-1)


def local_residual_from_features(
    features: torch.Tensor,
    anchors: torch.Tensor,
    residual_head: nn.Module,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = sample_feature_at_points(features, anchors)
    residual = torch.tanh(residual_head(torch.cat([sampled, anchors], dim=-1))) * float(residual_scale)
    coords = (anchors + residual).clamp(0.0, 1.0)
    return coords, residual


def bbox_mask_from_norm(bbox_norm: torch.Tensor, height: int, width: int, softness: float = 0.05) -> torch.Tensor:
    device = bbox_norm.device
    dtype = bbox_norm.dtype
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    center = bbox_norm[:, 0:2]
    size = bbox_norm[:, 2:4].clamp_min(1e-4)
    half = 0.5 * size
    dx = (torch.abs(xx[None] - center[:, None, None, 0]) - half[:, None, None, 0]) / max(float(softness), 1e-6)
    dy = (torch.abs(yy[None] - center[:, None, None, 1]) - half[:, None, None, 1]) / max(float(softness), 1e-6)
    outside = torch.maximum(dx, dy)
    return torch.sigmoid(-outside).unsqueeze(1)
