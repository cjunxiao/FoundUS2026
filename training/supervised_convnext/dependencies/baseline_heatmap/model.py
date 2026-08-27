from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RACE_SRC = PROJECT_ROOT / "training/supervised_convnext/dependencies/shared"
BASELINE_SHARED_SRC = PROJECT_ROOT / "training/supervised_convnext/dependencies/shared"
for path in [RACE_SRC, BASELINE_SHARED_SRC]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from baseline_like_lib import decode_heatmap, heatmap_optimizer_groups  # noqa: E402
from foundus_race_lib import HeatmapDecoderHead, resolve_project_path  # noqa: E402


MODEL_NAME = "BaselineHeatmap-ConvNeXt-Small-Heatmap"


def clean_weight_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw.lower() in {"", "none", "false", "null"}:
        return None
    return raw


def is_local_weight_path(value: Any) -> bool:
    raw = clean_weight_value(value)
    return bool(raw and raw.lower() not in {"pretrained", "true"} and resolve_project_path(raw).exists())


def load_weight_file(path: str | Path) -> dict[str, torch.Tensor]:
    resolved = resolve_project_path(path)
    if resolved.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load local ConvNeXt weights.") from exc
        state = load_file(str(resolved))
    else:
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(resolved, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            state = payload["model"]
        else:
            state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported weight file payload: {resolved}")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value
    return normalized


class LocalWeightConvNeXtFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, encoder_weights: str | None):
        super().__init__()
        timm = __import__("timm")
        raw_weights = clean_weight_value(encoder_weights)
        local_path = resolve_project_path(raw_weights) if is_local_weight_path(raw_weights) else None
        pretrained = bool(raw_weights and raw_weights.lower() in {"pretrained", "true"})
        self.backbone = timm.create_model(
            str(model_name),
            pretrained=pretrained and local_path is None,
            features_only=True,
            out_indices=(-1,),
        )
        self.out_channels = int(self.backbone.feature_info.channels()[-1])
        self.load_info: dict[str, Any] = {
            "model_name": str(model_name),
            "encoder_weights": raw_weights,
            "pretrained_timm": bool(pretrained and local_path is None),
            "local_weight_path": str(local_path) if local_path else None,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if local_path is not None:
            state = load_weight_file(local_path)
            result = self.backbone.load_state_dict(state, strict=False)
            self.load_info.update(
                {
                    "pretrained_timm": False,
                    "local_weight_path": str(local_path),
                    "missing_keys": list(result.missing_keys),
                    "unexpected_keys": list(result.unexpected_keys),
                }
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)[-1]


class ConvNeXtSmallHeatmapModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        task_configs: list[dict[str, Any]],
        heatmap_size: int,
        head_hidden_channels: int | None = None,
    ):
        super().__init__()
        self.encoder = LocalWeightConvNeXtFeatureBackbone(encoder_name, encoder_weights)
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


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    return ConvNeXtSmallHeatmapModel(
        encoder_name=str(settings["encoder"]),
        encoder_weights=settings.get("encoder_weights"),
        task_configs=task_configs,
        heatmap_size=int(settings["heatmap_size"]),
        head_hidden_channels=settings.get("head_hidden_channels"),
    )


def decode_outputs(outputs, settings, task_id):
    return decode_heatmap(outputs, settings, task_id)


def optimizer_param_groups(model: nn.Module, base_lr: float, settings: dict[str, Any]):
    return heatmap_optimizer_groups(model, base_lr, settings)
