from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP161_MODEL_PATH = PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/model.py"
MODEL_NAME = "Exp164-Task-Conditioned-Unlabeled-Appearance-Replay"


def _load_exp161_model():
    name = "exp161_model_for_exp164"
    spec = importlib.util.spec_from_file_location(name, EXP161_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(EXP161_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_exp161 = _load_exp161_model()


class TaskResidualAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 64, groups: int = 8):
        super().__init__()
        groups = min(int(groups), int(bottleneck))
        while int(bottleneck) % groups != 0 and groups > 1:
            groups -= 1
        self.down = nn.Conv2d(int(channels), int(bottleneck), 1)
        self.norm = nn.GroupNorm(groups, int(bottleneck))
        self.depthwise = nn.Conv2d(
            int(bottleneck),
            int(bottleneck),
            3,
            padding=1,
            groups=int(bottleneck),
        )
        self.up = nn.Conv2d(int(bottleneck), int(channels), 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        features: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.norm(self.down(features)))
        hidden = F.gelu(self.depthwise(hidden))
        delta = self.up(hidden) * float(scale)
        return features + delta, delta


class Exp164Model(nn.Module):
    def __init__(self, settings: dict[str, Any], task_configs: list[dict[str, Any]]):
        super().__init__()
        anchor = _exp161.build_model(settings, task_configs)
        self.encoder = anchor.encoder
        self.heads = anchor.heads
        self.task_ids = [str(config["task_id"]) for config in task_configs]
        channels = int(self.encoder.out_channels)
        bottleneck = int(settings.get("adapter_bottleneck", 64))
        groups = int(settings.get("adapter_groups", 8))
        self.task_adapters = nn.ModuleDict(
            {
                task_id: TaskResidualAdapter(channels, bottleneck, groups)
                for task_id in self.task_ids
            }
        )
        self.adapter_scale = 1.0

    def set_adapter_scale(self, scale: float) -> None:
        self.adapter_scale = float(scale)

    def _head(self, features: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        output = self.heads[str(task_id)](features)
        if isinstance(output, torch.Tensor):
            return {"heatmap_logits": output, "features": features}
        result = dict(output)
        result["features"] = features
        return result

    def _adapt(
        self,
        features: torch.Tensor,
        task_id: str,
        adapter_enabled: bool,
        adapter_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not adapter_enabled or abs(float(adapter_scale)) < 1e-12:
            return features, torch.zeros_like(features)
        return self.task_adapters[str(task_id)](features, float(adapter_scale))

    def forward(
        self,
        image: torch.Tensor,
        task_id: str,
        adapter_enabled: bool = True,
        adapter_scale: float | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        task = str(task_id)
        features = self.encoder(image)
        scale = self.adapter_scale if adapter_scale is None else float(adapter_scale)
        adapted, delta = self._adapt(features, task, adapter_enabled, scale)
        output = self._head(adapted, task)
        output["base_features"] = features
        output["adapter_delta"] = delta
        return output

    def forward_with_anchor(
        self,
        image: torch.Tensor,
        task_id: str,
        adapter_scale: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        task = str(task_id)
        features = self.encoder(image)
        with torch.no_grad():
            anchor_output = self._head(features, task)
        adapted, delta = self._adapt(features, task, True, adapter_scale)
        adapted_output = self._head(adapted, task)
        adapted_output["base_features"] = features
        adapted_output["adapter_delta"] = delta
        return adapted_output, anchor_output


def build_model(
    settings: dict[str, Any],
    task_configs: list[dict[str, Any]],
) -> Exp164Model:
    return Exp164Model(settings, task_configs)


def train_adapter_only(network: Exp164Model) -> None:
    network.train()
    network.encoder.eval()
    network.heads.eval()
    for parameter in network.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in network.heads.parameters():
        parameter.requires_grad_(False)
    network.task_adapters.train()
    for parameter in network.task_adapters.parameters():
        parameter.requires_grad_(True)


def adapter_parameters(network: Exp164Model) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in network.task_adapters.parameters()
        if parameter.requires_grad
    ]


def forward_inference(
    network: Exp164Model,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if settings is None:
        settings = phase_or_settings
    scale = float(settings.get("_evaluation_adapter_scale", 1.0))
    previous = float(network.adapter_scale)
    network.set_adapter_scale(scale)
    try:
        return _exp161.forward_inference(network, images, task_id, settings)
    finally:
        network.set_adapter_scale(previous)


def compute_supervised_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _exp161.compute_loss(output, target, task_id, {}, settings)


canonicalize_internal_points = _exp161.canonicalize_internal_points
canonicalize_training_target = _exp161.canonicalize_training_target
sort_official_vertical = _exp161.sort_official_vertical
endpoint_identity_policy = _exp161.endpoint_identity_policy

