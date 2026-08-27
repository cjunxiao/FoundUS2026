from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(int(channels), int(maximum))
    while int(channels) % groups and groups > 1:
        groups -= 1
    return groups


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=int(dilation),
                dilation=int(dilation),
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class ContextualTaskResidualHead(nn.Module):
    DILATIONS = (1, 2, 4, 8)

    def __init__(
        self,
        shared_channels: int,
        num_points: int,
        hidden_channels: int,
        logit_bound: float,
    ) -> None:
        super().__init__()
        input_channels = int(shared_channels) + 3 * int(num_points) + 2
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            *[
                DilatedContextBlock(hidden_channels, dilation)
                for dilation in self.DILATIONS
            ]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels, int(num_points), 1)
        self.logit_bound = float(logit_bound)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _center(logits: torch.Tensor) -> torch.Tensor:
        return logits - logits.amax((-2, -1), keepdim=True)

    @staticmethod
    def _coordinates(reference: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(reference.shape[0], -1, -1, -1)

    def forward(
        self,
        shared_feature: torch.Tensor,
        convnext_logits: torch.Tensor,
        foundation_logits: torch.Tensor,
        anchor_logits: torch.Tensor,
    ) -> torch.Tensor:
        evidence = torch.cat(
            (
                shared_feature,
                self._center(convnext_logits),
                self._center(foundation_logits),
                self._center(anchor_logits),
                self._coordinates(anchor_logits),
            ),
            dim=1,
        )
        hidden = self.input_projection(evidence)
        hidden = self.context(hidden)
        hidden = hidden + self.global_context(hidden)
        raw = self.output(hidden)
        return self.logit_bound * torch.tanh(raw)


class FunctionalCompressionStudent(nn.Module):
    def __init__(
        self,
        anchor: nn.Module,
        task_configs: list[dict[str, Any]],
        task_scales: dict[str, float],
        shared_channels: int = 128,
        hidden_channels: int = 64,
        logit_bound: float = 2.0,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        self.task_scales = {str(key): float(value) for key, value in task_scales.items()}
        task_points = {
            str(item["task_id"]): int(item["num_classes"]) for item in task_configs
        }
        if set(task_points) != set(self.task_scales):
            raise RuntimeError("Task scale map does not match the anchor task set.")
        self.residual_heads = nn.ModuleDict(
            {
                task: ContextualTaskResidualHead(
                    shared_channels,
                    points,
                    hidden_channels,
                    logit_bound,
                )
                for task, points in task_points.items()
            }
        )
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def anchor_logits(
        self, output: dict[str, torch.Tensor], task_id: str
    ) -> torch.Tensor:
        task = str(task_id)
        return output["base_heatmap_logits"] + self.task_scales[task] * output[
            "residual_heatmap_logits"
        ]

    def forward(
        self,
        image: torch.Tensor,
        task_id: str,
        residual_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        task = str(task_id)
        with torch.no_grad():
            output = self.anchor(image, task)
        anchor_logits = self.anchor_logits(output, task).detach()
        if abs(float(residual_scale)) < 1e-12:
            residual = torch.zeros_like(anchor_logits)
        else:
            residual = self.residual_heads[task](
                output["features"].detach(),
                output["base_heatmap_logits"].detach(),
                output["foundation_heatmap_logits"].detach(),
                anchor_logits,
            )
        return {
            "heatmap_logits": anchor_logits + float(residual_scale) * residual,
            "anchor_heatmap_logits": anchor_logits,
            "residual_heatmap_logits": residual,
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.residual_heads.parameters()
            if parameter.requires_grad
        ]

