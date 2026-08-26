from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_SIZE = 518
HEATMAP_SIZE = 64
TASK_POINTS = {
    "A4C": 16,
    "AOP": 4,
    "FA": 4,
    "FUGC": 2,
    "HC": 4,
    "IVC": 2,
    "PLAX": 22,
    "PSAX": 4,
    "fetal_femur": 2,
}
TASK_SCALES = {
    "A4C": 0.0,
    "AOP": 0.75,
    "FA": 0.75,
    "FUGC": 0.0,
    "HC": 0.5,
    "IVC": 1.0,
    "PLAX": 0.0,
    "PSAX": 1.0,
    "fetal_femur": 1.0,
}
VERTICAL_ORDER_TASKS = {"A4C", "PSAX"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ConvNeXtFeatureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "convnext_small.in12k_ft_in1k",
            pretrained=False,
            features_only=True,
            out_indices=(-1,),
        )
        self.out_channels = int(self.backbone.feature_info.channels()[-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)[-1]


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class HeatmapDecoderHead(nn.Module):
    def __init__(self, in_channels: int, num_points: int) -> None:
        super().__init__()
        hidden = max(in_channels // 2, 192)
        mid = max(hidden // 2, 96)
        self.heatmap_size = HEATMAP_SIZE
        self.trunk = nn.Sequential(
            ConvNormAct(in_channels, hidden),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvNormAct(hidden, mid),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvNormAct(mid, mid),
        )
        self.heatmap_logits = nn.Conv2d(mid, num_points, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(features)
        if hidden.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            hidden = F.interpolate(
                hidden,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.heatmap_logits(hidden)


def group_count(channels: int, maximum: int) -> int:
    groups = min(int(channels), int(maximum))
    while channels % groups and groups > 1:
        groups -= 1
    return groups


class ConvGroupNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels, groups), out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class PSAXPairAwareHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(in_channels // 2, 192)
        mid = max(hidden // 2, 96)
        self.heatmap_size = HEATMAP_SIZE
        self.trunk = nn.Sequential(
            ConvGroupNormAct(in_channels, hidden, 32),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGroupNormAct(hidden, mid, 32),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGroupNormAct(mid, mid, 32),
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


class ConvExpert(nn.Module):
    def __init__(self, task_configs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.encoder = ConvNeXtFeatureBackbone()
        self.heads = nn.ModuleDict(
            {
                str(item["task_id"]): HeatmapDecoderHead(
                    self.encoder.out_channels, int(item["num_classes"])
                )
                for item in task_configs
            }
        )
        self.heads["PSAX"] = PSAXPairAwareHead(self.encoder.out_channels)


def build_beit(input_size: int) -> nn.Module:
    from timm.models.beit import Beit

    return Beit(
        img_size=int(input_size),
        patch_size=16,
        in_chans=3,
        num_classes=0,
        global_pool="",
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        mlp_ratio=4.0,
        init_values=0.1,
        drop_path_rate=0.0,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
    )


class USFMBackbone(nn.Module):
    out_channels = 768
    patch_size = 16

    def __init__(self, input_size: int = 256) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.grid_size = self.input_size // self.patch_size
        self.backbone = build_beit(self.input_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.backbone.patch_embed(image)
        value = torch.cat(
            (self.backbone.cls_token.expand(value.shape[0], -1, -1), value), dim=1
        )
        if self.backbone.pos_embed is not None:
            value = value + self.backbone.pos_embed
        value = self.backbone.pos_drop(value)
        relative_bias = self.backbone.rel_pos_bias()
        for block in self.backbone.blocks:
            value = block(value, shared_rel_pos_bias=relative_bias)
        value = self.backbone.norm(value)[:, 1:]
        return value.transpose(1, 2).reshape(
            image.shape[0], self.out_channels, self.grid_size, self.grid_size
        )


class ViTPoseHeatmapDecoder(nn.Module):
    def __init__(self, in_channels: int, channels: int = 256) -> None:
        super().__init__()
        groups = group_count(channels, 32)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, channels, 4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                channels, channels, 4, stride=2, padding=1, bias=False
            ),
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
    def __init__(self, num_points: int) -> None:
        super().__init__()
        self.decoder = ViTPoseHeatmapDecoder(768, 256)
        self.output = nn.Conv2d(256, num_points, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.decoder(value)
        return self.output(feature), feature


class FoundationExpert(nn.Module):
    def __init__(self, task_configs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.encoder = USFMBackbone(256)
        self.decoder = nn.Identity()
        self.heads = nn.ModuleDict(
            {
                str(item["task_id"]): TaskPrivateViTPoseHead(
                    int(item["num_classes"])
                )
                for item in task_configs
            }
        )


def resize64(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == (64, 64):
        return value
    return F.interpolate(value, size=(64, 64), mode="bilinear", align_corners=False)


class FrozenDualExpertEncoder(nn.Module):
    def __init__(self, task_configs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.convnext = ConvExpert(task_configs)
        self.foundation = FoundationExpert(task_configs)
        self.foundation_input_size = 256

    @staticmethod
    def _convnext_output(
        network: nn.Module, image: torch.Tensor, task_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = network.encoder(image)
        head = network.heads[str(task_id)]
        hidden = resize64(head.trunk(feature))
        if hasattr(head, "endpoint_logits"):
            logits = head.endpoint_logits(hidden)
        else:
            logits = head.heatmap_logits(hidden)
        return logits, hidden


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels, 16), out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class TaskResidualFusionHead(nn.Module):
    def __init__(self, num_points: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(128 + 2 * num_points, 64),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.GELU(),
        )
        self.output = nn.Conv2d(64, num_points, 1)

    def forward(
        self,
        shared: torch.Tensor,
        convnext_logits: torch.Tensor,
        foundation_logits: torch.Tensor,
    ) -> torch.Tensor:
        conv_evidence = convnext_logits - convnext_logits.amax(
            (-2, -1), keepdim=True
        )
        foundation_evidence = foundation_logits - foundation_logits.amax(
            (-2, -1), keepdim=True
        )
        return self.output(
            self.body(
                torch.cat((shared, conv_evidence, foundation_evidence), dim=1)
            )
        )


class DualExpertDenseFusion(nn.Module):
    def __init__(self, task_configs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.encoder = FrozenDualExpertEncoder(task_configs)
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 64, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
        )
        self.foundation_projection = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
        )
        self.shared_fusion = nn.Sequential(
            ConvGNAct(128, 128),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.GELU(),
            ConvGNAct(128, 128, kernel_size=1),
        )
        self.heads = nn.ModuleDict(
            {
                str(item["task_id"]): TaskResidualFusionHead(
                    int(item["num_classes"])
                )
                for item in task_configs
            }
        )
        self.residual_scale = 1.0


class TaskPrivateLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        tasks: list[str],
        rank: int = 4,
        alpha: float = 4.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.ModuleDict(
            {
                task: nn.Linear(base.in_features, self.rank, bias=False)
                for task in tasks
            }
        )
        self.lora_up = nn.ModuleDict(
            {
                task: nn.Linear(self.rank, base.out_features, bias=False)
                for task in tasks
            }
        )
        self.active_task: str | None = None
        self.active_strength = 0.0

    @property
    def weight(self) -> torch.Tensor:
        if self.active_task is None or self.active_strength == 0.0:
            return self.base.weight
        task = self.active_task
        delta = self.lora_up[task].weight @ self.lora_down[task].weight
        return self.base.weight + delta * (self.scale * self.active_strength)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def set_route(self, task_id: str | None, strength: float) -> None:
        self.active_task = None if task_id is None else str(task_id)
        self.active_strength = float(strength)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = group_count(channels, 8)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class ContextualTaskResidualHead(nn.Module):
    def __init__(self, num_points: int) -> None:
        super().__init__()
        hidden = 64
        input_channels = 128 + 3 * num_points + 2
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 1, bias=False),
            nn.GroupNorm(group_count(hidden, 8), hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            *[DilatedContextBlock(hidden, value) for value in (1, 2, 4, 8)]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden, num_points, 1)
        self.logit_bound = 2.0

    @staticmethod
    def _center(logits: torch.Tensor) -> torch.Tensor:
        return logits - logits.amax((-2, -1), keepdim=True)

    @staticmethod
    def _coordinates(reference: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        y = torch.linspace(
            -1.0, 1.0, height, device=reference.device, dtype=reference.dtype
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=reference.device, dtype=reference.dtype
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(
            reference.shape[0], -1, -1, -1
        )

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
        return self.logit_bound * torch.tanh(self.output(hidden))


class FinalStudentNetwork(nn.Module):
    def __init__(self, task_configs: list[dict[str, Any]]) -> None:
        super().__init__()
        self.anchor = DualExpertDenseFusion(task_configs)
        self.task_scales = dict(TASK_SCALES)
        tasks = [str(item["task_id"]) for item in task_configs]
        self._lora_locations: list[tuple[int, str]] = []
        blocks = self.anchor.encoder.foundation.encoder.backbone.blocks
        for block_index in range(len(blocks) - 4, len(blocks)):
            block = blocks[block_index]
            for name in ("qkv", "proj"):
                setattr(
                    block.attn,
                    name,
                    TaskPrivateLoRALinear(getattr(block.attn, name), tasks),
                )
                self._lora_locations.append((block_index, name))
        self.residual_heads = nn.ModuleDict(
            {
                str(item["task_id"]): ContextualTaskResidualHead(
                    int(item["num_classes"])
                )
                for item in task_configs
            }
        )

    def lora_layers(self) -> Iterator[TaskPrivateLoRALinear]:
        blocks = self.anchor.encoder.foundation.encoder.backbone.blocks
        for block_index, name in self._lora_locations:
            yield getattr(blocks[block_index].attn, name)

    @contextmanager
    def routed_foundation(self, task_id: str | None, strength: float):
        previous = [
            (layer.active_task, layer.active_strength) for layer in self.lora_layers()
        ]
        for layer in self.lora_layers():
            layer.set_route(task_id, strength)
        try:
            yield
        finally:
            for layer, (task, value) in zip(self.lora_layers(), previous):
                layer.set_route(task, value)

    def _foundation_output(
        self, image: torch.Tensor, task_id: str, strength: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        foundation_image = F.interpolate(
            image,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        route = task_id if strength != 0.0 else None
        with self.routed_foundation(route, strength):
            encoded = self.anchor.encoder.foundation.encoder(foundation_image)
            logits, feature = self.anchor.encoder.foundation.heads[task_id](encoded)
        return logits.float(), feature.float()

    def _fusion_path(
        self,
        convnext_logits: torch.Tensor,
        convnext_feature: torch.Tensor,
        foundation_logits: torch.Tensor,
        foundation_feature: torch.Tensor,
        task_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.anchor.shared_fusion(
            torch.cat(
                (
                    self.anchor.fusion(convnext_feature),
                    self.anchor.foundation_projection(foundation_feature),
                ),
                dim=1,
            )
        )
        residual = self.anchor.heads[task_id](
            shared, convnext_logits, foundation_logits
        )
        return shared, residual

    def forward(self, image: torch.Tensor, task_id: str) -> torch.Tensor:
        task = str(task_id)
        convnext_logits, convnext_feature = self.anchor.encoder._convnext_output(
            self.anchor.encoder.convnext, image, task
        )
        base_foundation_logits, base_foundation_feature = self._foundation_output(
            image, task, 0.0
        )
        _, anchor_residual = self._fusion_path(
            convnext_logits,
            convnext_feature,
            base_foundation_logits,
            base_foundation_feature,
            task,
        )
        anchor_logits = convnext_logits + self.task_scales[task] * anchor_residual
        foundation_logits, foundation_feature = self._foundation_output(
            image, task, 1.0
        )
        adapted_shared, _ = self._fusion_path(
            convnext_logits,
            convnext_feature,
            foundation_logits,
            foundation_feature,
            task,
        )
        correction = self.residual_heads[task](
            adapted_shared,
            convnext_logits,
            foundation_logits,
            anchor_logits,
        )
        return anchor_logits + correction


def build_network(task_configs: list[dict[str, Any]]) -> FinalStudentNetwork:
    expected = {str(item["task_id"]): int(item["num_classes"]) for item in task_configs}
    if expected != TASK_POINTS:
        raise RuntimeError(f"Checkpoint task map differs: {expected}")
    return FinalStudentNetwork(task_configs)


def decode_topk(logits: torch.Tensor, topk: int = 25) -> torch.Tensor:
    batch, points, height, width = logits.shape
    values, indices = torch.topk(logits.flatten(2), min(topk, height * width), dim=-1)
    weights = torch.softmax(values, dim=-1)
    x = (indices % width).to(logits.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    return torch.stack(
        (
            (weights * x).sum(-1) / float(width - 1),
            (weights * y).sum(-1) / float(height - 1),
        ),
        dim=-1,
    ).reshape(batch, points, 2)


def sort_official_vertical(points: torch.Tensor, task_id: str) -> torch.Tensor:
    if task_id not in VERTICAL_ORDER_TASKS:
        return points
    output = points.clone()
    for first in range(0, points.shape[1], 2):
        second = first + 1
        swap = output[:, first, 1] > output[:, second, 1]
        first_value = output[:, first].clone()
        second_value = output[:, second].clone()
        output[:, first] = torch.where(swap[:, None], second_value, first_value)
        output[:, second] = torch.where(swap[:, None], first_value, second_value)
    return output


def letterbox_rgb(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    original_h, original_w = image.shape[:2]
    scale = float(INPUT_SIZE) / float(max(original_h, original_w))
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    pad_x = (INPUT_SIZE - resized_w) // 2
    pad_y = (INPUT_SIZE - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "original_w": float(original_w),
        "original_h": float(original_h),
    }


def normalize_image(canvas: np.ndarray) -> torch.Tensor:
    value = canvas.astype(np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(value, (2, 0, 1)).copy())


def canvas_to_original(
    points: torch.Tensor, metadata: list[dict[str, float]]
) -> list[np.ndarray]:
    values = points.detach().cpu().float().numpy().copy()
    values[..., 0] *= float(INPUT_SIZE - 1)
    values[..., 1] *= float(INPUT_SIZE - 1)
    outputs = []
    for index, meta in enumerate(metadata):
        current = values[index]
        current[:, 0] = (current[:, 0] - meta["pad_x"]) / meta["scale"]
        current[:, 1] = (current[:, 1] - meta["pad_y"]) / meta["scale"]
        current[:, 0] = np.clip(current[:, 0], 0.0, meta["original_w"] - 1.0)
        current[:, 1] = np.clip(current[:, 1], 0.0, meta["original_h"] - 1.0)
        outputs.append(current.reshape(-1).astype(np.float64))
    return outputs


class Model:
    def __init__(self):
        cv2.setNumThreads(1)
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        checkpoint = Path.cwd() / "best_model.pth"
        if not checkpoint.is_file():
            checkpoint = Path("/app/best_model.pth")
        if not checkpoint.is_file():
            raise FileNotFoundError("best_model.pth is absent")
        manifest_path = Path("/app/model_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(checkpoint)
        if actual_hash != (manifest.get("checkpoint_sha256") or manifest.get("checkpoint", {}).get("sha256")):
            raise RuntimeError("Deployment checkpoint checksum mismatch")
        try:
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        self.network = build_network(payload["task_configs"])
        result = self.network.load_state_dict(payload["model_state"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Strict deployment checkpoint load failed")
        del payload
        gc.collect()
        requested_device = os.environ.get("FOUNDUS_DEVICE", "auto").strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError(f"Unsupported FOUNDUS_DEVICE={requested_device}")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FOUNDUS_DEVICE=cuda but CUDA is unavailable")
        use_cuda = torch.cuda.is_available() and requested_device != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.network.eval().to(self.device)
        self.amp_dtype = torch.bfloat16
        self.use_amp = self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        print(
            f"Loaded final FUB 2026 model on {self.device}; checkpoint={actual_hash}",
            flush=True,
        )

    def _read_metadata(self, data_root: Path) -> list[dict[str, str]]:
        path = data_root / "csv" / "test_metadata.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"image_path", "task_id", "num_classes"}
        if not rows or not required.issubset(rows[0]):
            raise RuntimeError("test_metadata.csv is empty or incomplete")
        seen: set[str] = set()
        for row in rows:
            task = str(row["task_id"])
            if task not in TASK_POINTS:
                raise RuntimeError(f"Unknown task_id: {task}")
            if int(row["num_classes"]) != TASK_POINTS[task]:
                raise RuntimeError(f"Point-count mismatch for {task}")
            image_path = str(row["image_path"])
            if image_path in seen:
                raise RuntimeError(f"Duplicate image_path: {image_path}")
            seen.add(image_path)
        return rows

    def _load_batch(
        self,
        data_root: Path,
        rows: list[dict[str, str]],
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        images: list[torch.Tensor] = []
        metadata: list[dict[str, float]] = []
        images_root = data_root / "images"
        for row in rows:
            relative = Path(str(row["image_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe image path: {relative}")
            task = str(row["task_id"])
            if len(relative.parts) < 2 or relative.parts[0] != task:
                raise RuntimeError(f"Image path/task mismatch: {relative} vs {task}")
            # The fixed organizer entrypoint intentionally symlinks
            # /work/images/<task> to the read-only /input/<task>_test directory.
            image_path = images_root / relative
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            canvas, meta = letterbox_rgb(image)
            images.append(normalize_image(canvas))
            metadata.append(meta)
        return torch.stack(images), metadata

    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        root = Path(data_root)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        rows = self._read_metadata(root)
        predictions: list[dict[str, Any] | None] = [None] * len(rows)
        # The validated final prediction path used homogeneous task batches of four.
        # Keeping that batch shape also leaves headroom on the final RTX 3080 10 GB.
        effective_batch = max(1, min(int(batch_size), 4 if self.use_amp else 1))
        task_indices = {
            task: [index for index, row in enumerate(rows) if row["task_id"] == task]
            for task in TASK_POINTS
        }
        completed = 0
        with torch.inference_mode():
            for task in TASK_POINTS:
                indices = task_indices[task]
                for start in range(0, len(indices), effective_batch):
                    current_indices = indices[start : start + effective_batch]
                    current_rows = [rows[index] for index in current_indices]
                    images, metadata = self._load_batch(root, current_rows)
                    images = images.to(self.device, non_blocking=True)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        logits = self.network(images, task)
                    points = sort_official_vertical(
                        decode_topk(logits.float(), 25), task
                    )
                    originals = canvas_to_original(points, metadata)
                    for local_index, row_index in enumerate(current_indices):
                        pixels = originals[local_index]
                        width = metadata[local_index]["original_w"]
                        height = metadata[local_index]["original_h"]
                        if not np.isfinite(pixels).all():
                            raise RuntimeError("Non-finite prediction")
                        predictions[row_index] = {
                            "image_path": str(rows[row_index]["image_path"]),
                            "task_id": task,
                            "predicted_points_pixels": pixels.tolist(),
                        }
                    completed += len(current_indices)
                print(f"Predicted {task}: {len(indices)}; total={completed}", flush=True)
        if any(value is None for value in predictions):
            raise RuntimeError("Missing prediction rows")
        output_path = destination / "regression_predictions.json"
        temporary = destination / ".regression_predictions.json.tmp"
        temporary.write_text(
            json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, output_path)
        print(f"Saved {len(predictions)} predictions to {output_path}", flush=True)
        if self.device.type == "cuda":
            print(
                "CUDA peak bytes: "
                f"allocated={torch.cuda.max_memory_allocated(self.device)} "
                f"reserved={torch.cuda.max_memory_reserved(self.device)}",
                flush=True,
            )
