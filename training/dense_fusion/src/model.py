from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPERVISED_CONVNEXT_MODEL = PROJECT_ROOT / "training/supervised_convnext/src/model.py"
USFM_HEATMAP_SRC = PROJECT_ROOT / "training/usfm_heatmap/src"
MODEL_NAME = "DenseFusion-USFM-ConvNeXt-Dense-Fusion"
if str(USFM_HEATMAP_SRC) not in sys.path:
    sys.path.insert(0, str(USFM_HEATMAP_SRC))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


supervised_convnext = _load_module("supervised_convnext_model_for_dense_fusion", SUPERVISED_CONVNEXT_MODEL)
usfm_heatmap = _load_module("usfm_heatmap_model_for_dense_fusion", USFM_HEATMAP_SRC / "model.py")

canonicalize_training_target = supervised_convnext.canonicalize_training_target
canonicalize_internal_points = supervised_convnext.canonicalize_internal_points
sort_official_vertical = supervised_convnext.sort_official_vertical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _task_map(task_configs: list[dict[str, Any]]) -> dict[str, int]:
    return {str(item["task_id"]): int(item["num_classes"]) for item in task_configs}


def _load_expert(
    checkpoint_value: str,
    expected_sha256: str,
    module: Any,
    current_tasks: list[dict[str, Any]],
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = _resolve(checkpoint_value)
    actual_sha256 = _sha256(checkpoint)
    if expected_sha256 and actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"Expert checksum mismatch for {checkpoint}: {actual_sha256}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if _task_map(payload["task_configs"]) != _task_map(current_tasks):
        raise RuntimeError(f"Expert task map differs for {checkpoint}.")
    network = module.build_model(payload["settings"], payload["task_configs"])
    result = network.load_state_dict(payload["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict expert load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network, {
        "path": str(checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": actual_sha256,
        "epoch": int(payload.get("epoch", -1)),
        "settings": payload["settings"],
    }


def _resize64(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == (64, 64):
        return value
    return F.interpolate(value, size=(64, 64), mode="bilinear", align_corners=False)


class FrozenDualExpertEncoder(nn.Module):
    def __init__(self, settings: dict[str, Any], task_configs: list[dict[str, Any]]):
        super().__init__()
        self.convnext, self.convnext_source = _load_expert(
            settings["convnext_checkpoint"],
            settings["convnext_checkpoint_sha256"],
            supervised_convnext,
            task_configs,
        )
        self.foundation, self.foundation_source = _load_expert(
            settings["foundation_checkpoint"],
            settings["foundation_checkpoint_sha256"],
            usfm_heatmap,
            task_configs,
        )
        self.foundation_input_size = int(settings.get("foundation_input_size", 256))
        self.load_info = {
            "convnext": self.convnext_source,
            "foundation": self.foundation_source,
            "spatial_alignment": (
                "SupervisedConvNeXt 518x518 canvas; bilinear antialiased resize to "
                f"{self.foundation_input_size}x{self.foundation_input_size} for USFM"
            ),
        }

    def train(self, mode: bool = True):
        super().train(False)
        self.convnext.eval()
        self.foundation.eval()
        return self

    @staticmethod
    def _convnext_output(
        network: nn.Module, image: torch.Tensor, task_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = network.encoder(image)
        head = network.heads[str(task_id)]
        hidden = _resize64(head.trunk(feature))
        if hasattr(head, "endpoint_logits"):
            logits = head.endpoint_logits(hidden)
        else:
            logits = head.heatmap_logits(hidden)
        return logits, hidden

    def forward(
        self, image: torch.Tensor, task_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            conv_logits, conv_feature = self._convnext_output(
                self.convnext, image, task_id
            )
            foundation_image = F.interpolate(
                image,
                size=(self.foundation_input_size, self.foundation_input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            encoded = self.foundation.encoder(foundation_image)
            foundation_logits, foundation_feature = self.foundation.heads[str(task_id)](
                encoded
            )
        return (
            conv_logits.float(),
            conv_feature.float(),
            foundation_logits.float(),
            foundation_feature.float(),
        )


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        groups = min(16, int(out_channels))
        while out_channels % groups and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class TaskResidualFusionHead(nn.Module):
    def __init__(self, shared_channels: int, num_points: int, hidden_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(shared_channels + 2 * int(num_points), hidden_channels),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels, int(num_points), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        shared: torch.Tensor,
        convnext_logits: torch.Tensor,
        foundation_logits: torch.Tensor,
    ) -> torch.Tensor:
        conv_evidence = convnext_logits - convnext_logits.amax((-2, -1), keepdim=True)
        foundation_evidence = foundation_logits - foundation_logits.amax(
            (-2, -1), keepdim=True
        )
        return self.output(
            self.body(torch.cat((shared, conv_evidence, foundation_evidence), dim=1))
        )


class DualExpertDenseFusion(nn.Module):
    def __init__(self, settings: dict[str, Any], task_configs: list[dict[str, Any]]):
        super().__init__()
        self.encoder = FrozenDualExpertEncoder(settings, task_configs)
        projection_channels = int(settings.get("fusion_projection_channels", 64))
        shared_channels = int(settings.get("fusion_shared_channels", 128))
        head_channels = int(settings.get("fusion_head_channels", 64))
        self.fusion = nn.Sequential(
            nn.Conv2d(192, projection_channels, 1, bias=False),
            nn.GroupNorm(16, projection_channels),
            nn.GELU(),
        )
        self.foundation_projection = nn.Sequential(
            nn.Conv2d(256, projection_channels, 1, bias=False),
            nn.GroupNorm(16, projection_channels),
            nn.GELU(),
        )
        self.shared_fusion = nn.Sequential(
            ConvGNAct(2 * projection_channels, shared_channels),
            nn.Conv2d(
                shared_channels,
                shared_channels,
                3,
                padding=1,
                groups=shared_channels,
                bias=False,
            ),
            nn.GELU(),
            ConvGNAct(shared_channels, shared_channels, kernel_size=1),
        )
        self.heads = nn.ModuleDict(
            {
                str(item["task_id"]): TaskResidualFusionHead(
                    shared_channels,
                    int(item["num_classes"]),
                    head_channels,
                )
                for item in task_configs
            }
        )
        self.residual_scale = float(settings.get("fusion_residual_scale", 1.0))

    def forward(self, image: torch.Tensor, task_id: str) -> dict[str, torch.Tensor]:
        conv_logits, conv_feature, foundation_logits, foundation_feature = self.encoder(
            image, str(task_id)
        )
        shared = self.shared_fusion(
            torch.cat(
                (self.fusion(conv_feature), self.foundation_projection(foundation_feature)),
                dim=1,
            )
        )
        residual = self.heads[str(task_id)](
            shared, conv_logits, foundation_logits
        )
        return {
            "heatmap_logits": conv_logits + self.residual_scale * residual,
            "base_heatmap_logits": conv_logits,
            "foundation_heatmap_logits": foundation_logits,
            "residual_heatmap_logits": residual,
            "features": shared,
        }


def build_model(settings: dict[str, Any], task_configs: list[dict[str, Any]]) -> nn.Module:
    if int(settings.get("input_size", 518)) != 518:
        raise ValueError("DenseFusion uses the SupervisedConvNeXt 518x518 canvas as its output frame.")
    if int(settings.get("heatmap_size", 64)) != 64:
        raise ValueError("DenseFusion requires 64x64 heatmaps.")
    return DualExpertDenseFusion(settings, task_configs)


def forward_train(
    network: DualExpertDenseFusion,
    images: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    output = network(images, task_id)
    output["coords_norm"] = usfm_heatmap._soft_argmax(
        output["heatmap_logits"], float(settings.get("train_softargmax_beta", 4.0))
    )
    return output


def forward_inference(
    network: DualExpertDenseFusion,
    images: torch.Tensor,
    task_id: str,
    phase_or_settings: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if settings is None:
        settings = phase_or_settings
    output = network(images, task_id)
    internal = usfm_heatmap.decode_topk(
        output["heatmap_logits"],
        int(settings.get("decode_topk", 25)),
        float(settings.get("decode_topk_beta", 1.0)),
    )
    output["internal_coords_norm"] = internal
    output["coords_norm"] = sort_official_vertical(internal, task_id, settings)
    return output


def compute_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    task_id: str,
    phase: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, parts = usfm_heatmap.compute_loss(output, target, task_id, phase, settings)
    residual = output["residual_heatmap_logits"].square().mean()
    total = total + float(settings.get("fusion_residual_l2_weight", 1e-5)) * residual
    return total, {**parts, "total_loss": total, "fusion_residual_l2": residual}


def configure_trainable(
    network: DualExpertDenseFusion, scope: str
) -> dict[str, int]:
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    if str(scope) != "fusion_only":
        raise ValueError(f"DenseFusion first screen supports fusion_only, got {scope}.")
    for module in (
        network.fusion,
        network.foundation_projection,
        network.shared_fusion,
        network.heads,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return {
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in network.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in network.parameters()
            if not parameter.requires_grad
        ),
        "trainable_backbone_blocks": 0,
    }


def optimizer_groups(
    network: DualExpertDenseFusion, phase: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "name": "head",
            "params": [
                parameter for parameter in network.parameters() if parameter.requires_grad
            ],
            "lr": float(phase["head_lr"]),
            "weight_decay": float(phase.get("weight_decay", 0.01)),
        }
    ]


def set_train_mode(network: DualExpertDenseFusion, scope: str) -> None:
    del scope
    network.train()
    network.encoder.eval()


def trainable_parameter_names(network: DualExpertDenseFusion) -> list[str]:
    return [name for name, parameter in network.named_parameters() if parameter.requires_grad]


def trainable_parameters(network: DualExpertDenseFusion) -> Iterable[nn.Parameter]:
    return (parameter for parameter in network.parameters() if parameter.requires_grad)
