from __future__ import annotations

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP298_MODEL = (
    PROJECT_ROOT
    / "2-code/298-task-private-usfm-lora-distillation/src/model.py"
)
EXP295_MODEL = (
    PROJECT_ROOT
    / "2-code/295-contextual-foundation-functional-compression/src/model.py"
)
EXP298_SHA256 = "44377e4efaf12d6358cd5df75e9dc4a9af125e8c58dc1fe4245fe3cd37417e90"
EXP295_SHA256 = "9a4103957b60888b403dff784e2c8c9904f63ce528341a352fbfad8800c46a5e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_locked(name: str, path: Path, expected_sha256: str):
    actual = _sha256(path)
    if actual != str(expected_sha256):
        raise RuntimeError(
            f"Locked dependency changed: {path}; expected={expected_sha256}, actual={actual}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp298 = _load_locked("exp298_model_for_exp299", EXP298_MODEL, EXP298_SHA256)
exp295 = _load_locked("exp295_model_for_exp299", EXP295_MODEL, EXP295_SHA256)

TaskPrivateLoRALinear = exp298.TaskPrivateLoRALinear
ContextualTaskResidualHead = exp295.ContextualTaskResidualHead
frozen_anchor_state_sha256 = exp298.frozen_anchor_state_sha256
_sha256_tensors = exp298._sha256_tensors


class DecoupledTaskPrivateFoundationCorrection(nn.Module):
    """Frozen Exp205 anchor plus an independently scaled task-private correction.

    The locked Exp205 task scale defines only ``anchor_logits``. The new USFM
    LoRA and correction head are added after that anchor, so tasks whose locked
    scale is zero can still learn from official challenge-unlabeled images.
    """

    def __init__(
        self,
        anchor: nn.Module,
        task_configs: list[dict[str, Any]],
        task_scales: dict[str, float],
        shared_channels: int = 128,
        hidden_channels: int = 64,
        logit_bound: float = 2.0,
        lora_last_blocks: int = 4,
        qkv_lora_rank: int = 4,
        qkv_lora_alpha: float = 4.0,
        projection_lora_rank: int = 4,
        projection_lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor = anchor
        self.task_scales = {
            str(task): float(scale) for task, scale in task_scales.items()
        }
        task_points = {
            str(item["task_id"]): int(item["num_classes"])
            for item in task_configs
        }
        if set(task_points) != set(self.task_scales):
            raise RuntimeError("Task scale map does not match anchor tasks.")

        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self._lora_locations = self._inject_task_private_lora(
            tasks=list(task_points),
            last_blocks=int(lora_last_blocks),
            qkv_rank=int(qkv_lora_rank),
            qkv_alpha=float(qkv_lora_alpha),
            projection_rank=int(projection_lora_rank),
            projection_alpha=float(projection_lora_alpha),
            dropout=float(lora_dropout),
        )
        self.residual_heads = nn.ModuleDict(
            {
                task: ContextualTaskResidualHead(
                    shared_channels=int(shared_channels),
                    num_points=points,
                    hidden_channels=int(hidden_channels),
                    logit_bound=float(logit_bound),
                )
                for task, points in task_points.items()
            }
        )
        self.anchor.eval()

    @property
    def _foundation_model(self) -> nn.Module:
        return self.anchor.encoder.foundation

    @property
    def _foundation_backbone(self) -> nn.Module:
        return self._foundation_model.encoder.backbone

    def _inject_task_private_lora(
        self,
        tasks: list[str],
        last_blocks: int,
        qkv_rank: int,
        qkv_alpha: float,
        projection_rank: int,
        projection_alpha: float,
        dropout: float,
    ) -> list[tuple[int, str]]:
        blocks = self._foundation_backbone.blocks
        count = max(1, min(int(last_blocks), len(blocks)))
        locations: list[tuple[int, str]] = []
        for block_index in range(len(blocks) - count, len(blocks)):
            block = blocks[block_index]
            for name, rank, alpha in (
                ("qkv", qkv_rank, qkv_alpha),
                ("proj", projection_rank, projection_alpha),
            ):
                base = getattr(block.attn, name)
                setattr(
                    block.attn,
                    name,
                    TaskPrivateLoRALinear(base, tasks, rank, alpha, dropout),
                )
                locations.append((block_index, name))
        return locations

    def lora_layers(self) -> Iterator[TaskPrivateLoRALinear]:
        for block_index, name in self._lora_locations:
            layer = getattr(self._foundation_backbone.blocks[block_index].attn, name)
            if not isinstance(layer, TaskPrivateLoRALinear):
                raise TypeError(f"LoRA route missing at block {block_index}.{name}.")
            yield layer

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        self.residual_heads.train(mode)
        for layer in self.lora_layers():
            layer.train(mode)
            layer.base.eval()
        return self

    def _foundation_output(
        self,
        image: torch.Tensor,
        task_id: str,
        adapter_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = int(self.anchor.encoder.foundation_input_size)
        foundation_image = F.interpolate(
            image,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        route = task_id if float(adapter_strength) != 0.0 else None
        with self.routed_foundation(route, adapter_strength):
            encoded = self._foundation_model.encoder(foundation_image)
            logits, feature = self._foundation_model.heads[str(task_id)](encoded)
        return logits.float(), feature.float()

    def _frozen_fusion_path(
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
        residual = self.anchor.heads[str(task_id)](
            shared, convnext_logits, foundation_logits
        )
        return shared, residual

    def forward(
        self,
        image: torch.Tensor,
        task_id: str,
        residual_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        task = str(task_id)
        with torch.no_grad():
            conv_logits, conv_feature = self.anchor.encoder._convnext_output(
                self.anchor.encoder.convnext, image, task
            )
            base_foundation_logits, base_foundation_feature = self._foundation_output(
                image, task, adapter_strength=0.0
            )
            anchor_shared, anchor_residual = self._frozen_fusion_path(
                conv_logits,
                conv_feature,
                base_foundation_logits,
                base_foundation_feature,
                task,
            )
            anchor_logits = (
                conv_logits + self.task_scales[task] * anchor_residual
            ).detach()

        if abs(float(residual_scale)) < 1e-12:
            correction = torch.zeros_like(anchor_logits)
            return {
                "heatmap_logits": anchor_logits,
                "anchor_heatmap_logits": anchor_logits,
                "residual_heatmap_logits": correction,
                "student_minus_anchor_logits": correction,
                "features": anchor_shared.detach(),
                "foundation_feature_delta_abs_mean": correction.new_zeros(()),
            }

        foundation_logits, foundation_feature = self._foundation_output(
            image, task, adapter_strength=1.0
        )
        adapted_shared, _ = self._frozen_fusion_path(
            conv_logits.detach(),
            conv_feature.detach(),
            foundation_logits,
            foundation_feature,
            task,
        )
        correction = self.residual_heads[task](
            adapted_shared,
            conv_logits.detach(),
            foundation_logits,
            anchor_logits,
        )
        scaled = float(residual_scale) * correction
        return {
            "heatmap_logits": anchor_logits + scaled,
            "anchor_heatmap_logits": anchor_logits,
            "residual_heatmap_logits": correction,
            "student_minus_anchor_logits": scaled,
            "features": adapted_shared,
            "foundation_feature_delta_abs_mean": (
                foundation_feature - base_foundation_feature.detach()
            ).abs().mean(),
        }

    def lora_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.anchor.named_parameters():
            if ".lora_down." in name or ".lora_up." in name:
                yield name, parameter

    def lora_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.lora_named_parameters()]

    def correction_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.residual_heads.parameters()
            if parameter.requires_grad
        ]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.correction_parameters() + self.lora_parameters()

    def lora_up_regularization(self) -> torch.Tensor:
        values = [
            module.weight.square().mean()
            for layer in self.lora_layers()
            for module in layer.lora_up.values()
        ]
        return torch.stack(values).mean()

    def lora_state_sha256(self, task_id: str | None = None) -> str:
        marker = None if task_id is None else f".{str(task_id)}."

        def values():
            for name, parameter in self.lora_named_parameters():
                if marker is None or marker in name:
                    yield name, parameter

        return _sha256_tensors(values())

    def correction_state_sha256(self, task_id: str) -> str:
        return _sha256_tensors(
            (name, value)
            for name, value in self.residual_heads[str(task_id)].state_dict().items()
        )

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "task_private_lora": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.lora_named_parameters()
            },
            "task_private_correction": {
                name: value.detach().cpu().clone()
                for name, value in self.residual_heads.state_dict().items()
            },
        }


# The shared Stage-0 harness expects this class name.
FunctionalCompressionStudent = DecoupledTaskPrivateFoundationCorrection
