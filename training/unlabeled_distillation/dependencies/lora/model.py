from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskPrivateLoRALinear(nn.Module):
    """A frozen linear layer with one independently routed LoRA per task."""

    def __init__(
        self,
        base: nn.Linear,
        tasks: list[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("TaskPrivateLoRALinear requires nn.Linear.")
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive.")
        if float(dropout) != 0.0:
            raise ValueError(
                "Exp298 requires LoRA dropout=0 because timm BEiT reads qkv.weight directly."
            )
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.ModuleDict()
        self.lora_up = nn.ModuleDict()
        for task in tasks:
            task_id = str(task)
            down = nn.Linear(base.in_features, self.rank, bias=False)
            up = nn.Linear(self.rank, base.out_features, bias=False)
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5.0))
            nn.init.zeros_(up.weight)
            self.lora_down[task_id] = down
            self.lora_up[task_id] = up
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
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
        if task_id is not None and str(task_id) not in self.lora_down:
            raise KeyError(f"Unknown LoRA task route: {task_id}")
        self.active_task = None if task_id is None else str(task_id)
        self.active_strength = float(strength)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # BEiT qkv accesses ``weight`` directly, while attention projection
        # calls the module. Using the same synthesized tensor covers both.
        return F.linear(value, self.weight, self.bias)


class TaskScaleDeltas(nn.Module):
    def __init__(self, tasks: list[str], maximum_delta: float) -> None:
        super().__init__()
        self.values = nn.ParameterDict(
            {str(task): nn.Parameter(torch.zeros(())) for task in tasks}
        )
        self.maximum_delta = float(maximum_delta)

    def delta(self, task_id: str) -> torch.Tensor:
        return self.maximum_delta * torch.tanh(self.values[str(task_id)])

    def regularization(self) -> torch.Tensor:
        return torch.stack([value.square() for value in self.values.values()]).mean()


def _sha256_tensors(items: Iterator[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_anchor_state_sha256(anchor: nn.Module) -> str:
    def values():
        for name, value in anchor.state_dict().items():
            if ".lora_down." in name or ".lora_up." in name:
                continue
            yield name, value

    return _sha256_tensors(values())


class TaskPrivateUSFMLoRAStudent(nn.Module):
    """One Exp191 graph with task-private internal USFM LoRA and copied fusion."""

    def __init__(
        self,
        anchor: nn.Module,
        task_configs: list[dict[str, Any]],
        task_scales: dict[str, float],
        shared_channels: int = 128,
        hidden_channels: int = 64,
        logit_bound: float = 2.0,
        maximum_scale_delta: float = 0.5,
        lora_last_blocks: int = 4,
        qkv_lora_rank: int = 4,
        qkv_lora_alpha: float = 4.0,
        projection_lora_rank: int = 4,
        projection_lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del shared_channels, hidden_channels, logit_bound
        self.anchor = anchor
        self.task_scales = {
            str(task): float(scale) for task, scale in task_scales.items()
        }
        tasks = [str(item["task_id"]) for item in task_configs]
        if set(tasks) != set(self.task_scales):
            raise RuntimeError("Task scale map does not match anchor tasks.")
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self._lora_locations = self._inject_task_private_lora(
            tasks=tasks,
            last_blocks=int(lora_last_blocks),
            qkv_rank=int(qkv_lora_rank),
            qkv_alpha=float(qkv_lora_alpha),
            projection_rank=int(projection_lora_rank),
            projection_alpha=float(projection_lora_alpha),
            dropout=float(lora_dropout),
        )
        self.residual_heads = nn.ModuleDict(
            {
                "fusion": copy.deepcopy(anchor.fusion),
                "foundation_projection": copy.deepcopy(anchor.foundation_projection),
                "shared_fusion": copy.deepcopy(anchor.shared_fusion),
                "heads": copy.deepcopy(anchor.heads),
                "scale_deltas": TaskScaleDeltas(tasks, maximum_scale_delta),
            }
        )
        for parameter in self.residual_heads.parameters():
            parameter.requires_grad_(True)
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
                raise TypeError(f"LoRA layer was replaced at block {block_index}.{name}.")
            yield layer

    @contextmanager
    def routed_foundation(self, task_id: str | None, strength: float):
        previous = [(layer.active_task, layer.active_strength) for layer in self.lora_layers()]
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

    def _path(
        self,
        convnext_logits: torch.Tensor,
        convnext_feature: torch.Tensor,
        foundation_logits: torch.Tensor,
        foundation_feature: torch.Tensor,
        task_id: str,
        trainable: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.residual_heads if trainable else self.anchor
        shared = source.shared_fusion(
            torch.cat(
                (
                    source.fusion(convnext_feature),
                    source.foundation_projection(foundation_feature),
                ),
                dim=1,
            )
        )
        residual = source.heads[str(task_id)](
            shared, convnext_logits, foundation_logits
        )
        return shared, residual

    def effective_scale(self, task_id: str) -> torch.Tensor:
        values = self.residual_heads["scale_deltas"].values[str(task_id)]
        return values.new_tensor(self.task_scales[str(task_id)]) + self.residual_heads[
            "scale_deltas"
        ].delta(str(task_id))

    def effective_scales(self) -> dict[str, float]:
        return {
            task: float(self.effective_scale(task).detach().cpu())
            for task in sorted(self.task_scales)
        }

    def scale_regularization(self) -> torch.Tensor:
        return self.residual_heads["scale_deltas"].regularization()

    def lora_up_regularization(self) -> torch.Tensor:
        values = [
            module.weight.square().mean()
            for layer in self.lora_layers()
            for module in layer.lora_up.values()
        ]
        return torch.stack(values).mean()

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
            anchor_shared, anchor_residual = self._path(
                conv_logits,
                conv_feature,
                base_foundation_logits,
                base_foundation_feature,
                task,
                trainable=False,
            )
            anchor_logits = conv_logits + self.task_scales[task] * anchor_residual

        foundation_logits, foundation_feature = self._foundation_output(
            image, task, adapter_strength=1.0
        )
        student_shared, student_residual = self._path(
            conv_logits.detach(),
            conv_feature.detach(),
            foundation_logits,
            foundation_feature,
            task,
            trainable=True,
        )
        student_logits = conv_logits.detach() + self.effective_scale(task) * student_residual
        difference = student_logits - anchor_logits.detach()
        output_logits = anchor_logits.detach() + float(residual_scale) * difference
        return {
            "heatmap_logits": output_logits,
            "anchor_heatmap_logits": anchor_logits.detach(),
            "residual_heatmap_logits": student_residual,
            "student_minus_anchor_logits": difference,
            "features": student_shared,
            "anchor_features": anchor_shared.detach(),
            "foundation_feature_delta_abs_mean": (
                foundation_feature - base_foundation_feature.detach()
            ).abs().mean(),
            "effective_scale": self.effective_scale(task),
        }

    def lora_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.anchor.named_parameters():
            if ".lora_down." in name or ".lora_up." in name:
                yield name, parameter

    def lora_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.lora_named_parameters()]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.lora_parameters() + [
            parameter
            for parameter in self.residual_heads.parameters()
            if parameter.requires_grad
        ]

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.lora_named_parameters()
        }

    def lora_state_sha256(self, task_id: str | None = None) -> str:
        marker = None if task_id is None else f".{str(task_id)}."

        def values():
            for name, parameter in self.lora_named_parameters():
                if marker is None or marker in name:
                    yield name, parameter

        return _sha256_tensors(values())

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "task_private_lora": self.lora_state_dict(),
            "fusion": {
                key: value.detach().cpu().clone()
                for key, value in self.residual_heads.state_dict().items()
            },
        }
