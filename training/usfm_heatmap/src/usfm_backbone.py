from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
USFM_EXPECTED_SHA256 = "d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(value: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_beit(input_size: int, drop_path_rate: float = 0.0) -> nn.Module:
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
        drop_path_rate=float(drop_path_rate),
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
    )


def load_usfm_state(
    backbone: nn.Module,
    checkpoint_value: str | Path,
    expected_sha256: str = USFM_EXPECTED_SHA256,
) -> tuple[dict[str, Any], torch.Tensor]:
    from timm.models.beit import checkpoint_filter_fn

    checkpoint = resolve_path(checkpoint_value)
    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 and actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            f"USFM checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "mask_token" not in payload:
        raise TypeError("USFM checkpoint must be the official raw state dictionary.")
    mask_token = payload["mask_token"].detach().clone()
    source = {
        str(key): value
        for key, value in payload.items()
        if str(key) not in {"mask_token", "rel_pos_bias.relative_position_index"}
    }
    filtered = checkpoint_filter_fn(source, backbone)
    index_key = "rel_pos_bias.relative_position_index"
    filtered[index_key] = backbone.state_dict()[index_key]
    result = backbone.load_state_dict(filtered, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict USFM mapping failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return (
        {
            "source": "https://github.com/openmedlab/USFM",
            "source_commit": "960a1e1d30b9490e53e6e7cf1b4dd24425fffc67",
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": actual_sha256,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "mapping": "official shared BEiT relative-position bias; timm geometric resize",
            "mapped_tensor_count": len(filtered),
            "missing_keys": [],
            "unexpected_keys": [],
        },
        mask_token,
    )


class USFMBackbone(nn.Module):
    out_channels = 768
    patch_size = 16

    def __init__(self, settings: dict[str, Any]):
        super().__init__()
        self.input_size = int(settings.get("input_size", 256))
        if self.input_size % self.patch_size:
            raise ValueError("USFM input_size must be divisible by patch size 16.")
        self.grid_size = self.input_size // self.patch_size
        self.backbone = build_beit(
            self.input_size,
            float(settings.get("backbone_drop_path_rate", 0.0)),
        )
        self.load_info, mask_token = load_usfm_state(
            self.backbone,
            settings["usfm_checkpoint"],
            str(settings.get("usfm_checkpoint_sha256", USFM_EXPECTED_SHA256)),
        )
        self.register_buffer("pretrained_mask_token", mask_token, persistent=False)
        adapted = settings.get("adapted_backbone_checkpoint")
        if adapted:
            path = resolve_path(adapted)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("student_backbone_state", payload.get("backbone_state", payload))
            result = self.backbone.load_state_dict(state, strict=True)
            self.load_info["adapted_backbone_checkpoint"] = str(path.relative_to(PROJECT_ROOT))
            self.load_info["adapted_backbone_sha256"] = sha256_file(path)
            self.load_info["adapted_missing_keys"] = list(result.missing_keys)
            self.load_info["adapted_unexpected_keys"] = list(result.unexpected_keys)

    def forward_tokens(
        self,
        image: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.backbone.patch_embed(image)
        if patch_mask is not None:
            if patch_mask.shape != value.shape[:2]:
                raise ValueError(
                    f"Patch mask shape {tuple(patch_mask.shape)} != token grid {tuple(value.shape[:2])}"
                )
            token = self.pretrained_mask_token if mask_token is None else mask_token
            value = torch.where(patch_mask[..., None], token.to(value.dtype), value)
        value = torch.cat((self.backbone.cls_token.expand(value.shape[0], -1, -1), value), dim=1)
        if self.backbone.pos_embed is not None:
            value = value + self.backbone.pos_embed
        value = self.backbone.pos_drop(value)
        relative_bias = self.backbone.rel_pos_bias()
        for block in self.backbone.blocks:
            value = block(value, shared_rel_pos_bias=relative_bias)
        return self.backbone.norm(value)[:, 1:]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(image)
        return tokens.transpose(1, 2).reshape(
            image.shape[0], self.out_channels, self.grid_size, self.grid_size
        )

    def set_trainable_last_blocks(self, count: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        count = max(0, min(int(count), len(self.backbone.blocks)))
        if count:
            for block in self.backbone.blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

