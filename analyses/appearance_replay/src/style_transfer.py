from __future__ import annotations

from typing import Any

import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def masked_channel_stats(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4 or valid_mask.ndim != 4:
        raise ValueError("Expected image and mask tensors shaped [B,C,H,W].")
    if valid_mask.shape[1] != 1 or value.shape[0] != valid_mask.shape[0]:
        raise ValueError("Mask must be shaped [B,1,H,W] and match image batch size.")
    mask = valid_mask.to(dtype=value.dtype).clamp(0.0, 1.0)
    denominator = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / denominator
    variance = ((value - mean).square() * mask).sum(
        dim=(-2, -1), keepdim=True
    ) / denominator
    return mean, variance.sqrt().clamp_min(1e-4)


def reduce_donor_stats(
    donor_mean: torch.Tensor,
    donor_std: torch.Tensor,
    target_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    donor_count = int(donor_mean.shape[0])
    target_batch_size = int(target_batch_size)
    if donor_count <= 0 or target_batch_size <= 0:
        raise ValueError("Donor and target batches must be non-empty.")
    if donor_count < target_batch_size:
        repeats = (target_batch_size + donor_count - 1) // donor_count
        indices = torch.arange(donor_count, device=donor_mean.device).repeat(repeats)
        indices = indices[:target_batch_size]
        groups = [[int(value)] for value in indices.cpu().tolist()]
        return donor_mean[indices], donor_std[indices], groups
    raw_groups = torch.tensor_split(
        torch.arange(donor_count, device=donor_mean.device), target_batch_size
    )
    reduced_mean = torch.cat(
        [donor_mean[group].mean(dim=0, keepdim=True) for group in raw_groups], dim=0
    )
    reduced_std = torch.cat(
        [donor_std[group].mean(dim=0, keepdim=True) for group in raw_groups], dim=0
    )
    groups = [[int(value) for value in group.cpu().tolist()] for group in raw_groups]
    return reduced_mean, reduced_std, groups


def bounded_masked_moment_transfer(
    content: torch.Tensor,
    donor: torch.Tensor,
    content_mask: torch.Tensor,
    donor_mask: torch.Tensor,
    alpha: float,
    mean_shift_limit: float = 0.15,
    std_ratio_min: float = 0.60,
    std_ratio_max: float = 1.60,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transfer donor appearance moments without changing content geometry."""

    mean = content.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = content.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    content_rgb = (content * std + mean).clamp(0.0, 1.0)
    donor_rgb = (donor * std + mean).clamp(0.0, 1.0)
    content_mean, content_std = masked_channel_stats(content_rgb, content_mask)
    donor_mean, donor_std = masked_channel_stats(donor_rgb, donor_mask)
    donor_mean, donor_std, groups = reduce_donor_stats(
        donor_mean, donor_std, content.shape[0]
    )

    amount = float(alpha)
    target_mean = content_mean.lerp(donor_mean, amount)
    target_mean = content_mean + (target_mean - content_mean).clamp(
        -float(mean_shift_limit), float(mean_shift_limit)
    )
    donor_ratio = donor_std / content_std.clamp_min(1e-4)
    target_ratio = torch.ones_like(donor_ratio).lerp(donor_ratio, amount).clamp(
        float(std_ratio_min), float(std_ratio_max)
    )
    target_std = content_std * target_ratio
    styled_rgb = (content_rgb - content_mean) / content_std * target_std + target_mean
    mask = content_mask.to(dtype=content.dtype).clamp(0.0, 1.0)
    styled_normalized = (styled_rgb.clamp(0.0, 1.0) - mean) / std
    styled = torch.where(mask > 0.5, styled_normalized, content)
    used = sorted({value for group in groups for value in group})
    return styled, {
        "donor_count": int(donor.shape[0]),
        "donor_groups": groups,
        "all_donors_contributed": used == list(range(int(donor.shape[0]))),
        "mean_abs_shift": float((target_mean - content_mean).abs().mean().detach().cpu()),
        "mean_std_ratio": float(target_ratio.mean().detach().cpu()),
    }
