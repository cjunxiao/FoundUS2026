import argparse
import csv
import importlib
import json
import random
import shutil
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recursive_update(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = recursive_update(out[key], value)
        else:
            out[key] = value
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def load_canonical_train_dataframe(csv_dir: str | Path) -> pd.DataFrame:
    csv_dir = resolve_project_path(csv_dir)
    csv_files = sorted(csv_dir.glob("*_train.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No canonical train CSV files found in {csv_dir}")
    frames = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        frame["canonical_csv"] = str(csv_path.relative_to(PROJECT_ROOT))
        frames.append(frame)
    dataframe = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    required = {"image_path", "source_image_path", "task_id", "num_classes"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Canonical train CSVs missing required columns: {missing}")
    return dataframe


def filter_tasks(dataframe: pd.DataFrame, task_ids: list[str] | None) -> pd.DataFrame:
    if not task_ids:
        return dataframe.reset_index(drop=True)
    wanted = {str(v) for v in task_ids}
    filtered = dataframe[dataframe["task_id"].astype(str).isin(wanted)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No rows matched task filter: {sorted(wanted)}")
    return filtered


def build_task_configs(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for task_id, group in dataframe.groupby("task_id", sort=True):
        num_classes = sorted({int(v) for v in group["num_classes"].tolist()})
        if len(num_classes) != 1:
            raise ValueError(f"Task {task_id} has inconsistent num_classes: {num_classes}")
        configs.append({"task_id": str(task_id), "task_name": "Regression", "num_classes": int(num_classes[0])})
    if not configs:
        raise ValueError("No task configs could be built.")
    return configs


def stratified_split_indices(dataframe: pd.DataFrame, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for _, group in dataframe.groupby("task_id", sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)
        total = len(indices)
        val_count = int(round(total * float(val_split)))
        if total >= 2:
            val_count = max(1, min(total - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def parse_point(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        point = json.loads(value)
    elif isinstance(value, (list, tuple)):
        point = value
    else:
        raise ValueError(f"Unsupported point value: {value!r}")
    if len(point) != 2:
        raise ValueError(f"Expected point length 2, got {point!r}")
    return float(point[0]), float(point[1])


def read_record_points(record: pd.Series) -> np.ndarray:
    num_points = int(record["num_classes"])
    coords: list[float] = []
    for point_idx in range(1, num_points + 1):
        column = f"point_{point_idx}_xy"
        if column not in record or pd.isna(record[column]):
            raise ValueError(f"Missing {column} for image {record.get('image_path', '<unknown>')}")
        x, y = parse_point(record[column])
        coords.extend([x, y])
    return np.asarray(coords, dtype=np.float32)


def letterbox_rgb(image: np.ndarray, input_size: int, pad_value: int = 0) -> tuple[np.ndarray, dict[str, float]]:
    original_h, original_w = image.shape[:2]
    scale = float(input_size) / float(max(original_h, original_w))
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), pad_value, dtype=np.uint8)
    pad_x = int((input_size - resized_w) // 2)
    pad_y = int((input_size - resized_h) // 2)
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "resized_w": float(resized_w),
        "resized_h": float(resized_h),
        "input_size": float(input_size),
        "original_w": float(original_w),
        "original_h": float(original_h),
    }


def points_original_to_canvas(points_original: np.ndarray, meta: dict[str, float]) -> np.ndarray:
    points = points_original.astype(np.float32).copy().reshape(-1, 2)
    points[:, 0] = points[:, 0] * float(meta["scale"]) + float(meta["pad_x"])
    points[:, 1] = points[:, 1] * float(meta["scale"]) + float(meta["pad_y"])
    input_size = float(meta["input_size"])
    points[:, 0] = np.clip(points[:, 0], 0.0, input_size - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, input_size - 1.0)
    return points.reshape(-1)


def canvas_norm_to_original_pixels(points_canvas_norm: torch.Tensor, metas: list[dict[str, float]], input_size: int) -> torch.Tensor:
    points = points_canvas_norm.reshape(points_canvas_norm.shape[0], -1, 2).clone()
    points[:, :, 0] *= float(input_size - 1)
    points[:, :, 1] *= float(input_size - 1)
    out = []
    for batch_idx, meta in enumerate(metas):
        current = points[batch_idx]
        current[:, 0] = (current[:, 0] - float(meta["pad_x"])) / max(float(meta["scale"]), 1e-8)
        current[:, 1] = (current[:, 1] - float(meta["pad_y"])) / max(float(meta["scale"]), 1e-8)
        current[:, 0] = current[:, 0].clamp(0.0, float(meta["original_w"]) - 1.0)
        current[:, 1] = current[:, 1].clamp(0.0, float(meta["original_h"]) - 1.0)
        out.append(current.reshape(-1))
    return torch.stack(out, dim=0)


def normalize_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    arr = image.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr).float()


def apply_light_intensity_aug(image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-10.0, 10.0))
    out = image.astype(np.float32) * alpha + beta
    if rng.rand() < 0.15:
        out = out + rng.normal(0.0, 3.0, size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_gaussian_heatmaps(centers_hm: np.ndarray, heatmap_size: int, sigma: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing="ij")
    heatmaps = np.zeros((len(centers_hm), heatmap_size, heatmap_size), dtype=np.float32)
    for point_idx, (x, y) in enumerate(centers_hm):
        dist2 = (xx - float(x)) ** 2 + (yy - float(y)) ** 2
        heatmaps[point_idx] = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    return heatmaps


def square_bbox_from_points_norm(points_norm_flat: np.ndarray, expand: float, min_side: float = 0.05) -> np.ndarray:
    points = points_norm_flat.reshape(-1, 2).astype(np.float32)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = max(float(x_max - x_min), min_side)
    height = max(float(y_max - y_min), min_side)
    cx = float((x_min + x_max) * 0.5)
    cy = float((y_min + y_max) * 0.5)
    side = max(width, height) * float(expand)
    side = max(side, min_side)
    side = min(side, 1.0)
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    return np.asarray([cx, cy, side, side], dtype=np.float32)


class RaceTrainDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_size: int,
        heatmap_size: int,
        heatmap_sigma: float,
        bbox_expand: float,
        augment: bool,
        seed: int,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.bbox_expand = float(bbox_expand)
        self.augment = bool(augment)
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.dataframe.iloc[idx]
        image_path = resolve_project_path(record["source_image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w = image.shape[:2]
        canvas, meta = letterbox_rgb(image, self.input_size)
        if self.augment:
            canvas = apply_light_intensity_aug(canvas, self.rng)
        points_original = read_record_points(record)
        points_original[0::2] = np.clip(points_original[0::2], 0.0, max(float(image_w) - 1.0, 0.0))
        points_original[1::2] = np.clip(points_original[1::2], 0.0, max(float(image_h) - 1.0, 0.0))
        points_canvas = points_original_to_canvas(points_original, meta)
        points_canvas_norm = points_canvas / float(self.input_size - 1)
        centers_hm = points_canvas_norm.reshape(-1, 2).copy()
        centers_hm[:, 0] *= float(self.heatmap_size - 1)
        centers_hm[:, 1] *= float(self.heatmap_size - 1)
        heatmaps = generate_gaussian_heatmaps(centers_hm, self.heatmap_size, self.heatmap_sigma)
        center = points_canvas_norm.reshape(-1, 2).mean(axis=0, keepdims=True)
        center_hm = center.copy()
        center_hm[:, 0] *= float(self.heatmap_size - 1)
        center_hm[:, 1] *= float(self.heatmap_size - 1)
        center_heatmap = generate_gaussian_heatmaps(center_hm, self.heatmap_size, self.heatmap_sigma)
        bbox_norm = square_bbox_from_points_norm(points_canvas_norm, self.bbox_expand)
        return {
            "image": normalize_image_to_tensor(canvas),
            "heatmap": torch.from_numpy(heatmaps).float(),
            "center_heatmap": torch.from_numpy(center_heatmap).float(),
            "centers_hm": torch.from_numpy(centers_hm.reshape(-1)).float(),
            "points_canvas_norm": torch.from_numpy(points_canvas_norm).float(),
            "points_original": torch.from_numpy(points_original).float(),
            "bbox_norm": torch.from_numpy(bbox_norm).float(),
            "task_id": str(record["task_id"]),
            "task_name": str(record.get("task_name", "Regression")),
            "image_path": str(record["image_path"]),
            "source_image_path": str(record["source_image_path"]),
            "num_classes": int(record["num_classes"]),
            "letterbox": meta,
        }


class TaskUniformBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: RaceTrainDataset, batch_size: int, steps_per_epoch: int | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.indices_by_task: dict[str, list[int]] = {}
        for idx, task_id in enumerate(dataset.dataframe["task_id"].astype(str).tolist()):
            self.indices_by_task.setdefault(task_id, []).append(idx)
        self.task_ids = sorted(self.indices_by_task)
        if not self.task_ids:
            raise ValueError("TaskUniformBatchSampler received an empty dataset.")
        for indices in self.indices_by_task.values():
            random.shuffle(indices)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else max(1, len(dataset) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        cursors = {task_id: 0 for task_id in self.task_ids}
        for _ in range(self.steps_per_epoch):
            task_id = random.choice(self.task_ids)
            indices = self.indices_by_task[task_id]
            start = cursors[task_id]
            end = start + self.batch_size
            if end > len(indices):
                batch = indices[start:]
                random.shuffle(indices)
                needed = self.batch_size - len(batch)
                batch.extend(indices[:needed])
                cursors[task_id] = needed
            else:
                batch = indices[start:end]
                cursors[task_id] = end
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch


def race_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "task_id": [item["task_id"] for item in batch],
        "task_name": [item["task_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "source_image_path": [item["source_image_path"] for item in batch],
        "num_classes": [item["num_classes"] for item in batch],
        "letterbox": [item["letterbox"] for item in batch],
    }
    for key in ["heatmap", "center_heatmap", "centers_hm", "points_canvas_norm", "points_original", "bbox_norm"]:
        output[key] = [item[key] for item in batch]
    return output


def tensor_stack(items: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack(items, dim=0).to(device, non_blocking=True)


def batch_target(batch: dict[str, Any], indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "heatmap": tensor_stack([batch["heatmap"][idx] for idx in indices], device),
        "center_heatmap": tensor_stack([batch["center_heatmap"][idx] for idx in indices], device),
        "centers_hm": tensor_stack([batch["centers_hm"][idx] for idx in indices], device),
        "points_canvas_norm": tensor_stack([batch["points_canvas_norm"][idx] for idx in indices], device),
        "points_original": tensor_stack([batch["points_original"][idx] for idx in indices], device),
        "bbox_norm": tensor_stack([batch["bbox_norm"][idx] for idx in indices], device),
    }


class DINOv2FeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, img_size: int):
        super().__init__()
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=int(img_size))
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
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


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HeatmapDecoderHead(nn.Module):
    def __init__(self, in_channels: int, num_points: int, heatmap_size: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = int(hidden_channels or max(in_channels // 2, 192))
        mid = max(hidden // 2, 96)
        self.heatmap_size = int(heatmap_size)
        self.trunk = nn.Sequential(
            ConvNormAct(in_channels, hidden),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvNormAct(hidden, mid),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvNormAct(mid, mid),
        )
        self.heatmap_logits = nn.Conv2d(mid, int(num_points), kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.trunk(features)
        if x.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            x = F.interpolate(x, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
        return self.heatmap_logits(x)


def decode_argmax_norm(heatmap_logits: torch.Tensor) -> torch.Tensor:
    batch, num_points, height, width = heatmap_logits.shape
    flat_idx = heatmap_logits.reshape(batch, num_points, -1).argmax(dim=-1)
    y = torch.div(flat_idx, width, rounding_mode="floor").float()
    x = (flat_idx % width).float()
    x = x / max(float(width - 1), 1.0)
    y = y / max(float(height - 1), 1.0)
    return torch.stack([x, y], dim=-1).reshape(batch, -1)


def soft_argmax_2d(logits: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    batch, num_points, height, width = logits.shape
    prob = F.softmax(logits.reshape(batch, num_points, -1) / float(temperature), dim=-1)
    prob = prob.reshape(batch, num_points, height, width)
    ys = torch.linspace(0, 1, height, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0, 1, width, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (prob * xx[None, None]).sum(dim=(-2, -1))
    y = (prob * yy[None, None]).sum(dim=(-2, -1))
    return torch.stack([x, y], dim=-1).reshape(batch, -1)


def make_gaussian_heatmaps_torch(centers_norm: torch.Tensor, heatmap_size: int, sigma: float) -> torch.Tensor:
    batch, num_points, _ = centers_norm.shape
    device = centers_norm.device
    dtype = centers_norm.dtype
    ys = torch.arange(heatmap_size, device=device, dtype=dtype)
    xs = torch.arange(heatmap_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = centers_norm[:, :, 0].clamp(0.0, 1.0) * float(heatmap_size - 1)
    y = centers_norm[:, :, 1].clamp(0.0, 1.0) * float(heatmap_size - 1)
    dist2 = (xx[None, None] - x[:, :, None, None]) ** 2 + (yy[None, None] - y[:, :, None, None]) ** 2
    return torch.exp(-dist2 / (2.0 * float(sigma) * float(sigma)))


def bbox_points_to_norm(points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    pts = points.reshape(points.shape[0], -1, 2)
    center = bbox[:, None, 0:2]
    size = bbox[:, None, 2:4].clamp_min(1e-4)
    top_left = center - 0.5 * size
    return ((pts - top_left) / size).clamp(0.0, 1.0)


def bbox_norm_to_points(local_points: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    pts = local_points.reshape(local_points.shape[0], -1, 2)
    center = bbox[:, None, 0:2]
    size = bbox[:, None, 2:4].clamp_min(1e-4)
    top_left = center - 0.5 * size
    out = pts * size + top_left
    return out.clamp(0.0, 1.0).reshape(local_points.shape[0], -1)


def roi_align_square(features: torch.Tensor, bbox_norm: torch.Tensor, output_size: int) -> torch.Tensor:
    batch = features.shape[0]
    dtype = features.dtype
    device = features.device
    lin = torch.linspace(-0.5, 0.5, int(output_size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    center = bbox_norm[:, 0:2].to(dtype)
    size = bbox_norm[:, 2:4].clamp_min(1e-4).to(dtype)
    grid_x = center[:, None, None, 0] + xx[None] * size[:, None, None, 0]
    grid_y = center[:, None, None, 1] + yy[None] * size[:, None, None, 1]
    grid = torch.stack([grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0], dim=-1)
    if grid.shape[0] != batch:
        raise RuntimeError("ROI grid batch mismatch.")
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def sample_feature_at_points(features: torch.Tensor, points_norm: torch.Tensor) -> torch.Tensor:
    batch, _, _, _ = features.shape
    pts = points_norm.reshape(batch, -1, 2).to(features.dtype)
    grid = pts * 2.0 - 1.0
    sampled = F.grid_sample(features, grid[:, :, None, :], mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


def default_heatmap_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor], settings: dict[str, Any], task_id: str) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(torch.sigmoid(outputs["heatmap_logits"]), target["heatmap"])
    return loss, {"total_loss": float(loss.detach().cpu().item()), "heatmap_loss": float(loss.detach().cpu().item())}


def default_decode(outputs: dict[str, torch.Tensor], settings: dict[str, Any], task_id: str) -> torch.Tensor:
    return decode_argmax_norm(outputs["heatmap_logits"])


def save_split_csv(path: Path, dataframe: pd.DataFrame, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.iloc[indices].reset_index(drop=True).to_csv(path, index=False)


def save_checkpoint(path: Path, model: nn.Module, settings: dict[str, Any], task_configs: list[dict[str, Any]], epoch: int, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "settings": settings,
            "task_configs": task_configs,
            "epoch": int(epoch),
            "best_val_task_macro_mre_original_px": float(best_score),
        },
        path,
    )


def validate_settings(settings: dict[str, Any]) -> None:
    input_size = int(settings["input_size"])
    patch = int(settings.get("encoder_patch_size", 14))
    if input_size % patch != 0:
        raise ValueError(f"input_size={input_size} must be divisible by encoder_patch_size={patch}")
    if int(settings["heatmap_size"]) <= 0:
        raise ValueError("heatmap_size must be positive")


def make_dataset(dataframe: pd.DataFrame, settings: dict[str, Any], augment: bool, seed: int) -> RaceTrainDataset:
    return RaceTrainDataset(
        dataframe=dataframe,
        input_size=int(settings["input_size"]),
        heatmap_size=int(settings["heatmap_size"]),
        heatmap_sigma=float(settings["heatmap_sigma"]),
        bbox_expand=float(settings.get("bbox_expand", 1.8)),
        augment=augment,
        seed=seed,
    )


def evaluate_model(model: nn.Module, local_model: Any, loader: DataLoader, device: torch.device, settings: dict[str, Any]) -> pd.DataFrame:
    model.eval()
    per_task_errors: dict[str, list[float]] = defaultdict(list)
    input_size = int(settings["input_size"])
    decode_fn = getattr(local_model, "decode_outputs", default_decode)
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            for task_id in sorted(set(batch["task_id"])):
                indices = [idx for idx, value in enumerate(batch["task_id"]) if value == task_id]
                outputs = model(images[indices], task_id=task_id)
                pred_canvas_norm = decode_fn(outputs, settings, task_id)
                pred_original = canvas_norm_to_original_pixels(pred_canvas_norm, [batch["letterbox"][idx] for idx in indices], input_size)
                target = tensor_stack([batch["points_original"][idx] for idx in indices], device).reshape(len(indices), -1, 2)
                pred = pred_original.reshape(len(indices), -1, 2)
                distances = torch.linalg.norm(pred - target, dim=-1)
                per_task_errors[str(task_id)].extend(float(v) for v in distances.mean(dim=-1).detach().cpu().numpy().tolist())
    rows = []
    for task_id in sorted(per_task_errors):
        values = np.asarray(per_task_errors[task_id], dtype=np.float64)
        rows.append(
            {
                "task_id": task_id,
                "num_images": int(len(values)),
                "mre_original_px": float(values.mean()) if len(values) else float("nan"),
                "p50_original_px": float(np.percentile(values, 50)) if len(values) else float("nan"),
                "p90_original_px": float(np.percentile(values, 90)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def smoke_forward(run_dir: Path, settings: dict[str, Any], full_df: pd.DataFrame, task_configs: list[dict[str, Any]], local_model: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = local_model.build_model(settings, task_configs).to(device)
    model.eval()
    loss_fn = getattr(local_model, "compute_loss", default_heatmap_loss)
    decode_fn = getattr(local_model, "decode_outputs", default_decode)
    rows = []
    for config in task_configs:
        task_id = str(config["task_id"])
        sample_df = full_df[full_df["task_id"].astype(str) == task_id].head(1).reset_index(drop=True)
        item = make_dataset(sample_df, settings, augment=False, seed=int(settings["seed"]))[0]
        image = item["image"].unsqueeze(0).to(device)
        target = {
            "heatmap": item["heatmap"].unsqueeze(0).to(device),
            "center_heatmap": item["center_heatmap"].unsqueeze(0).to(device),
            "centers_hm": item["centers_hm"].unsqueeze(0).to(device),
            "points_canvas_norm": item["points_canvas_norm"].unsqueeze(0).to(device),
            "points_original": item["points_original"].unsqueeze(0).to(device),
            "bbox_norm": item["bbox_norm"].unsqueeze(0).to(device),
        }
        with torch.no_grad():
            outputs = model(image, task_id=task_id)
            loss, parts = loss_fn(outputs, target, settings, task_id)
            decoded = decode_fn(outputs, settings, task_id)
        rows.append(
            {
                "task_id": task_id,
                "num_points": int(config["num_classes"]),
                "decoded_shape": list(decoded.shape),
                "output_shapes": {
                    key: list(value.shape)
                    for key, value in outputs.items()
                    if isinstance(value, torch.Tensor)
                },
                "loss_parts": parts,
                "total_loss": float(loss.detach().cpu().item()),
            }
        )
    summary = {
        "status": "complete",
        "mode": "smoke_forward",
        "device": str(device),
        "model_name": getattr(local_model, "MODEL_NAME", local_model.__name__),
        "settings": settings,
        "rows": rows,
        "note": "Forward/loss/decode check only; no optimizer step.",
    }
    write_json(run_dir / "smoke_forward_summary.json", summary)
    print(json.dumps(summary, indent=2))


def merge_settings(defaults: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    settings = recursive_update(defaults, config)
    for key in [
        "run_dir",
        "train_csv_dir",
        "epochs",
        "batch_size",
        "val_batch_size",
        "val_split",
        "learning_rate",
        "input_size",
        "heatmap_size",
        "heatmap_sigma",
        "encoder",
        "encoder_weights",
        "num_workers",
        "seed",
        "max_steps_per_epoch",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if str(settings.get("encoder_weights", "")).lower() in {"", "none", "false"}:
        settings["encoder_weights"] = None
    settings["run_dir"] = str(resolve_project_path(settings["run_dir"]))
    settings["train_csv_dir"] = str(resolve_project_path(settings["train_csv_dir"]))
    return settings


def train_or_run(args: argparse.Namespace, settings: dict[str, Any], local_model: Any) -> None:
    validate_settings(settings)
    run_dir = Path(settings["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    write_json(run_dir / "config.resolved.json", settings)
    if args.config:
        shutil.copy2(args.config, run_dir / "config.input.json")

    set_seed(int(settings["seed"]))
    full_df = filter_tasks(load_canonical_train_dataframe(settings["train_csv_dir"]), settings.get("tasks"))
    task_configs = build_task_configs(full_df)
    write_json(run_dir / "task_configs.json", task_configs)
    train_indices, val_indices = stratified_split_indices(full_df, float(settings["val_split"]), int(settings["seed"]))
    save_split_csv(run_dir / "split_train.csv", full_df, train_indices)
    save_split_csv(run_dir / "split_val.csv", full_df, val_indices)
    split_info = {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "val_split": float(settings["val_split"]),
        "task_counts_total": full_df.groupby("task_id").size().to_dict(),
        "task_counts_train": full_df.iloc[train_indices].groupby("task_id").size().to_dict(),
        "task_counts_val": full_df.iloc[val_indices].groupby("task_id").size().to_dict(),
        "tasks": sorted(full_df["task_id"].astype(str).unique().tolist()),
    }
    write_json(run_dir / "split_info.json", split_info)

    if args.dry_run:
        summary = {
            "status": "complete",
            "mode": "dry_run",
            "model_name": getattr(local_model, "MODEL_NAME", local_model.__name__),
            "settings": settings,
            "split_info": split_info,
        }
        write_json(run_dir / "dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return

    if args.smoke_forward:
        smoke_forward(run_dir, settings, full_df, task_configs, local_model)
        return

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)
    train_dataset = make_dataset(train_df, settings, augment=True, seed=int(settings["seed"]))
    val_dataset = make_dataset(val_df, settings, augment=False, seed=int(settings["seed"]))
    train_sampler = TaskUniformBatchSampler(train_dataset, batch_size=int(settings["batch_size"]), steps_per_epoch=settings.get("max_steps_per_epoch"))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=int(settings["num_workers"]),
        pin_memory=True,
        collate_fn=race_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(settings["val_batch_size"]),
        shuffle=False,
        num_workers=int(settings["num_workers"]),
        pin_memory=True,
        collate_fn=race_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Run dir: {run_dir}")
    print(f"Train/val split: {len(train_dataset)} / {len(val_dataset)}")

    model = local_model.build_model(settings, task_configs).to(device)
    init_checkpoint = settings.get("init_checkpoint")
    if init_checkpoint:
        init_path = resolve_project_path(init_checkpoint)
        try:
            init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        except TypeError:
            init_payload = torch.load(init_path, map_location="cpu")
        init_state = init_payload["model_state"] if isinstance(init_payload, dict) and "model_state" in init_payload else init_payload
        result = model.load_state_dict(init_state, strict=bool(settings.get("init_strict", False)))
        print(f"Loaded init checkpoint: {init_path}")
        print(f"Missing keys: {list(result.missing_keys)}")
        print(f"Unexpected keys: {list(result.unexpected_keys)}")
    base_lr = float(settings["learning_rate"])
    if hasattr(local_model, "optimizer_param_groups"):
        param_groups = local_model.optimizer_param_groups(model, base_lr, settings)
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(settings["epochs"]), eta_min=1e-6)
    loss_fn = getattr(local_model, "compute_loss", default_heatmap_loss)
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(settings.get("amp_dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["epoch", "task_id", "train_total_loss", "val_mre_original_px"]).writeheader()

    best_score = float("inf")
    best_epoch = None
    save_epoch_checkpoints = {int(v) for v in settings.get("save_epoch_checkpoints", [])}
    for epoch in range(1, int(settings["epochs"]) + 1):
        epoch_settings = dict(settings)
        epoch_settings["current_epoch"] = epoch
        model.train()
        losses_by_task: dict[str, list[float]] = defaultdict(list)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{settings['epochs']} [Train]"):
            images = batch["image"].to(device, non_blocking=True)
            for task_id in sorted(set(batch["task_id"])):
                indices = [idx for idx, value in enumerate(batch["task_id"]) if value == task_id]
                optimizer.zero_grad(set_to_none=True)
                target = batch_target(batch, indices, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(images[indices], task_id=task_id)
                    loss, _ = loss_fn(outputs, target, epoch_settings, task_id)
                scaler.scale(loss).backward()
                if float(settings.get("grad_clip_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                losses_by_task[str(task_id)].append(float(loss.detach().cpu().item()))
        metrics = evaluate_model(model, local_model, val_loader, device, settings)
        metrics.to_csv(run_dir / f"val_metrics_epoch_{epoch:03d}.csv", index=False)
        selected = float(metrics["mre_original_px"].mean()) if not metrics.empty else float("inf")
        print(f"Epoch {epoch} validation task-macro original-pixel MRE: {selected:.6f}")
        print(metrics.to_string(index=False))
        val_by_task = {str(row["task_id"]): float(row["mre_original_px"]) for _, row in metrics.iterrows()}
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "task_id", "train_total_loss", "val_mre_original_px"])
            for task_id in sorted(losses_by_task):
                writer.writerow(
                    {
                        "epoch": epoch,
                        "task_id": task_id,
                        "train_total_loss": float(np.mean(losses_by_task[task_id])),
                        "val_mre_original_px": val_by_task.get(task_id, ""),
                    }
                )
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, settings, task_configs, epoch, min(best_score, selected))
        if epoch in save_epoch_checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, settings, task_configs, epoch, min(best_score, selected))
        if selected < best_score:
            best_score = selected
            best_epoch = epoch
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, settings, task_configs, epoch, best_score)
            save_checkpoint(run_dir / "checkpoints" / "best_mre.pt", model, settings, task_configs, epoch, best_score)
            save_checkpoint(run_dir / "best_model.pth", model, settings, task_configs, epoch, best_score)
        scheduler.step()
        write_json(
            run_dir / "run_summary.json",
            {
                "status": "running",
                "completed_epochs": epoch,
                "best_epoch": best_epoch,
                "best_val_task_macro_mre_original_px": best_score,
                "settings": settings,
                "split_info": split_info,
            },
        )

    write_json(
        run_dir / "run_summary.json",
        {
            "status": "complete",
            "completed_epochs": int(settings["epochs"]),
            "best_epoch": best_epoch,
            "best_val_task_macro_mre_original_px": best_score,
            "settings": settings,
            "split_info": split_info,
        },
    )


def run_experiment_cli(local_model: Any, defaults: dict[str, Any], description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--train-csv-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--heatmap-size", type=int, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-forward", action="store_true")
    args = parser.parse_args()
    settings = merge_settings(defaults, load_json(args.config), args)
    run_dir = Path(settings["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / ("dry_run.log" if args.dry_run else "smoke_forward.log" if args.smoke_forward else "train.log")
    with log_path.open("a", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            train_or_run(args, settings, local_model)
