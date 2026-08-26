from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != str(expected).lower():
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={resolve_path(path)}")
    return actual


def load_unlabeled_manifest(settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_path(settings["unlabeled_manifest"])
    if not path.exists():
        raise FileNotFoundError(f"Missing unlabeled manifest: {path}")
    frame = pd.read_csv(path)
    required = {"source_image_path", "task_id", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Unlabeled manifest missing columns: {sorted(missing)}")

    frame["source_image_path"] = frame["source_image_path"].astype(str)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["sha256"] = frame["sha256"].astype(str)
    if bool(settings.get("require_foundus_unlabeled_path_marker", True)):
        marker = frame["source_image_path"].str.replace("\\", "/", regex=False).str.lower().str.contains("/unlabeled/")
        if not bool(marker.all()):
            examples = frame.loc[~marker, "source_image_path"].head(10).tolist()
            raise RuntimeError(f"Unlabeled manifest contains paths outside task unlabeled folders: {examples}")
    bad_paths = {str(resolve_path(path)) for path in settings.get("bad_image_paths", [])}
    bad_mask = frame["source_image_path"].map(lambda value: str(resolve_path(value)) in bad_paths)
    bad_rows = frame[bad_mask].copy()
    frame = frame[~bad_mask].copy()

    enabled = settings.get("unlabeled_tasks")
    if enabled:
        enabled_set = {str(task_id) for task_id in enabled}
        frame = frame[frame["task_id"].isin(enabled_set)].copy()
    if frame.empty:
        raise ValueError("No canonical unlabeled images remain after filtering.")
    if frame["source_image_path"].duplicated().any():
        raise ValueError("Canonical unlabeled manifest has duplicate source_image_path values.")
    if frame["sha256"].duplicated().any():
        raise ValueError("Canonical unlabeled manifest has duplicate SHA256 values.")

    frame = frame.reset_index(drop=True)
    frame["canonical_id"] = frame["sha256"]
    frame["manifest_index"] = np.arange(len(frame), dtype=np.int64)
    summary_path = settings.get("unlabeled_summary")
    raw_total = None
    duplicate_count = None
    duplicate_groups = None
    if summary_path and resolve_path(summary_path).exists():
        import json

        with resolve_path(summary_path).open("r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
        raw_total = summary_payload.get("unlabeled_raw_total_images")
        duplicate_count = summary_payload.get("unlabeled_duplicates_removed")
        duplicate_groups = summary_payload.get("unlabeled_duplicate_groups")

    info = {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "unlabeled_total_unique": int(len(frame)),
        "unlabeled_bad_image_skip_count": int(len(bad_rows)),
        "unlabeled_bad_image_paths": bad_rows["source_image_path"].tolist(),
        "unlabeled_unique_by_task": frame.groupby("task_id").size().astype(int).to_dict(),
        "unlabeled_raw_total_images": raw_total,
        "unlabeled_duplicates_represented": duplicate_count,
        "unlabeled_duplicate_groups": duplicate_groups,
        "canonical_representation_note": (
            "Every SHA256-deduplicated canonical image is scheduled once. Exact duplicate files are represented "
            "by their canonical image; unreadable files are explicitly listed."
        ),
    }
    expected = settings.get("unlabeled_manifest_sha256")
    if expected and info["manifest_sha256"].lower() != str(expected).lower():
        raise RuntimeError(
            f"Unlabeled manifest SHA256 mismatch: expected={expected}, actual={info['manifest_sha256']}"
        )
    return frame, info


@dataclass(frozen=True)
class UnlabeledBatchSpec:
    task_id: str
    indices: tuple[int, ...]


def _interleave_task_batches(
    frame: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> tuple[list[UnlabeledBatchSpec], dict[str, int]]:
    rng = random.Random(int(seed))
    batches: list[UnlabeledBatchSpec] = []
    counts: dict[str, int] = {}
    for task_id, group in frame.groupby("task_id", sort=True):
        indices = group.index.astype(int).tolist()
        rng.shuffle(indices)
        task_batches = [tuple(indices[start : start + batch_size]) for start in range(0, len(indices), batch_size)]
        counts[str(task_id)] = len(task_batches)
        batches.extend(UnlabeledBatchSpec(str(task_id), values) for values in task_batches)
    rng.shuffle(batches)
    return batches, counts


def build_exhaustive_schedule(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    labeled_frame: pd.DataFrame | None = None,
) -> tuple[dict[int, list[UnlabeledBatchSpec]], dict[str, Any]]:
    epochs = int(settings["epochs"])
    warmup = int(settings.get("ssl_warmup_epochs", 0))
    active_epochs = list(range(warmup + 1, epochs + 1))
    if not active_epochs:
        active_epochs = list(range(1, epochs + 1))
    all_batches, task_batch_counts = _interleave_task_batches(
        frame,
        int(settings["unlabeled_batch_size"]),
        int(settings["seed"]),
    )
    partitions: dict[int, list[UnlabeledBatchSpec]] = {epoch: [] for epoch in range(1, epochs + 1)}
    for position, batch in enumerate(all_batches):
        partitions[active_epochs[position % len(active_epochs)]].append(batch)
    for epoch in active_epochs:
        random.Random(int(settings["seed"]) + epoch * 1009).shuffle(partitions[epoch])

    active_step_target = max((len(partitions[epoch]) for epoch in active_epochs), default=0)
    task_pool = [batch.task_id for batch in all_batches]
    for epoch in range(1, min(warmup, epochs) + 1):
        rng = random.Random(int(settings["seed"]) + epoch * 9176)
        partitions[epoch] = [UnlabeledBatchSpec(rng.choice(task_pool), tuple()) for _ in range(active_step_target)]

    mean_batches = float(np.mean(list(task_batch_counts.values())))
    power = float(settings.get("unlabeled_task_balance_power", 0.5))
    minimum = float(settings.get("unlabeled_task_weight_min", 0.25))
    maximum = float(settings.get("unlabeled_task_weight_max", 4.0))
    task_weights = {
        task_id: float(np.clip((mean_batches / max(count, 1)) ** power, minimum, maximum))
        for task_id, count in task_batch_counts.items()
    }
    supervised_only_batches: dict[str, int] = {}
    if labeled_frame is not None:
        for task_id in [str(value) for value in settings.get("supervised_only_tasks", ["fetal_femur"])]:
            row_count = int((labeled_frame["task_id"].astype(str) == task_id).sum())
            if row_count <= 0:
                raise ValueError(f"No labeled rows for supervised-only task {task_id}")
            batch_count = int(math.ceil(row_count / int(settings["batch_size"])))
            supervised_only_batches[task_id] = batch_count
            task_weights[task_id] = float(settings.get("supervised_only_task_weight", 1.0))
            for epoch in range(1, epochs + 1):
                partitions[epoch].extend(UnlabeledBatchSpec(task_id, tuple()) for _ in range(batch_count))
                random.Random(int(settings["seed"]) + epoch * 12011).shuffle(partitions[epoch])
    max_steps = settings.get("max_steps_per_epoch")
    if max_steps is not None:
        partitions = {epoch: batches[: int(max_steps)] for epoch, batches in partitions.items()}
    scheduled_indices = [idx for epoch in active_epochs for batch in partitions[epoch] for idx in batch.indices]
    schedule_info = {
        "epochs": epochs,
        "ssl_warmup_epochs": warmup,
        "active_ssl_epochs": active_epochs,
        "unlabeled_batch_size": int(settings["unlabeled_batch_size"]),
        "total_canonical_batches": int(len(all_batches)),
        "scheduled_unique_indices": int(len(set(scheduled_indices))),
        "expected_unique_indices": int(len(frame)),
        "task_batch_counts": task_batch_counts,
        "task_loss_weights": task_weights,
        "supervised_only_task_batches_per_epoch": supervised_only_batches,
        "steps_by_epoch": {str(epoch): len(batches) for epoch, batches in partitions.items()},
        "truncated_by_max_steps": max_steps is not None,
    }
    return partitions, schedule_info


def write_schedule_assignments(
    frame: pd.DataFrame,
    schedule: dict[int, list[UnlabeledBatchSpec]],
    path: Path,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    logical_digest = hashlib.sha256()
    for epoch in sorted(schedule):
        for batch_order, batch in enumerate(schedule[epoch]):
            for index in batch.indices:
                row = frame.iloc[int(index)]
                record = {
                    "canonical_id": str(row["canonical_id"]),
                    "task_id": str(batch.task_id),
                    "epoch": int(epoch),
                    "batch_order": int(batch_order),
                    "manifest_index": int(index),
                    "augmentation_seed": int(seed) + int(epoch) * 1000003 + int(index) * 7919,
                }
                rows.append(record)
                logical_digest.update(
                    (
                        f"{record['canonical_id']}|{record['task_id']}|{record['epoch']}|"
                        f"{record['batch_order']}|{record['manifest_index']}|{record['augmentation_seed']}\n"
                    ).encode("utf-8")
                )
    assignment = pd.DataFrame(rows)
    counts = assignment["canonical_id"].value_counts() if not assignment.empty else pd.Series(dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return {
        "assignment_path": str(path),
        "assignment_rows": int(len(assignment)),
        "assignment_unique": int(assignment["canonical_id"].nunique()) if not assignment.empty else 0,
        "assignment_repeated_unique": int((counts > 1).sum()),
        "assignment_max_repeat": int(counts.max()) if len(counts) else 0,
        "assignment_logical_sha256": logical_digest.hexdigest(),
        "assignment_file_sha256": sha256_file(path),
    }


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: list[UnlabeledBatchSpec]):
        self.batches = [list(batch.indices) for batch in batches if batch.indices]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def image_affine_to_grid_theta(matrix_weak_to_strong: np.ndarray, size: int) -> np.ndarray:
    full = np.eye(3, dtype=np.float32)
    full[:2] = matrix_weak_to_strong.astype(np.float32)
    inv = np.linalg.inv(full)
    pixel_from_norm = np.asarray(
        [
            [0.5 * (size - 1), 0.0, 0.5 * (size - 1)],
            [0.0, 0.5 * (size - 1), 0.5 * (size - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    norm_from_pixel = np.asarray(
        [
            [2.0 / max(size - 1, 1), 0.0, -1.0],
            [0.0, 2.0 / max(size - 1, 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (norm_from_pixel @ inv @ pixel_from_norm)[:2].astype(np.float32)


def _weak_photo(image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = image.astype(np.float32)
    gamma = float(rng.uniform(0.95, 1.05))
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)
    out = out * float(rng.uniform(0.95, 1.05)) + float(rng.uniform(-3.0, 3.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def _strong_photo(image: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    out = image.astype(np.float32)
    valid = np.ones(out.shape[:2], dtype=np.float32)
    gamma = float(rng.uniform(0.85, 1.18))
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)
    out = out * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-8.0, 8.0))
    if rng.rand() < 0.25:
        out += rng.normal(0.0, 3.0, size=out.shape).astype(np.float32)
    if rng.rand() < 0.15:
        out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0).astype(np.float32)
    if rng.rand() < 0.20:
        height, width = out.shape[:2]
        cut = int(rng.uniform(0.04, 0.10) * min(height, width))
        corner = int(rng.randint(0, 4))
        y0 = 0 if corner < 2 else height - cut
        x0 = 0 if corner % 2 == 0 else width - cut
        out[y0 : y0 + cut, x0 : x0 + cut] = 0
        valid[y0 : y0 + cut, x0 : x0 + cut] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8), valid


def _normalize_image(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).float()


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = float(size) / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    mask[y0 : y0 + resized_height, x0 : x0 + resized_width] = 1.0
    return canvas, mask


class UnlabeledGeoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        input_size: int,
        seed: int,
        epoch: int,
        spatial_mode: str = "geometry",
    ):
        self.frame = frame.reset_index(drop=True)
        self.input_size = int(input_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.spatial_mode = str(spatial_mode).lower()
        if self.spatial_mode not in {"geometry", "identity"}:
            raise ValueError(f"Unsupported SSL spatial mode: {self.spatial_mode}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        rng = np.random.RandomState(self.seed + self.epoch * 1000003 + int(index) * 7919)
        path = resolve_path(str(row["source_image_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read canonical unlabeled image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base, weak_content = _letterbox(image, self.input_size)

        if self.spatial_mode == "identity":
            matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            center = ((self.input_size - 1) * 0.5, (self.input_size - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(
                center,
                float(rng.uniform(-5.0, 5.0)),
                float(rng.uniform(0.92, 1.08)),
            ).astype(np.float32)
            matrix[0, 2] += float(rng.uniform(-0.04, 0.04) * self.input_size)
            matrix[1, 2] += float(rng.uniform(-0.04, 0.04) * self.input_size)
        strong_base = cv2.warpAffine(
            base,
            matrix,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        strong_content = cv2.warpAffine(
            weak_content,
            matrix,
            (self.input_size, self.input_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        strong_image, photometric_valid = _strong_photo(strong_base, rng)
        strong_content = strong_content * photometric_valid
        return {
            "weak": _normalize_image(_weak_photo(base, rng)),
            "strong": _normalize_image(strong_image),
            "weak_content_mask": torch.from_numpy(weak_content[None]).float(),
            "strong_content_mask": torch.from_numpy(strong_content[None]).float(),
            "matrix_weak_to_strong": torch.from_numpy(matrix).float(),
            "theta_strong_to_weak": torch.from_numpy(image_affine_to_grid_theta(matrix, self.input_size)).float(),
            "task_id": str(row["task_id"]),
            "source_image_path": str(row["source_image_path"]),
            "canonical_id": str(row["canonical_id"]),
        }


def unlabeled_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "weak": torch.stack([item["weak"] for item in batch]),
        "strong": torch.stack([item["strong"] for item in batch]),
        "weak_content_mask": torch.stack([item["weak_content_mask"] for item in batch]),
        "strong_content_mask": torch.stack([item["strong_content_mask"] for item in batch]),
        "matrix_weak_to_strong": torch.stack([item["matrix_weak_to_strong"] for item in batch]),
        "theta_strong_to_weak": torch.stack([item["theta_strong_to_weak"] for item in batch]),
        "task_id": [item["task_id"] for item in batch],
        "source_image_path": [item["source_image_path"] for item in batch],
        "canonical_id": [item["canonical_id"] for item in batch],
    }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None and hasattr(worker.dataset, "rng"):
        worker.dataset.rng = np.random.RandomState(seed)


def make_unlabeled_loader(
    frame: pd.DataFrame,
    batches: list[UnlabeledBatchSpec],
    settings: dict[str, Any],
    epoch: int,
) -> DataLoader:
    dataset = UnlabeledGeoDataset(
        frame,
        int(settings["input_size"]),
        int(settings["seed"]),
        epoch,
        spatial_mode=str(settings.get("ssl_spatial_mode", "geometry")),
    )
    generator = torch.Generator().manual_seed(int(settings["seed"]) + epoch * 1237)
    return DataLoader(
        dataset,
        batch_sampler=FixedBatchSampler(batches),
        num_workers=int(settings.get("unlabeled_num_workers", 0)),
        pin_memory=True,
        persistent_workers=False,
        collate_fn=unlabeled_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )


class CyclingLabeledBatches:
    def __init__(
        self,
        train_frame: pd.DataFrame,
        settings: dict[str, Any],
        make_dataset: Any,
        collate_fn: Any,
        epoch: int,
    ):
        self.train_frame = train_frame
        self.settings = settings
        self.make_dataset = make_dataset
        self.collate_fn = collate_fn
        self.epoch = int(epoch)
        self.cycles: dict[str, int] = {}
        self.iterators: dict[str, Iterator[dict[str, Any]]] = {}

    def _new_iterator(self, task_id: str) -> Iterator[dict[str, Any]]:
        cycle = self.cycles.get(task_id, 0)
        self.cycles[task_id] = cycle + 1
        frame = self.train_frame[self.train_frame["task_id"].astype(str) == str(task_id)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No labeled rehearsal rows for task {task_id}")
        dataset = self.make_dataset(
            frame,
            self.settings,
            augment=True,
            seed=int(self.settings["seed"]) + self.epoch * 1009 + cycle * 97,
        )
        generator = torch.Generator().manual_seed(int(self.settings["seed"]) + self.epoch * 7919 + cycle * 313)
        loader = DataLoader(
            dataset,
            batch_size=int(self.settings["batch_size"]),
            shuffle=True,
            drop_last=False,
            num_workers=int(self.settings.get("num_workers", 0)),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        return iter(loader)

    def next(self, task_id: str) -> dict[str, Any]:
        task_id = str(task_id)
        iterator = self.iterators.get(task_id)
        if iterator is None:
            iterator = self._new_iterator(task_id)
            self.iterators[task_id] = iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = self._new_iterator(task_id)
            self.iterators[task_id] = iterator
            return next(iterator)


def line_length(points: torch.Tensor, first: int, second: int) -> torch.Tensor:
    return torch.linalg.norm(points[:, first] - points[:, second], dim=-1)


def ellipse_circumference(points: torch.Tensor) -> torch.Tensor:
    first = line_length(points, 0, 1)
    second = line_length(points, 2, 3)
    a = torch.maximum(first, second) * 0.5
    b = torch.minimum(first, second) * 0.5
    h = ((a - b) ** 2) / ((a + b).clamp_min(1e-6) ** 2)
    return torch.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + torch.sqrt((4.0 - 3.0 * h).clamp_min(1e-6))))


def angle_degrees(points: torch.Tensor) -> torch.Tensor:
    # FoundUS AOP uses point 1 as the shared vertex and points 2/4 as arms.
    first = points[:, 1] - points[:, 0]
    second = points[:, 3] - points[:, 0]
    denominator = torch.linalg.norm(first, dim=-1).clamp_min(1e-6) * torch.linalg.norm(second, dim=-1).clamp_min(1e-6)
    cosine = ((first * second).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def measurement_values(points: torch.Tensor, task_id: str) -> torch.Tensor:
    pairs = {
        "A4C": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
        "FUGC": [(0, 1)],
        "IVC": [(0, 1)],
        "PLAX": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)],
        "PSAX": [(0, 1), (2, 3)],
        "fetal_femur": [(0, 1)],
    }
    if task_id == "HC" and points.shape[1] >= 4:
        return torch.stack([line_length(points, 0, 1), ellipse_circumference(points)], dim=-1)
    if task_id == "FA" and points.shape[1] >= 4:
        return ellipse_circumference(points).unsqueeze(-1)
    if task_id == "AOP" and points.shape[1] >= 4:
        return torch.stack([angle_degrees(points), line_length(points, 0, 2)], dim=-1)
    task_pairs = pairs.get(str(task_id), [])
    if not task_pairs:
        return points.new_zeros((points.shape[0], 0))
    return torch.stack([line_length(points, first, second) for first, second in task_pairs], dim=-1)


def evaluate_with_proxy(
    model: torch.nn.Module,
    local_model: Any,
    loader: DataLoader,
    device: torch.device,
    settings: dict[str, Any],
    adapter_scale: float,
    canvas_norm_to_original_pixels: Any,
    tensor_stack: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    input_size = int(settings["input_size"])
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Val scale={adapter_scale:g}]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            for task_id in sorted(set(batch["task_id"])):
                indices = [index for index, value in enumerate(batch["task_id"]) if value == task_id]
                base_scale = float(dict(settings.get("adapter_scale_by_task", {})).get(str(task_id), 1.0))
                effective_scale = float(adapter_scale) * base_scale
                outputs = model(
                    images[indices],
                    task_id=str(task_id),
                    adapter_enabled=True,
                    adapter_scale=effective_scale,
                )
                pred_norm = local_model.decode_outputs(outputs, settings, str(task_id))
                pred_original = canvas_norm_to_original_pixels(
                    pred_norm,
                    [batch["letterbox"][index] for index in indices],
                    input_size,
                ).reshape(len(indices), -1, 2)
                target_original = tensor_stack(
                    [batch["points_original"][index] for index in indices], device
                ).reshape(len(indices), -1, 2)
                mre = torch.linalg.norm(pred_original - target_original, dim=-1).mean(dim=-1)
                measurement_error = (
                    measurement_values(pred_original, str(task_id))
                    - measurement_values(target_original, str(task_id))
                ).abs()
                proxy = measurement_error.mean(dim=-1) if measurement_error.numel() else torch.zeros_like(mre)
                for local_index, source_index in enumerate(indices):
                    rows.append(
                        {
                            "task_id": str(task_id),
                            "source_image_path": str(batch["source_image_path"][source_index]),
                            "mre_original_px": float(mre[local_index].cpu()),
                            "measurement_proxy_mae": float(proxy[local_index].cpu()),
                            "adapter_scale": float(adapter_scale),
                            "task_base_scale": base_scale,
                            "effective_adapter_scale": effective_scale,
                        }
                    )
    per_image = pd.DataFrame(rows)
    per_task = (
        per_image.groupby("task_id", sort=True)
        .agg(
            num_images=("mre_original_px", "size"),
            mre_original_px=("mre_original_px", "mean"),
            measurement_proxy_mae=("measurement_proxy_mae", "mean"),
        )
        .reset_index()
    )
    per_task["final_proxy_score"] = 0.5 * per_task["mre_original_px"] + 0.5 * per_task["measurement_proxy_mae"]
    summary = {
        "adapter_scale": float(adapter_scale),
        "task_macro_mre_original_px": float(per_task["mre_original_px"].mean()),
        "task_macro_parameter_mae_proxy": float(per_task["measurement_proxy_mae"].mean()),
        "final_proxy_score": float(per_task["final_proxy_score"].mean()),
    }
    return per_image, per_task, summary


def paired_delta(adapter: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    merged = adapter.merge(
        anchor[["task_id", "source_image_path", "mre_original_px", "measurement_proxy_mae"]],
        on=["task_id", "source_image_path"],
        suffixes=("_adapter", "_anchor"),
        validate="one_to_one",
    )
    merged["delta_mre_adapter_minus_anchor"] = merged["mre_original_px_adapter"] - merged["mre_original_px_anchor"]
    merged["delta_measurement_proxy_adapter_minus_anchor"] = (
        merged["measurement_proxy_mae_adapter"] - merged["measurement_proxy_mae_anchor"]
    )
    return merged


def partition_coverage(
    frame: pd.DataFrame,
    seen_counts: Counter[str],
    draws_by_task: dict[str, int],
) -> dict[str, Any]:
    per_task = frame.groupby("task_id")["canonical_id"].agg(list).to_dict()
    unique_by_task = {
        str(task_id): sum(1 for value in ids if seen_counts.get(str(value), 0) > 0)
        for task_id, ids in per_task.items()
    }
    total = int(len(frame))
    expected_ids = set(frame["canonical_id"].astype(str))
    seen_ids = {value for value, count in seen_counts.items() if count > 0}
    seen = int(len(expected_ids & seen_ids))
    repeated = {value: count for value, count in seen_counts.items() if count > 1}
    unexpected = seen_ids - expected_ids
    return {
        "unlabeled_total_unique": total,
        "unlabeled_seen_unique": seen,
        "unlabeled_seen_ratio": float(seen / max(total, 1)),
        "unlabeled_total_draws": int(sum(seen_counts.values())),
        "unlabeled_missing_unique": int(len(expected_ids - seen_ids)),
        "unlabeled_repeated_unique": int(len(repeated)),
        "unlabeled_max_repeat_count": int(max(repeated.values(), default=1)),
        "unlabeled_unexpected_unique": int(len(unexpected)),
        "unlabeled_draws_by_task": {str(key): int(value) for key, value in draws_by_task.items()},
        "unlabeled_unique_by_task": unique_by_task,
        "bad_image_skip_count": 0,
    }
