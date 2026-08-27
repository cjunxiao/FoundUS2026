from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RACE_SRC = PROJECT_ROOT / "training/supervised_convnext/dependencies/shared"
if str(RACE_SRC) not in sys.path:
    sys.path.append(str(RACE_SRC))

from foundus_race_lib import (  # noqa: E402
    apply_light_intensity_aug,
    build_task_configs,
    generate_gaussian_heatmaps,
    letterbox_rgb,
    load_canonical_train_dataframe,
    normalize_image_to_tensor,
    points_original_to_canvas,
    read_record_points,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(value: str | Path) -> str:
    path = resolve_path(value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_id(row: pd.Series) -> str:
    point_columns = sorted(
        column
        for column in row.index
        if str(column).startswith("point_") and str(column).endswith("_xy")
    )
    payload = {
        "task_id": str(row["task_id"]),
        "source_image_path": str(row["source_image_path"]),
        "image_path": str(row["image_path"]),
        "num_classes": int(row["num_classes"]),
        "points": {
            column: None if pd.isna(row[column]) else str(row[column])
            for column in point_columns
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_grouped_fold_fold(
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    full = load_canonical_train_dataframe(resolve_path(settings["train_csv_dir"])).reset_index(drop=True)
    wanted = settings.get("tasks")
    if wanted:
        selected = {str(value) for value in wanted}
        full = full[full["task_id"].astype(str).isin(selected)].reset_index(drop=True)
    full["row_id_current"] = [stable_row_id(row) for _, row in full.iterrows()]

    manifest_path = resolve_path(settings["protocol_manifest"])
    actual_sha = sha256_file(manifest_path)
    expected_sha = str(settings.get("protocol_manifest_sha256", ""))
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"Protocol manifest SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    manifest = pd.read_csv(manifest_path)
    required = {
        "row_id",
        "task_id",
        "source_image_path",
        "group_id",
        "grouping_evidence",
        "fold",
        "train_eligible",
        "validation_eligible",
        "excluded_reason",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"FiveFoldTrainer protocol manifest missing columns: {sorted(missing)}")
    merged = full.merge(
        manifest[list(required)],
        on=["task_id", "source_image_path"],
        how="left",
        validate="one_to_one",
    )
    if merged["fold"].isna().any():
        raise RuntimeError("FiveFoldTrainer protocol does not cover every selected canonical row.")
    mismatch = merged["row_id_current"].astype(str) != merged["row_id"].astype(str)
    if mismatch.any():
        raise RuntimeError(f"FiveFoldTrainer protocol row checksum mismatch: {int(mismatch.sum())} rows")
    for column in ("train_eligible", "validation_eligible"):
        if merged[column].dtype != bool:
            merged[column] = merged[column].astype(str).str.lower().map({"true": True, "false": False})
        if merged[column].isna().any():
            raise RuntimeError(f"Invalid boolean values in {column}")

    val_fold = int(settings["fold"])
    validation = merged[
        (merged["fold"].astype(int) == val_fold) & merged["validation_eligible"]
    ].copy()
    train = merged[
        merged["train_eligible"]
        & (
            (merged["fold"].astype(int) != val_fold)
            | (~merged["validation_eligible"])
        )
    ].copy()
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    overlap = set(zip(train.task_id.astype(str), train.group_id.astype(str))) & set(
        zip(validation.task_id.astype(str), validation.group_id.astype(str))
    )
    if overlap:
        raise RuntimeError(f"FiveFoldTrainer train/validation group leakage: {sorted(overlap)[:10]}")
    excluded = merged[~merged["train_eligible"] & ~merged["validation_eligible"]]
    if int((excluded["excluded_reason"] == "administrator_fetal_femur_mirror_exclusion").sum()) != 25:
        raise RuntimeError("FiveFoldTrainer must exclude exactly 25 administrator-listed fetal-femur rows.")

    active = merged[merged["train_eligible"] | merged["validation_eligible"]]
    task_configs = build_task_configs(active)
    info = {
        "protocol": "fivefold_trainer_active_vertical_explicit_sequence_grouped_5fold_filtered",
        "fold": val_fold,
        "protocol_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "protocol_manifest_sha256": actual_sha,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "excluded_rows": int(len(excluded)),
        "femur_excluded_rows": 25,
        "split_screen_training_only_rows": int(
            (merged["excluded_reason"] == "confirmed_split_screen_training_only_group").sum()
        ),
        "train_groups": int(train[["task_id", "group_id"]].drop_duplicates().shape[0]),
        "validation_groups": int(validation[["task_id", "group_id"]].drop_duplicates().shape[0]),
        "cross_fold_groups": 0,
        "patient_group_safe": False,
        "device_group_safe": False,
        "sequence_group_safe_for_explicit_frame_names": True,
        "task_counts_train": train.groupby("task_id").size().astype(int).to_dict(),
        "task_counts_validation": validation.groupby("task_id").size().astype(int).to_dict(),
        "grouping_limit": (
            "Canonical labels do not expose reliable patient/device identifiers. This protocol is "
            "explicit-sequence grouped screening, not patient/device-safe CV."
        ),
    }
    return train, validation, task_configs, info


class GroupedFoldDataset(Dataset):
    """BaselineHeatmap image pipeline with stateless per-occurrence augmentation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        settings: dict[str, Any],
        augment: bool,
        epoch: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.settings = settings
        self.augment = bool(augment)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(item, tuple):
            index, occurrence = int(item[0]), int(item[1])
        else:
            index, occurrence = int(item), 0
        row = self.frame.iloc[index]
        path = resolve_path(str(row["source_image_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not decode labeled image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        size = int(self.settings["input_size"])
        canvas, letterbox = letterbox_rgb(image, size)
        augmentation_seed = (
            int(self.settings["seed"])
            + self.epoch * 1_000_003
            + index * 7_919
            + occurrence * 97_409
        ) % (2**32)
        if self.augment:
            canvas = apply_light_intensity_aug(canvas, np.random.RandomState(augmentation_seed))

        points_original = read_record_points(row).reshape(-1, 2).astype(np.float32)
        points_original[:, 0] = np.clip(points_original[:, 0], 0.0, original_width - 1.0)
        points_original[:, 1] = np.clip(points_original[:, 1], 0.0, original_height - 1.0)
        points_canvas = points_original_to_canvas(points_original.reshape(-1), letterbox).reshape(-1, 2)
        points_norm = points_canvas / float(size - 1)
        heatmap_size = int(self.settings["heatmap_size"])
        centers = points_norm.copy()
        centers[:, 0] *= float(heatmap_size - 1)
        centers[:, 1] *= float(heatmap_size - 1)
        heatmap = generate_gaussian_heatmaps(
            centers,
            heatmap_size,
            float(self.settings["heatmap_sigma"]),
        )
        return {
            "image": normalize_image_to_tensor(canvas),
            "heatmap": torch.from_numpy(heatmap).float(),
            "points_norm": torch.from_numpy(points_norm).float(),
            "points_original": torch.from_numpy(points_original).float(),
            "task_id": str(row["task_id"]),
            "source_image_path": str(row["source_image_path"]),
            "group_id": str(row["group_id"]),
            "letterbox": letterbox,
            "dataset_index": index,
            "occurrence_id": occurrence,
            "augmentation_seed": int(augmentation_seed),
        }


def collate_grouped_fold(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"GroupedFold batches must be task homogeneous, got {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "heatmap": torch.stack([item["heatmap"] for item in items]),
        "points_norm": torch.stack([item["points_norm"] for item in items]),
        "points_original": torch.stack([item["points_original"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "source_image_path": [str(item["source_image_path"]) for item in items],
        "group_id": [str(item["group_id"]) for item in items],
        "letterbox": [item["letterbox"] for item in items],
        "dataset_index": [int(item["dataset_index"]) for item in items],
        "occurrence_id": [int(item["occurrence_id"]) for item in items],
        "augmentation_seed": [int(item["augmentation_seed"]) for item in items],
    }


class DeterministicTaskUniformBatchSampler(Sampler[list[tuple[int, int]]]):
    """Match BaselineHeatmap task-uniform sampling while making repeats independently augmented."""

    def __init__(self, dataset: GroupedFoldDataset, batch_size: int, seed: int, steps: int | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.steps = int(steps) if steps is not None else max(1, len(dataset) // self.batch_size)
        self.indices_by_task = {
            str(task): group.index.astype(int).tolist()
            for task, group in dataset.frame.groupby("task_id", sort=True)
        }
        if not self.indices_by_task:
            raise ValueError("Task-uniform sampler received an empty dataset.")
        self.last_audit: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = random.Random(self.seed + self.dataset.epoch * 104_729)
        task_ids = sorted(self.indices_by_task)
        pools = {task: list(values) for task, values in self.indices_by_task.items()}
        for values in pools.values():
            rng.shuffle(values)
        cursors = {task: 0 for task in task_ids}
        occurrences: Counter[int] = Counter()
        task_draws: Counter[str] = Counter()
        schedule_digest = hashlib.sha256()
        for _ in range(self.steps):
            task = rng.choice(task_ids)
            batch: list[tuple[int, int]] = []
            for _ in range(self.batch_size):
                if cursors[task] >= len(pools[task]):
                    rng.shuffle(pools[task])
                    cursors[task] = 0
                index = pools[task][cursors[task]]
                cursors[task] += 1
                occurrence = occurrences[index]
                occurrences[index] += 1
                batch.append((index, occurrence))
                schedule_digest.update(f"{task}:{index}:{occurrence}\n".encode("utf-8"))
            task_draws[task] += len(batch)
            yield batch
        counts = list(occurrences.values())
        self.last_audit = {
            "epoch": int(self.dataset.epoch),
            "steps": int(self.steps),
            "draws": int(sum(counts)),
            "unique_rows_drawn": int(len(occurrences)),
            "minimum_draw_count_for_drawn_rows": int(min(counts)) if counts else 0,
            "maximum_draw_count": int(max(counts)) if counts else 0,
            "task_draws": dict(sorted(task_draws.items())),
            "schedule_sha256": schedule_digest.hexdigest(),
            "sampling": "task_uniform_no_inverse_frequency_loss_weight",
        }


class SequentialTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: GroupedFoldDataset, batch_size: int):
        self.batches: list[list[int]] = []
        for _, group in dataset.frame.groupby("task_id", sort=True):
            indices = group.index.astype(int).tolist()
            self.batches.extend(
                indices[start : start + int(batch_size)]
                for start in range(0, len(indices), int(batch_size))
            )

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches


def phase_frame(train: pd.DataFrame, phase: dict[str, Any]) -> pd.DataFrame:
    tasks = {str(value) for value in phase.get("tasks", [])}
    if not tasks:
        return train.reset_index(drop=True)
    output = train[train["task_id"].astype(str).isin(tasks)].copy().reset_index(drop=True)
    if output.empty:
        raise RuntimeError(f"Phase {phase.get('name')} selected no rows for tasks {sorted(tasks)}")
    return output


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def task_uniform_expected_draws(frame: pd.DataFrame, batch_size: int) -> dict[str, float]:
    steps = max(1, len(frame) // int(batch_size))
    per_task = steps * int(batch_size) / max(frame["task_id"].nunique(), 1)
    return {
        str(task): float(per_task / max(int(count), 1))
        for task, count in frame.groupby("task_id", sort=True).size().items()
    }
