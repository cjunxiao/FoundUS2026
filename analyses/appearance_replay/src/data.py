from __future__ import annotations

import hashlib
import importlib.util
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP161_PROTOCOL_PATH = PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/protocol.py"
BRIDGE_PROTOCOL_PATH = PROJECT_ROOT / "2-code/116-exp56-ssl-bridge-exp61/src/bridge_protocol.py"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fold_protocol = _load_module("exp161_protocol_for_exp164", EXP161_PROTOCOL_PATH)
bridge_protocol = _load_module("bridge_protocol_for_exp164", BRIDGE_PROTOCOL_PATH)

Exp164LabeledDataset = fold_protocol.Exp148Dataset
SequentialTaskBatchSampler = fold_protocol.SequentialTaskBatchSampler
collate_labeled = fold_protocol.collate_exp148
load_fold = fold_protocol.load_exp148_fold
worker_init_fn = fold_protocol.worker_init_fn
load_unlabeled_manifest = bridge_protocol.load_unlabeled_manifest
build_exhaustive_schedule = bridge_protocol.build_exhaustive_schedule
write_schedule_assignments = bridge_protocol.write_schedule_assignments
partition_coverage = bridge_protocol.partition_coverage
UnlabeledBatchSpec = bridge_protocol.UnlabeledBatchSpec


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = float(size) / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    mask[y0 : y0 + resized_height, x0 : x0 + resized_width] = 1.0
    return canvas, mask


def _normalize(image: np.ndarray) -> torch.Tensor:
    value = image.astype(np.float32) / 255.0
    value = (value - IMAGENET_MEAN[None, None]) / IMAGENET_STD[None, None]
    return torch.from_numpy(value.transpose(2, 0, 1)).float()


class StyleDonorDataset(Dataset):
    """Raw letterboxed images used only as appearance-statistic donors."""

    def __init__(self, frame: pd.DataFrame, input_size: int, id_column: str):
        self.frame = frame.reset_index(drop=True)
        self.input_size = int(input_size)
        self.id_column = str(id_column)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        path = resolve_path(str(row["source_image_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not decode style donor: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas, mask = _letterbox(image, self.input_size)
        return {
            "image": _normalize(canvas),
            "content_mask": torch.from_numpy(mask[None]).float(),
            "task_id": str(row["task_id"]),
            "source_image_path": str(row["source_image_path"]),
            "donor_id": str(row[self.id_column]),
            "dataset_index": int(index),
        }


def collate_donors(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"Style donor batch mixes tasks: {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "content_mask": torch.stack([item["content_mask"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "source_image_path": [str(item["source_image_path"]) for item in items],
        "donor_id": [str(item["donor_id"]) for item in items],
        "dataset_index": [int(item["dataset_index"]) for item in items],
    }


class FixedPlanBatchSampler(Sampler[list[Any]]):
    def __init__(self, batches: list[list[Any]]):
        self.batches = batches

    def __iter__(self) -> Iterator[list[Any]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class TaskIndexCycler:
    def __init__(self, frame: pd.DataFrame, seed: int):
        self.frame = frame
        self.rng = random.Random(int(seed))
        self.pools = {
            str(task): group.index.astype(int).tolist()
            for task, group in frame.groupby("task_id", sort=True)
        }
        self.cursors = {task: 0 for task in self.pools}
        self.occurrences: Counter[int] = Counter()
        for values in self.pools.values():
            self.rng.shuffle(values)

    def draw(self, task_id: str, count: int, with_occurrence: bool) -> list[Any]:
        task = str(task_id)
        if task not in self.pools:
            raise ValueError(f"No labeled rows for task {task}")
        output: list[Any] = []
        for _ in range(int(count)):
            if self.cursors[task] >= len(self.pools[task]):
                self.rng.shuffle(self.pools[task])
                self.cursors[task] = 0
            index = self.pools[task][self.cursors[task]]
            self.cursors[task] += 1
            occurrence = int(self.occurrences[index])
            self.occurrences[index] += 1
            output.append((index, occurrence) if with_occurrence else index)
        return output


def build_labeled_plans(
    train_frame: pd.DataFrame,
    epoch_batches: list[Any],
    batch_size: int,
    seed: int,
    epoch: int,
    supervised_only_tasks: set[str],
) -> tuple[list[list[tuple[int, int]]], list[list[int]], dict[str, Any]]:
    content = TaskIndexCycler(train_frame, int(seed) + int(epoch) * 10_007)
    donor = TaskIndexCycler(train_frame, int(seed) + int(epoch) * 20_011 + 1)
    content_batches: list[list[tuple[int, int]]] = []
    labeled_donor_batches: list[list[int]] = []
    content_digest = hashlib.sha256()
    donor_digest = hashlib.sha256()
    for step, spec in enumerate(epoch_batches):
        task = str(spec.task_id)
        content_batch = content.draw(task, int(batch_size), with_occurrence=True)
        content_batches.append(content_batch)
        for index, occurrence in content_batch:
            content_digest.update(
                f"{epoch}|{step}|{task}|{index}|{occurrence}\n".encode("utf-8")
            )
        if task in supervised_only_tasks or not spec.indices:
            continue
        donor_batch = donor.draw(task, len(spec.indices), with_occurrence=False)
        labeled_donor_batches.append(donor_batch)
        for index in donor_batch:
            donor_digest.update(
                f"{epoch}|{step}|{task}|{index}\n".encode("utf-8")
            )
    return content_batches, labeled_donor_batches, {
        "epoch": int(epoch),
        "steps": int(len(epoch_batches)),
        "content_schedule_sha256": content_digest.hexdigest(),
        "labeled_donor_schedule_sha256": donor_digest.hexdigest(),
        "active_style_batches": int(len(labeled_donor_batches)),
    }


def content_masks_from_letterbox(
    letterboxes: list[dict[str, Any]],
    input_size: int,
    device: torch.device,
) -> torch.Tensor:
    masks = []
    size = int(input_size)
    for meta in letterboxes:
        mask = torch.zeros((1, size, size), dtype=torch.float32)
        x0 = int(round(float(meta["pad_x"])))
        y0 = int(round(float(meta["pad_y"])))
        width = int(round(float(meta["resized_w"])))
        height = int(round(float(meta["resized_h"])))
        mask[:, y0 : y0 + height, x0 : x0 + width] = 1.0
        masks.append(mask)
    return torch.stack(masks).to(device, non_blocking=True)

