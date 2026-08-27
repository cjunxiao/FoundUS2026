from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
TEACHER_RUNTIME_SOURCE = (
    PROJECT_ROOT / "training/unlabeled_distillation/dependencies/teacher_ensemble/common.py"
)
DENSE_FUSION_SOURCE = (
    PROJECT_ROOT / "training/dense_fusion/src/model.py"
)
TEACHER_RUNTIME_SHA256 = "5b4a79bde04cf2d246f2e58b7ecd87406cb54b2c519b0afa38b3bc4cc2fe05f8"
DENSE_FUSION_SHA256 = "b3e007d1ae10e2a48574ad13668f007b94cd7eb3ea7b6f6fb45aff039787bbd4"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(value: str | Path) -> str:
    path = resolve(value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_locked(name: str, path: Path, expected_sha256: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    teacher_runtime = load_locked("teacher_runtime_for_final_distillation", TEACHER_RUNTIME_SOURCE, TEACHER_RUNTIME_SHA256)
    dense_fusion = load_locked("dense_fusion_for_final_distillation", DENSE_FUSION_SOURCE, DENSE_FUSION_SHA256)
    return teacher_runtime, dense_fusion


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def filesystem_used_bytes() -> int:
    return int(shutil.disk_usage(PROJECT_ROOT).used)


def enforce_space_limit(settings: dict[str, Any]) -> None:
    used = filesystem_used_bytes()
    maximum = int(settings["maximum_filesystem_used_bytes"])
    if used > maximum:
        raise RuntimeError(f"Filesystem use {used} exceeds locked limit {maximum}.")


def deterministic_rank(seed: int, namespace: str, canonical_id: str) -> str:
    payload = f"{int(seed)}|{namespace}|{canonical_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_manifest(settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = resolve(settings["manifest"])
    actual = sha256_file(manifest_path)
    if actual != str(settings["manifest_sha256"]):
        raise RuntimeError(f"Unlabeled manifest checksum mismatch: {actual}")
    frame = pd.read_csv(manifest_path)
    expected = int(settings["expected_unique_images"])
    if len(frame) != expected or frame["sha256"].astype(str).nunique() != expected:
        raise RuntimeError("Readable manifest is not the locked 182,870-ID set.")
    tasks = [str(value) for value in settings["tasks"]]
    if sorted(frame["task_id"].astype(str).unique()) != sorted(tasks):
        raise RuntimeError("Manifest task set differs from the configured task set.")
    missing = [value for value in frame["source_image_path"] if not resolve(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official unlabeled image: {missing[0]}")

    screen_path = resolve(settings["warmstart_split"])
    screen_sha = sha256_file(screen_path)
    if screen_sha != str(settings["warmstart_split_sha256"]):
        raise RuntimeError("WarmStart split checksum mismatch.")
    screen = pd.read_csv(screen_path)
    screen_ids = set(screen["sha256"].astype(str))
    if len(screen_ids) != len(screen) or not screen_ids.issubset(
        set(frame["sha256"].astype(str))
    ):
        raise RuntimeError("WarmStart split IDs are invalid for the readable manifest.")

    frame = frame.copy()
    frame["role"] = "full_train"
    role_by_id = dict(
        zip(
            screen["sha256"].astype(str),
            screen["stage0_split"].astype(str).map(
                {"train": "warmstart_train", "heldout": "warmstart_audit"}
            ),
        )
    )
    frame.loc[frame["sha256"].astype(str).isin(screen_ids), "role"] = (
        frame.loc[frame["sha256"].astype(str).isin(screen_ids), "sha256"]
        .astype(str)
        .map(role_by_id)
    )

    final_count = int(settings["final_audit_images_per_task"])
    final_audit_ids: set[str] = set()
    for task in tasks:
        current = frame[
            (frame["task_id"].astype(str) == task) & (frame["role"] == "full_train")
        ].copy()
        current["audit_rank"] = [
            deterministic_rank(int(settings["seed"]), "final-audit", value)
            for value in current["sha256"].astype(str)
        ]
        chosen = current.sort_values(["audit_rank", "sha256"], kind="mergesort").head(
            final_count
        )
        if len(chosen) != final_count:
            raise RuntimeError(f"Task {task} cannot supply the final audit split.")
        final_audit_ids.update(chosen["sha256"].astype(str))
    frame.loc[frame["sha256"].astype(str).isin(final_audit_ids), "role"] = "final_audit"

    frame = frame.sort_values(["task_id", "sha256"], kind="mergesort").reset_index(
        drop=True
    )
    frame["task_position"] = frame.groupby("task_id", sort=False).cumcount()
    role_counts = frame.groupby(["task_id", "role"]).size().unstack(fill_value=0)
    audit_ids = set(frame.loc[frame["role"] != "full_train", "sha256"].astype(str))
    train_ids = set(frame.loc[frame["role"] == "full_train", "sha256"].astype(str))
    if train_ids & audit_ids:
        raise RuntimeError("Full-bank train and audit IDs overlap.")
    summary = {
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": actual,
        "rows": int(len(frame)),
        "unique": int(frame["sha256"].astype(str).nunique()),
        "role_counts": {
            task: {role: int(value) for role, value in row.items()}
            for task, row in role_counts.to_dict(orient="index").items()
        },
        "full_train_images": int((frame["role"] == "full_train").sum()),
        "warmstart_train_images": int((frame["role"] == "warmstart_train").sum()),
        "warmstart_audit_images": int((frame["role"] == "warmstart_audit").sum()),
        "final_audit_images": int((frame["role"] == "final_audit").sum()),
        "train_audit_disjoint": True,
    }
    return frame, summary


class IndexedUnlabeledDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, input_size: int, teacher_runtime: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.base = teacher_runtime.UnlabeledDataset(self.frame, int(input_size))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[int(index)]
        row = self.frame.iloc[int(index)]
        item["task_position"] = int(row["task_position"])
        return item


class HomogeneousTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, frame: pd.DataFrame, batch_size: int) -> None:
        batches: list[list[int]] = []
        tasks = frame["task_id"].astype(str).to_numpy()
        for task in sorted(set(tasks)):
            indices = np.flatnonzero(tasks == task)
            for start in range(0, len(indices), int(batch_size)):
                batches.append(indices[start : start + int(batch_size)].tolist())
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_indexed(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"A batch mixes tasks: {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "canonical_id": [str(item["canonical_id"]) for item in items],
        "task_position": torch.tensor(
            [int(item["task_position"]) for item in items], dtype=torch.long
        ),
    }


def indexed_loader(
    frame: pd.DataFrame,
    input_size: int,
    batch_size: int,
    workers: int,
    teacher_runtime: Any,
) -> DataLoader:
    dataset = IndexedUnlabeledDataset(frame, input_size, teacher_runtime)
    return DataLoader(
        dataset,
        batch_sampler=HomogeneousTaskBatchSampler(frame, batch_size),
        num_workers=int(workers),
        persistent_workers=int(workers) > 0,
        pin_memory=True,
        collate_fn=collate_indexed,
    )


def spatial_probability(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    flat = logits.float().flatten(2) / max(float(temperature), 1e-6)
    return torch.softmax(flat, dim=-1).reshape_as(logits)


def topk_coordinates_px(probability: torch.Tensor, topk: int) -> torch.Tensor:
    _, _, height, width = probability.shape
    flat = probability.flatten(2)
    values, indices = torch.topk(flat, min(int(topk), flat.shape[-1]), dim=-1)
    weights = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
    x = (indices % width).to(probability.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(probability.dtype)
    return torch.stack(((weights * x).sum(-1), (weights * y).sum(-1)), dim=-1)


def symmetric_js(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate = candidate.flatten(2).clamp_min(1e-12)
    target = target.flatten(2).clamp_min(1e-12)
    middle = 0.5 * (candidate + target)
    return 0.5 * (
        (candidate * (candidate.log() - middle.log())).sum(-1)
        + (target * (target.log() - middle.log())).sum(-1)
    )


def bank_paths(bank_dir: Path, task: str) -> dict[str, Path]:
    safe = str(task).replace("/", "_")
    return {
        "sum": bank_dir / f"{safe}.probability_sum.float32.npy",
        "coordinate_sum": bank_dir / f"{safe}.coordinate_sum.float32.npy",
        "coordinate_square_sum": bank_dir / f"{safe}.coordinate_square_sum.float32.npy",
        "mean": bank_dir / f"{safe}.teacher_mean.float16.npy",
        "dispersion": bank_dir / f"{safe}.coordinate_dispersion.float16.npy",
    }


def snapshot_sources(run_dir: Path, config: Path) -> None:
    del run_dir, config

def output_manifest(run_dir: Path, hash_large_files: bool = True) -> None:
    rows = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "output_manifest.json":
            continue
        size = path.stat().st_size
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": size,
                "sha256": sha256_file(path) if hash_large_files or size < 1024**3 else None,
            }
        )
    write_json(run_dir / "output_manifest.json", rows)

