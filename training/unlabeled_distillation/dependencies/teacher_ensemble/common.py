from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DENSE_FUSION_SRC = PROJECT_ROOT / "training/dense_fusion/src"
RACE_SRC = PROJECT_ROOT / "training/supervised_convnext/dependencies/shared"
if str(RACE_SRC) not in sys.path:
    sys.path.append(str(RACE_SRC))
from foundus_race_lib import letterbox_rgb, normalize_image_to_tensor  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(value: str | Path) -> str:
    path = resolve_path(value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_rank(seed: int, task: str, canonical_id: str) -> str:
    value = f"{int(seed)}|{task}|{canonical_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_task_balanced_sample(
    frame: pd.DataFrame,
    tasks: list[str],
    count_per_task: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for task in tasks:
        current = frame[frame["task_id"].astype(str) == str(task)].copy()
        if len(current) < int(count_per_task):
            raise RuntimeError(
                f"Task {task} has {len(current)} rows, below {count_per_task}."
            )
        current["selection_rank"] = [
            deterministic_rank(seed, task, value)
            for value in current["sha256"].astype(str)
        ]
        current = current.sort_values(
            ["selection_rank", "sha256"], kind="mergesort"
        ).head(int(count_per_task))
        rows.append(current)
    selected = pd.concat(rows, ignore_index=True)
    selected = selected.sort_values(
        ["task_id", "selection_rank", "sha256"], kind="mergesort"
    ).reset_index(drop=True)
    if selected["sha256"].astype(str).duplicated().any():
        raise RuntimeError("The task-balanced unlabeled sample contains duplicates.")
    return selected


class UnlabeledDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, input_size: int):
        self.frame = frame.reset_index(drop=True)
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        source = str(row["source_image_path"])
        image = cv2.imread(str(resolve_path(source)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas, _ = letterbox_rgb(image, self.input_size)
        return {
            "image": normalize_image_to_tensor(canvas),
            "task_id": str(row["task_id"]),
            "canonical_id": str(row["sha256"]),
        }


class HomogeneousTaskBatchSampler(Sampler[list[int]]):
    def __init__(self, frame: pd.DataFrame, batch_size: int):
        self.batches: list[list[int]] = []
        tasks = frame["task_id"].astype(str).to_numpy()
        for task in sorted(set(tasks)):
            indices = np.flatnonzero(tasks == task).tolist()
            for start in range(0, len(indices), int(batch_size)):
                self.batches.append(indices[start : start + int(batch_size)])

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_unlabeled(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"A batch mixes tasks: {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "canonical_id": [str(item["canonical_id"]) for item in items],
    }


def decode_probability(probability: torch.Tensor, topk: int = 25) -> torch.Tensor:
    if probability.ndim != 4:
        raise ValueError(f"Expected BxKxHxW probability, got {probability.shape}.")
    batch, points, height, width = probability.shape
    flat = probability.reshape(batch, points, -1)
    count = min(int(topk), flat.shape[-1])
    values, indices = torch.topk(flat, k=count, dim=-1)
    weights = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
    x = (indices % width).to(probability.dtype) / max(width - 1, 1)
    y = (indices // width).to(probability.dtype) / max(height - 1, 1)
    return torch.stack(
        ((weights * x).sum(-1), (weights * y).sum(-1)), dim=-1
    )


def symmetric_js(
    candidate: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    candidate = candidate.clamp_min(1e-12)
    target = target.clamp_min(1e-12)
    middle = 0.5 * (candidate + target)
    return 0.5 * (
        (candidate * (candidate.log() - middle.log())).sum(-1)
        + (target * (target.log() - middle.log())).sum(-1)
    )


def average_model_states(
    states: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not states:
        raise ValueError("No states were supplied.")
    keys = list(states[0])
    for index, state in enumerate(states[1:], 1):
        if list(state) != keys:
            raise RuntimeError(f"State-key order differs at model {index}.")
    averaged: dict[str, torch.Tensor] = {}
    nonfloating_differences = []
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise RuntimeError(f"State shape differs for {key}.")
        if any(value.dtype != reference.dtype for value in values[1:]):
            raise RuntimeError(f"State dtype differs for {key}.")
        if torch.is_floating_point(reference) or torch.is_complex(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for value in values:
                accumulator.add_(value.float(), alpha=1.0 / len(values))
            averaged[key] = accumulator.to(reference.dtype)
        elif key.endswith("num_batches_tracked"):
            mean = sum(float(value.item()) for value in values) / len(values)
            averaged[key] = torch.as_tensor(
                round(mean), dtype=reference.dtype
            ).reshape(reference.shape)
        else:
            equal = all(torch.equal(reference, value) for value in values[1:])
            if not equal:
                nonfloating_differences.append(key)
            averaged[key] = reference.clone()
    if nonfloating_differences:
        raise RuntimeError(
            "Non-floating model state differs: "
            + ", ".join(nonfloating_differences[:10])
        )
    return averaged, {
        "models_averaged": len(states),
        "state_keys": len(keys),
        "nonfloating_differences": nonfloating_differences,
        "batch_counter_policy": "rounded arithmetic mean",
    }


def load_payloads(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    records = []
    reference_task_map = None
    reference_keys = None
    for item in config["checkpoints"]:
        path = resolve_path(item["path"])
        actual = sha256_file(path)
        if actual != str(item["sha256"]):
            raise RuntimeError(
                f"Checkpoint checksum mismatch for fold {item['fold']}: {actual}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model_state", "settings", "task_configs"}
        if not required.issubset(payload):
            raise RuntimeError(f"Checkpoint {path} is missing required fields.")
        task_map = {
            str(value["task_id"]): int(value["num_classes"])
            for value in payload["task_configs"]
        }
        keys = list(payload["model_state"])
        if reference_task_map is None:
            reference_task_map = task_map
            reference_keys = keys
        elif task_map != reference_task_map or keys != reference_keys:
            raise RuntimeError("Fold checkpoints have incompatible model structures.")
        payloads.append(payload)
        records.append(
            {
                "fold": int(item["fold"]),
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": actual,
                "epoch": int(payload.get("epoch", -1)),
                "state_keys": len(keys),
            }
        )
    return payloads, records


def build_model(
    dense_fusion: Any,
    payload: dict[str, Any],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    model = dense_fusion.build_model(payload["settings"], payload["task_configs"])
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict model load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    task_scales: dict[str, float],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    description: str,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {}
    for batch in tqdm(loader, desc=description, leave=False):
        task = str(batch["task_id"])
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            result = model(images, task)
            scale = float(task_scales[task])
            logits = result["base_heatmap_logits"] + scale * result[
                "residual_heatmap_logits"
            ]
        probability = torch.softmax(logits.float().flatten(2), dim=-1)
        probability = probability.reshape_as(logits).cpu().to(torch.float16).numpy()
        outputs.setdefault(task, []).append(probability)
    return {task: np.concatenate(values, axis=0) for task, values in outputs.items()}
