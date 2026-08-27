from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from model import FunctionalCompressionStudent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
TEACHER_RUNTIME_SOURCE = (
    PROJECT_ROOT / "training/unlabeled_distillation/dependencies/teacher_ensemble/common.py"
)
DENSE_FUSION_SOURCE = PROJECT_ROOT / "training/dense_fusion/src/model.py"


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


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot_sources(run_dir: Path, config: Path) -> None:
    del run_dir, config

def write_output_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "output_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(run_dir / "output_manifest.json", rows)


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def spatial_probability(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    flat = logits.float().flatten(2) / max(float(temperature), 1e-6)
    return torch.softmax(flat, dim=-1).reshape_as(logits)


def probability_coordinates(probability: torch.Tensor) -> torch.Tensor:
    batch, points, height, width = probability.shape
    flat = probability.reshape(batch, points, -1)
    x = torch.arange(width, dtype=probability.dtype, device=probability.device)
    y = torch.arange(height, dtype=probability.dtype, device=probability.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.reshape(-1) / max(width - 1, 1)
    grid_y = grid_y.reshape(-1) / max(height - 1, 1)
    return torch.stack(
        ((flat * grid_x).sum(-1), (flat * grid_y).sum(-1)), dim=-1
    )


def symmetric_js(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate = candidate.flatten(2).clamp_min(1e-12)
    target = target.flatten(2).clamp_min(1e-12)
    middle = 0.5 * (candidate + target)
    return 0.5 * (
        (candidate * (candidate.log() - middle.log())).sum(-1)
        + (target * (target.log() - middle.log())).sum(-1)
    )


def split_rank(seed: int, canonical_id: str) -> str:
    return hashlib.sha256(f"{seed}|stage0|{canonical_id}".encode()).hexdigest()


def make_split(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    rows = []
    train_count = int(settings["train_samples_per_task"])
    expected = int(settings["samples_per_task"])
    for task in settings["tasks"]:
        current = frame[frame["task_id"].astype(str) == str(task)].copy()
        if len(current) != expected:
            raise RuntimeError(f"Task {task} has {len(current)} samples, expected {expected}.")
        current["split_rank"] = [
            split_rank(int(settings["seed"]), value)
            for value in current["sha256"].astype(str)
        ]
        current = current.sort_values(["split_rank", "sha256"], kind="mergesort")
        current["stage0_split"] = "heldout"
        current.iloc[:train_count, current.columns.get_loc("stage0_split")] = "train"
        rows.append(current)
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["task_id", "selection_rank"], kind="mergesort")
    result["task_position"] = result.groupby("task_id", sort=False).cumcount()
    result = result.reset_index(drop=True)
    train_ids = set(result.loc[result["stage0_split"] == "train", "sha256"].astype(str))
    heldout_ids = set(
        result.loc[result["stage0_split"] == "heldout", "sha256"].astype(str)
    )
    if train_ids & heldout_ids:
        raise RuntimeError("Stage-0 train and held-out canonical IDs overlap.")
    return result


class TeacherTargetDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        input_size: int,
        teacher: dict[str, np.ndarray],
        teacher_runtime: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.base = teacher_runtime.UnlabeledDataset(self.frame, input_size)
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[int(index)]
        row = self.frame.iloc[int(index)]
        task = str(row["task_id"])
        position = int(row["task_position"])
        item["teacher_probability"] = torch.from_numpy(
            self.teacher[task][position].astype(np.float32)
        )
        return item


class DeterministicTaskBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        seed: int,
        epoch: int,
        shuffle: bool,
    ) -> None:
        batches: list[list[int]] = []
        rng = np.random.RandomState(int(seed) + int(epoch) * 1_000_003)
        tasks = frame["task_id"].astype(str).to_numpy()
        for task in sorted(set(tasks)):
            indices = np.flatnonzero(tasks == task)
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), int(batch_size)):
                batches.append(indices[start : start + int(batch_size)].tolist())
        if shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_target(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"A batch mixes tasks: {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "canonical_id": [str(item["canonical_id"]) for item in items],
        "teacher_probability": torch.stack(
            [item["teacher_probability"] for item in items]
        ),
    }


def loader_for(
    frame: pd.DataFrame,
    teacher: dict[str, np.ndarray],
    teacher_runtime: Any,
    settings: dict[str, Any],
    epoch: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TeacherTargetDataset(frame, int(settings["input_size"]), teacher, teacher_runtime)
    sampler = DeterministicTaskBatchSampler(
        frame,
        int(settings["batch_size"]),
        int(settings["seed"]),
        int(epoch),
        shuffle,
    )
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=collate_target,
    )


@torch.inference_mode()
def evaluate_student(
    network: FunctionalCompressionStudent,
    frame: pd.DataFrame,
    teacher: dict[str, np.ndarray],
    teacher_runtime: Any,
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    loader = loader_for(frame, teacher, teacher_runtime, settings, epoch=0, shuffle=False)
    task_values: dict[str, dict[str, list[float]]] = {}
    scale_zero_max = 0.0
    amp_dtype = torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    for batch in tqdm(loader, desc="WarmStartBase held-out representability", leave=False):
        task = str(batch["task_id"])
        images = batch["image"].to(device, non_blocking=True)
        target = batch["teacher_probability"].to(device, non_blocking=True).float()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(settings.get("amp", True)),
        ):
            output = network(images, task)
            fallback = network(images, task, residual_scale=0.0)
        student = spatial_probability(output["heatmap_logits"], settings["temperature"])
        anchor = spatial_probability(
            output["anchor_heatmap_logits"], settings["temperature"]
        )
        scale_zero_max = max(
            scale_zero_max,
            float(
                (fallback["heatmap_logits"] - output["anchor_heatmap_logits"])
                .abs()
                .max()
                .item()
            ),
        )
        target_coords = teacher_runtime.decode_probability(target, int(settings["decode_topk"]))
        anchor_coords = teacher_runtime.decode_probability(anchor, int(settings["decode_topk"]))
        student_coords = teacher_runtime.decode_probability(student, int(settings["decode_topk"]))
        values = task_values.setdefault(
            task,
            {
                "anchor_js": [],
                "student_js": [],
                "anchor_coord": [],
                "student_coord": [],
                "residual_abs": [],
            },
        )
        values["anchor_js"].extend(symmetric_js(anchor, target).mean(1).cpu().tolist())
        values["student_js"].extend(
            symmetric_js(student, target).mean(1).cpu().tolist()
        )
        values["anchor_coord"].extend(
            (
                torch.linalg.vector_norm(anchor_coords - target_coords, dim=-1).mean(1)
                * float(int(settings["input_size"]) - 1)
            )
            .cpu()
            .tolist()
        )
        values["student_coord"].extend(
            (
                torch.linalg.vector_norm(student_coords - target_coords, dim=-1).mean(1)
                * float(int(settings["input_size"]) - 1)
            )
            .cpu()
            .tolist()
        )
        values["residual_abs"].extend(
            output["residual_heatmap_logits"].abs().mean((1, 2, 3)).float().cpu().tolist()
        )

    rows = []
    for task in sorted(task_values):
        value = task_values[task]
        anchor_js = float(np.mean(value["anchor_js"]))
        student_js = float(np.mean(value["student_js"]))
        anchor_coord = float(np.mean(value["anchor_coord"]))
        student_coord = float(np.mean(value["student_coord"]))
        rows.append(
            {
                "task_id": task,
                "images": len(value["anchor_js"]),
                "anchor_js_to_teacher": anchor_js,
                "student_js_to_teacher": student_js,
                "js_ratio": student_js / max(anchor_js, 1e-12),
                "anchor_coordinate_distance_px": anchor_coord,
                "student_coordinate_distance_px": student_coord,
                "coordinate_ratio": student_coord / max(anchor_coord, 1e-12),
                "student_residual_abs_mean": float(np.mean(value["residual_abs"])),
            }
        )
    aggregate = {
        "task_macro_anchor_js": float(np.mean([row["anchor_js_to_teacher"] for row in rows])),
        "task_macro_student_js": float(np.mean([row["student_js_to_teacher"] for row in rows])),
        "task_macro_anchor_coordinate_distance_px": float(
            np.mean([row["anchor_coordinate_distance_px"] for row in rows])
        ),
        "task_macro_student_coordinate_distance_px": float(
            np.mean([row["student_coordinate_distance_px"] for row in rows])
        ),
    }
    aggregate["task_macro_js_ratio"] = aggregate["task_macro_student_js"] / max(
        aggregate["task_macro_anchor_js"], 1e-12
    )
    aggregate["task_macro_coordinate_ratio"] = aggregate[
        "task_macro_student_coordinate_distance_px"
    ] / max(aggregate["task_macro_anchor_coordinate_distance_px"], 1e-12)
    return aggregate, rows, scale_zero_max


def train_stage0(
    network: FunctionalCompressionStudent,
    train_frame: pd.DataFrame,
    teacher: dict[str, np.ndarray],
    teacher_runtime: Any,
    settings: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        network.trainable_parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    initial_lr = float(settings["learning_rate"])
    minimum_lr = float(settings["minimum_learning_rate"])
    amp_dtype = torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    history = []
    network.train()
    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / max(epochs - 1, 1)
        lr = minimum_lr + 0.5 * (initial_lr - minimum_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = loader_for(train_frame, teacher, teacher_runtime, settings, epoch, shuffle=True)
        totals = {"loss": 0.0, "kl": 0.0, "coordinate": 0.0, "l2": 0.0}
        steps = 0
        for batch in tqdm(loader, desc=f"WarmStartBase train epoch {epoch:02d}", leave=False):
            task = str(batch["task_id"])
            images = batch["image"].to(device, non_blocking=True)
            target = batch["teacher_probability"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(settings.get("amp", True)),
            ):
                output = network(images, task)
            logits = output["heatmap_logits"].float().flatten(2) / max(
                float(settings["temperature"]), 1e-6
            )
            log_probability = F.log_softmax(logits, dim=-1)
            target_flat = target.flatten(2).clamp_min(1e-12)
            kl = (
                target_flat
                * (target_flat.log() - log_probability)
            ).sum(-1).mean()
            student_probability = log_probability.exp().reshape_as(target)
            coordinate = F.smooth_l1_loss(
                probability_coordinates(student_probability),
                probability_coordinates(target),
            )
            l2 = output["residual_heatmap_logits"].float().square().mean()
            loss = (
                kl
                + float(settings["coordinate_loss_weight"]) * coordinate
                + float(settings["residual_l2_weight"]) * l2
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite WarmStartBase Stage-0 loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.trainable_parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["kl"] += float(kl.detach())
            totals["coordinate"] += float(coordinate.detach())
            totals["l2"] += float(l2.detach())
            steps += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "steps": steps,
                **{key: value / max(steps, 1) for key, value in totals.items()},
            }
        )
    return history


def run(config_path: Path) -> None:
    config_path = resolve_path(config_path)
    settings = json.loads(config_path.read_text())
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("WarmStartBase Stage 0 requires CUDA.")
    device = torch.device("cuda")
    run_dir = resolve_path(settings["run_dir"])
    if bool(settings.get("enforce_empty_output", True)) and run_dir.exists() and any(
        run_dir.iterdir()
    ):
        raise RuntimeError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_sources(run_dir, config_path)
    write_json(run_dir / "config.resolved.json", settings)
    write_json(run_dir / "environment.json", environment_record())

    teacher_runtime = load_module("teacher_runtime_for_warmstart_base", TEACHER_RUNTIME_SOURCE)
    dense_fusion = load_module("dense_fusion_for_warmstart_base", DENSE_FUSION_SOURCE)
    manifest_path = resolve_path(settings["unlabeled_manifest"])
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(settings["unlabeled_manifest_sha256"]):
        raise RuntimeError("WarmStartBase unlabeled manifest checksum mismatch.")
    manifest = pd.read_csv(manifest_path)
    selected = teacher_runtime.select_task_balanced_sample(
        manifest,
        [str(value) for value in settings["tasks"]],
        int(settings["samples_per_task"]),
        seed,
    )
    selected = make_split(selected, settings)
    selected.to_csv(run_dir / "stage0_unlabeled_split.csv", index=False)

    payloads, checkpoint_records = teacher_runtime.load_payloads(settings)
    model_outputs: list[dict[str, np.ndarray]] = []
    teacher_dataset = teacher_runtime.UnlabeledDataset(selected, int(settings["input_size"]))
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_sampler=teacher_runtime.HomogeneousTaskBatchSampler(
            selected, int(settings["teacher_batch_size"])
        ),
        num_workers=int(settings["num_workers"]),
        persistent_workers=int(settings["num_workers"]) > 0,
        pin_memory=True,
        collate_fn=teacher_runtime.collate_unlabeled,
    )
    amp_dtype = torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    medoid_index = None
    for index, (item, payload) in enumerate(zip(settings["checkpoints"], payloads)):
        model = teacher_runtime.build_model(dense_fusion, payload, payload["model_state"], device)
        outputs = teacher_runtime.evaluate_probabilities(
            model,
            teacher_loader,
            settings["task_scales"],
            device,
            bool(settings.get("amp", True)),
            amp_dtype,
            f"WarmStartBase teacher fold {item['fold']}",
        )
        model_outputs.append(outputs)
        if int(item["fold"]) == int(settings["medoid_fold"]):
            medoid_index = index
        del model
        torch.cuda.empty_cache()
    if medoid_index is None:
        raise RuntimeError("Configured medoid fold is absent from teacher checkpoints.")
    teacher = {
        task: np.mean(
            np.stack([output[task].astype(np.float32) for output in model_outputs], axis=0),
            axis=0,
        ).astype(np.float32)
        for task in settings["tasks"]
    }
    anchor_probabilities = model_outputs[int(medoid_index)]

    medoid_payload = payloads[int(medoid_index)]
    anchor = teacher_runtime.build_model(
        dense_fusion, medoid_payload, medoid_payload["model_state"], device
    )
    network = FunctionalCompressionStudent(
        anchor,
        medoid_payload["task_configs"],
        settings["task_scales"],
        shared_channels=int(
            medoid_payload["settings"].get("fusion_shared_channels", 128)
        ),
        hidden_channels=int(settings["residual_hidden_channels"]),
        logit_bound=float(settings["residual_logit_bound"]),
    ).to(device)
    anchor_hash_before = module_state_sha256(network.anchor)

    train_frame = selected[selected["stage0_split"] == "train"].reset_index(drop=True)
    heldout_frame = selected[selected["stage0_split"] == "heldout"].reset_index(drop=True)
    expected_train = int(settings["train_samples_per_task"]) * len(settings["tasks"])
    if len(train_frame) != expected_train or len(heldout_frame) != len(selected) - expected_train:
        raise RuntimeError("Unexpected Stage-0 train/held-out split sizes.")
    history = train_stage0(network, train_frame, teacher, teacher_runtime, settings, device)
    aggregate, task_rows, scale_zero_max = evaluate_student(
        network, heldout_frame, teacher, teacher_runtime, settings, device
    )
    anchor_hash_after = module_state_sha256(network.anchor)

    gates = {
        "js_ratio_pass": aggregate["task_macro_js_ratio"]
        <= float(settings["gate"]["maximum_js_ratio"]),
        "coordinate_ratio_pass": aggregate["task_macro_coordinate_ratio"]
        <= float(settings["gate"]["maximum_coordinate_ratio"]),
        "per_task_js_pass": max(row["js_ratio"] for row in task_rows)
        <= float(settings["gate"]["maximum_per_task_js_ratio"]),
        "frozen_anchor_hash_pass": anchor_hash_before == anchor_hash_after,
        "scale_zero_exact_pass": scale_zero_max == 0.0,
        "split_disjoint_pass": not bool(
            set(train_frame["sha256"].astype(str))
            & set(heldout_frame["sha256"].astype(str))
        ),
    }
    passed = all(bool(value) for value in gates.values())
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(task_rows).to_csv(run_dir / "heldout_task_metrics.csv", index=False)
    torch.save(
        {
            "residual_heads": {
                key: value.detach().cpu()
                for key, value in network.residual_heads.state_dict().items()
            },
            "settings": settings,
            "stage0_only": True,
        },
        run_dir / "stage0_residual_heads.pt",
    )
    summary = {
        "status": "pass" if passed else "fail",
        "decision": "authorize_full_bank" if passed else "stop_before_full_bank",
        "scope": "label-free teacher-function representability only",
        "official_or_labeled_ground_truth_read": False,
        "sample_images": int(len(selected)),
        "train_images": int(len(train_frame)),
        "heldout_images": int(len(heldout_frame)),
        "tasks": settings["tasks"],
        "teacher_models": checkpoint_records,
        "medoid_fold": int(settings["medoid_fold"]),
        "aggregate": aggregate,
        "task_metrics": task_rows,
        "gates": gates,
        "scale_zero_max_logit_difference": scale_zero_max,
        "anchor_hash_before": anchor_hash_before,
        "anchor_hash_after": anchor_hash_after,
        "manifest": {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": manifest_sha,
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in network.trainable_parameters()
        ),
    }
    write_json(run_dir / "metrics_summary.json", summary)
    write_json(
        run_dir / "qc_summary.json",
        {
            "status": "pass" if all(
                gates[key]
                for key in (
                    "frozen_anchor_hash_pass",
                    "scale_zero_exact_pass",
                    "split_disjoint_pass",
                )
            ) else "fail",
            "checks": gates,
        },
    )
    write_output_manifest(run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
