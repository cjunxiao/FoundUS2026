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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
EXP191_SRC = PROJECT_ROOT / "2-code/191-usfm-convnext-dense-fusion/src"
RACE_SRC = PROJECT_ROOT / "2-code/11-e2e-roialign-cascade/src"
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
    exp191: Any,
    payload: dict[str, Any],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    model = exp191.build_model(payload["settings"], payload["task_configs"])
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


def candidate_metrics(
    candidate: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    originals: list[dict[str, np.ndarray]],
    tasks: list[str],
    topk: int,
    canvas_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for task in tasks:
        cand = torch.from_numpy(candidate[task].astype(np.float32))
        target = torch.from_numpy(teacher[task].astype(np.float32))
        cand_flat = cand.flatten(2)
        target_flat = target.flatten(2)
        js = symmetric_js(cand_flat, target_flat)
        cand_coords = decode_probability(cand, topk)
        teacher_coords = decode_probability(target, topk)
        coordinate_distance = torch.linalg.vector_norm(
            cand_coords - teacher_coords, dim=-1
        ) * float(canvas_size - 1)
        original_coords = torch.stack(
            [
                decode_probability(
                    torch.from_numpy(value[task].astype(np.float32)), topk
                )
                for value in originals
            ],
            dim=0,
        )
        dispersion = torch.linalg.vector_norm(
            original_coords - teacher_coords.unsqueeze(0), dim=-1
        ) * float(canvas_size - 1)
        entropy = -(
            target_flat.clamp_min(1e-12) * target_flat.clamp_min(1e-12).log()
        ).sum(-1) / math.log(target_flat.shape[-1])
        rows.append(
            {
                "task_id": task,
                "images": int(cand.shape[0]),
                "landmarks": int(cand.shape[1]),
                "js_to_teacher": float(js.mean()),
                "coordinate_distance_px_to_teacher": float(
                    coordinate_distance.mean()
                ),
                "teacher_fold_coordinate_dispersion_px": float(dispersion.mean()),
                "teacher_normalized_entropy": float(entropy.mean()),
            }
        )
    summary = {
        "task_macro_js_to_teacher": float(
            np.mean([row["js_to_teacher"] for row in rows])
        ),
        "task_macro_coordinate_distance_px_to_teacher": float(
            np.mean([row["coordinate_distance_px_to_teacher"] for row in rows])
        ),
        "task_macro_teacher_fold_coordinate_dispersion_px": float(
            np.mean([row["teacher_fold_coordinate_dispersion_px"] for row in rows])
        ),
        "task_macro_teacher_normalized_entropy": float(
            np.mean([row["teacher_normalized_entropy"] for row in rows])
        ),
    }
    return summary, rows


def snapshot_sources(run_dir: Path, config_path: Path) -> None:
    destination = run_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = set((EXPERIMENT_DIR / "src").glob("*.py"))
    sources.update(
        {
            config_path,
            EXPERIMENT_DIR / "README.md",
            EXPERIMENT_DIR / "COMMAND.md",
            PROJECT_ROOT
            / "1-docs/290-single-foundation-unlabeled-ensemble-compression.md",
            EXP191_SRC / "model.py",
            RACE_SRC / "foundus_race_lib.py",
            PROJECT_ROOT
            / "2-code/205-crossfold-residual-scale-calibration/src/calibrate_scales.py",
        }
    )
    records = []
    for source in sorted(value.resolve() for value in sources if value.exists()):
        relative = source.relative_to(PROJECT_ROOT)
        target = destination / "__".join(relative.parts)
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(relative),
                "snapshot": str(target.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    write_json(run_dir / "source_status.json", records)


def write_output_manifest(run_dir: Path) -> None:
    records = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "output_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(run_dir / "output_manifest.json", records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    config = json.loads(config_path.read_text())
    run_dir = resolve_path(config["run_dir"])
    if (
        bool(config.get("enforce_empty_output", True))
        and run_dir.exists()
        and any(run_dir.iterdir())
    ):
        raise RuntimeError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.resolved.json", config)
    write_json(
        run_dir / "command.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.executable,
        },
    )
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    snapshot_sources(run_dir, config_path)

    manifest_path = resolve_path(config["unlabeled_manifest"])
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(config["unlabeled_manifest_sha256"]):
        raise RuntimeError(f"Unlabeled manifest checksum mismatch: {manifest_sha}")
    manifest = pd.read_csv(manifest_path)
    tasks = [str(value) for value in config["tasks"]]
    selected = select_task_balanced_sample(
        manifest,
        tasks,
        int(config["samples_per_task"]),
        int(config["seed"]),
    )
    selected.to_csv(run_dir / "unlabeled_sample_manifest.csv", index=False)

    dataset = UnlabeledDataset(selected, int(config["input_size"]))
    batch_sampler = HomogeneousTaskBatchSampler(selected, int(config["batch_size"]))
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_unlabeled,
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
    )

    payloads, checkpoint_records = load_payloads(config)
    write_json(
        run_dir / "input_manifest.json",
        {
            "unlabeled_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "unlabeled_manifest_sha256": manifest_sha,
            "selected_images": len(selected),
            "selected_unique": int(selected["sha256"].astype(str).nunique()),
            "selected_by_task": {
                str(key): int(value)
                for key, value in selected.groupby("task_id").size().items()
            },
            "checkpoints": checkpoint_records,
        },
    )

    exp191 = load_module("exp191_model_for_exp290", EXP191_SRC / "model.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config.get("amp_dtype", "bfloat16"))]
    original_probabilities = []
    for record, payload in zip(checkpoint_records, payloads):
        model = build_model(exp191, payload, payload["model_state"], device)
        probabilities = evaluate_probabilities(
            model,
            loader,
            config["task_scales"],
            device,
            bool(config.get("amp", True)),
            amp_dtype,
            f"Fold {record['fold']}",
        )
        original_probabilities.append(probabilities)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    teacher = {
        task: np.mean(
            [value[task].astype(np.float32) for value in original_probabilities],
            axis=0,
        ).astype(np.float32)
        for task in tasks
    }

    all_rows = []
    original_summaries = {}
    original_task_rows = {}
    for record, probabilities in zip(checkpoint_records, original_probabilities):
        name = f"fold{record['fold']}_original"
        summary, rows = candidate_metrics(
            probabilities,
            teacher,
            original_probabilities,
            tasks,
            int(config["decode_topk"]),
            int(config["input_size"]),
        )
        original_summaries[name] = summary
        original_task_rows[name] = {row["task_id"]: row for row in rows}
        all_rows.extend({"candidate": name, **row} for row in rows)
    medoid_name = min(
        original_summaries,
        key=lambda name: original_summaries[name]["task_macro_js_to_teacher"],
    )
    medoid_index = int(medoid_name.removeprefix("fold").split("_")[0])

    averaged_state, averaging_audit = average_model_states(
        [payload["model_state"] for payload in payloads]
    )
    soup_probabilities = {}
    full_model = build_model(exp191, payloads[0], averaged_state, device)
    soup_probabilities["full_uniform_soup"] = evaluate_probabilities(
        full_model,
        loader,
        config["task_scales"],
        device,
        bool(config.get("amp", True)),
        amp_dtype,
        "Full soup",
    )
    del full_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    medoid_state = payloads[medoid_index]["model_state"]
    fusion_state = {
        key: (
            averaged_state[key]
            if not key.startswith("encoder.")
            else value
        )
        for key, value in medoid_state.items()
    }
    fusion_model = build_model(
        exp191, payloads[medoid_index], fusion_state, device
    )
    soup_probabilities["fusion_uniform_soup"] = evaluate_probabilities(
        fusion_model,
        loader,
        config["task_scales"],
        device,
        bool(config.get("amp", True)),
        amp_dtype,
        "Fusion soup",
    )
    del fusion_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    soup_summaries = {}
    soup_task_rows = {}
    for name, probabilities in soup_probabilities.items():
        summary, rows = candidate_metrics(
            probabilities,
            teacher,
            original_probabilities,
            tasks,
            int(config["decode_topk"]),
            int(config["input_size"]),
        )
        soup_summaries[name] = summary
        soup_task_rows[name] = {row["task_id"]: row for row in rows}
        all_rows.extend({"candidate": name, **row} for row in rows)

    primary_soup = min(
        soup_summaries,
        key=lambda name: soup_summaries[name]["task_macro_js_to_teacher"],
    )
    medoid = original_summaries[medoid_name]
    candidate = soup_summaries[primary_soup]
    medoid_tasks = original_task_rows[medoid_name]
    candidate_tasks = soup_task_rows[primary_soup]
    gate = config["gate"]
    task_js_ratios = {
        task: candidate_tasks[task]["js_to_teacher"]
        / max(medoid_tasks[task]["js_to_teacher"], 1e-12)
        for task in tasks
    }
    checks = {
        "js_ratio": candidate["task_macro_js_to_teacher"]
        / max(medoid["task_macro_js_to_teacher"], 1e-12)
        <= float(gate["maximum_js_ratio_to_medoid"]),
        "coordinate_ratio": candidate[
            "task_macro_coordinate_distance_px_to_teacher"
        ]
        / max(medoid["task_macro_coordinate_distance_px_to_teacher"], 1e-12)
        <= float(gate["maximum_coordinate_ratio_to_medoid"]),
        "per_task_js_safety": max(task_js_ratios.values())
        <= float(gate["maximum_per_task_js_ratio_to_medoid"]),
        "teacher_diversity": candidate[
            "task_macro_teacher_fold_coordinate_dispersion_px"
        ]
        >= float(gate["minimum_teacher_dispersion_px"]),
        "finite": all(
            math.isfinite(float(value))
            for summary in [*original_summaries.values(), *soup_summaries.values()]
            for value in summary.values()
        ),
    }
    passed = all(checks.values())
    summary = {
        "status": "pass" if passed else "stop",
        "stage": "unlabeled_weight_soup_representability",
        "teacher_models": len(payloads),
        "unlabeled_images": len(selected),
        "unlabeled_images_per_task": int(config["samples_per_task"]),
        "medoid": medoid_name,
        "primary_soup": primary_soup,
        "originals": original_summaries,
        "soups": soup_summaries,
        "medoid_summary": medoid,
        "primary_soup_summary": candidate,
        "primary_vs_medoid": {
            "js_ratio": candidate["task_macro_js_to_teacher"]
            / max(medoid["task_macro_js_to_teacher"], 1e-12),
            "coordinate_ratio": candidate[
                "task_macro_coordinate_distance_px_to_teacher"
            ]
            / max(
                medoid["task_macro_coordinate_distance_px_to_teacher"], 1e-12
            ),
            "task_js_ratios": task_js_ratios,
        },
        "gate": gate,
        "checks": checks,
        "advance_to_training": passed,
        "decision": (
            "Authorize a separately preregistered full-data B0/B1 compression run."
            if passed
            else "Stop cross-fold weight-soup compression before training."
        ),
    }
    pd.DataFrame(all_rows).to_csv(run_dir / "candidate_metrics_by_task.csv", index=False)
    write_json(run_dir / "state_averaging_audit.json", averaging_audit)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "qc_summary.json",
        {
            "status": "pass",
            "manifest_checksum": True,
            "checkpoint_checksums": True,
            "task_balanced": all(
                value == int(config["samples_per_task"])
                for value in selected.groupby("task_id").size().tolist()
            ),
            "selected_unique": int(selected["sha256"].astype(str).nunique())
            == len(selected),
            "strict_state_load": True,
            "nonfloating_state_compatible": True,
            "no_validation_labels_read": True,
        },
    )
    write_output_manifest(run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

