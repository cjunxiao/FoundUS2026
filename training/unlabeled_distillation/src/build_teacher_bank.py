from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import common


def allocate_banks(
    bank_dir: Path,
    frame,
    point_counts: dict[str, int],
    heatmap_size: int,
) -> tuple[dict[str, np.memmap], dict[str, np.memmap], dict[str, np.memmap]]:
    probability_sums = {}
    coordinate_sums = {}
    coordinate_square_sums = {}
    for task, points in point_counts.items():
        count = int((frame["task_id"].astype(str) == task).sum())
        paths = common.bank_paths(bank_dir, task)
        probability_sums[task] = np.lib.format.open_memmap(
            paths["sum"],
            mode="w+",
            dtype=np.float32,
            shape=(count, int(points), heatmap_size, heatmap_size),
        )
        coordinate_sums[task] = np.lib.format.open_memmap(
            paths["coordinate_sum"],
            mode="w+",
            dtype=np.float32,
            shape=(count, int(points), 2),
        )
        coordinate_square_sums[task] = np.lib.format.open_memmap(
            paths["coordinate_square_sum"],
            mode="w+",
            dtype=np.float32,
            shape=(count, int(points), 2),
        )
        probability_sums[task][:] = 0.0
        coordinate_sums[task][:] = 0.0
        coordinate_square_sums[task][:] = 0.0
    return probability_sums, coordinate_sums, coordinate_square_sums


@torch.inference_mode()
def accumulate_teacher(
    model: torch.nn.Module,
    loader,
    settings: dict[str, Any],
    device: torch.device,
    probability_sums: dict[str, np.memmap],
    coordinate_sums: dict[str, np.memmap],
    coordinate_square_sums: dict[str, np.memmap],
    description: str,
) -> dict[str, Any]:
    amp_dtype = (
        torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    )
    seen = {
        task: np.zeros(value.shape[0], dtype=np.uint8)
        for task, value in probability_sums.items()
    }
    duplicate_positions = 0
    for batch in tqdm(loader, desc=description):
        task = str(batch["task_id"])
        positions = batch["task_position"].numpy().astype(np.int64)
        duplicate_positions += int(seen[task][positions].sum())
        seen[task][positions] += 1
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(settings.get("amp", True)),
        ):
            output = model(images, task)
            logits = output["base_heatmap_logits"] + float(
                settings["task_scales"][task]
            ) * output["residual_heatmap_logits"]
        probability = common.spatial_probability(
            logits, float(settings["temperature"])
        )
        coordinate = common.topk_coordinates_px(
            probability, int(settings["decode_topk"])
        )
        probability_np = probability.cpu().numpy().astype(np.float32)
        coordinate_np = coordinate.cpu().numpy().astype(np.float32)
        probability_sums[task][positions] += probability_np
        coordinate_sums[task][positions] += coordinate_np
        coordinate_square_sums[task][positions] += np.square(coordinate_np)
    missing = {task: int((values == 0).sum()) for task, values in seen.items()}
    repeated = {task: int((values > 1).sum()) for task, values in seen.items()}
    return {
        "duplicate_positions_during_iteration": duplicate_positions,
        "missing_positions_by_task": missing,
        "repeated_positions_by_task": repeated,
        "exactly_once": duplicate_positions == 0
        and sum(missing.values()) == 0
        and sum(repeated.values()) == 0,
    }


def finalize_banks(
    bank_dir: Path,
    point_counts: dict[str, int],
    teacher_count: int,
    probability_sums: dict[str, np.memmap],
    coordinate_sums: dict[str, np.memmap],
    coordinate_square_sums: dict[str, np.memmap],
) -> dict[str, Any]:
    records = {}
    for task in point_counts:
        source = probability_sums[task]
        paths = common.bank_paths(bank_dir, task)
        mean = np.lib.format.open_memmap(
            paths["mean"], mode="w+", dtype=np.float16, shape=source.shape
        )
        dispersion = np.lib.format.open_memmap(
            paths["dispersion"],
            mode="w+",
            dtype=np.float16,
            shape=source.shape[:2],
        )
        coordinate_mean = coordinate_sums[task] / float(teacher_count)
        coordinate_variance = (
            coordinate_square_sums[task] / float(teacher_count)
            - np.square(coordinate_mean)
        ).clip(min=0.0)
        dispersion[:] = np.sqrt(coordinate_variance.sum(axis=-1)).astype(np.float16)
        for start in tqdm(
            range(0, source.shape[0], 64),
            desc=f"Finalize teacher bank {task}",
            leave=False,
        ):
            stop = min(start + 64, source.shape[0])
            value = source[start:stop] / float(teacher_count)
            value /= value.sum(axis=(-2, -1), keepdims=True).clip(min=1e-12)
            mean[start:stop] = value.astype(np.float16)
        mean.flush()
        dispersion.flush()
        records[task] = {
            "images": int(source.shape[0]),
            "landmarks": int(source.shape[1]),
            "teacher_mean_path": str(paths["mean"].relative_to(common.PROJECT_ROOT)),
            "coordinate_dispersion_path": str(
                paths["dispersion"].relative_to(common.PROJECT_ROOT)
            ),
            "teacher_mean_size_bytes": int(paths["mean"].stat().st_size),
            "coordinate_dispersion_size_bytes": int(
                paths["dispersion"].stat().st_size
            ),
            "dispersion_mean_px": float(np.asarray(dispersion, dtype=np.float32).mean()),
            "dispersion_p90_px": float(
                np.quantile(np.asarray(dispersion, dtype=np.float32), 0.90)
            ),
        }
        del mean, dispersion
    del probability_sums, coordinate_sums, coordinate_square_sums
    for task in point_counts:
        paths = common.bank_paths(bank_dir, task)
        for key in ("sum", "coordinate_sum", "coordinate_square_sum"):
            paths[key].unlink()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = common.resolve(args.config)
    settings = json.loads(config_path.read_text())
    random.seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("FinalDistillation teacher-bank generation requires CUDA.")
    common.enforce_space_limit(settings)

    run_dir = common.resolve(settings["bank_run_dir"])
    if bool(settings.get("enforce_empty_output", True)) and run_dir.exists() and any(
        run_dir.iterdir()
    ):
        raise RuntimeError(f"Teacher-bank run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = run_dir / "teacher_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    common.snapshot_sources(run_dir, config_path)
    common.write_json(run_dir / "config.resolved.json", settings)
    common.write_json(run_dir / "environment.json", common.environment_record())

    frame, coverage = common.prepare_manifest(settings)
    frame.to_csv(run_dir / "full_unlabeled_roles.csv", index=False)
    common.write_json(run_dir / "input_data_manifest.json", coverage)
    teacher_runtime, dense_fusion = common.load_dependencies()
    payloads, checkpoint_records = teacher_runtime.load_payloads(settings)
    point_counts = {
        str(item["task_id"]): int(item["num_classes"])
        for item in payloads[0]["task_configs"]
        if str(item["task_id"]) in settings["tasks"]
    }
    if set(point_counts) != set(settings["tasks"]):
        raise RuntimeError("Teacher task point map differs from FinalDistillation tasks.")

    probability_sums, coordinate_sums, coordinate_square_sums = allocate_banks(
        bank_dir, frame, point_counts, int(settings["heatmap_size"])
    )
    loader = common.indexed_loader(
        frame,
        int(settings["input_size"]),
        int(settings["teacher_batch_size"]),
        int(settings["num_workers"]),
        teacher_runtime,
    )
    device = torch.device("cuda")
    fold_coverage = []
    for item, payload in zip(settings["checkpoints"], payloads):
        model = teacher_runtime.build_model(dense_fusion, payload, payload["model_state"], device)
        audit = accumulate_teacher(
            model,
            loader,
            settings,
            device,
            probability_sums,
            coordinate_sums,
            coordinate_square_sums,
            f"FinalDistillation teacher fold {int(item['fold'])}",
        )
        audit["fold"] = int(item["fold"])
        fold_coverage.append(audit)
        for value in probability_sums.values():
            value.flush()
        for value in coordinate_sums.values():
            value.flush()
        for value in coordinate_square_sums.values():
            value.flush()
        del model
        torch.cuda.empty_cache()
        common.enforce_space_limit(settings)

    bank_records = finalize_banks(
        bank_dir,
        point_counts,
        len(payloads),
        probability_sums,
        coordinate_sums,
        coordinate_square_sums,
    )
    exact_coverage = all(bool(value["exactly_once"]) for value in fold_coverage)
    summary = {
        "status": "pass" if exact_coverage else "fail",
        "scope": "five-teacher soft heatmap bank over official challenge-unlabeled data",
        "official_or_labeled_ground_truth_read": False,
        "teacher_models": checkpoint_records,
        "teacher_count": int(len(payloads)),
        "coverage": coverage,
        "fold_coverage": fold_coverage,
        "bank": bank_records,
        "filesystem_used_bytes_after": common.filesystem_used_bytes(),
    }
    common.write_json(run_dir / "bank_summary.json", summary)
    common.write_json(
        run_dir / "qc_summary.json",
        {
            "status": summary["status"],
            "checks": {
                "manifest_exact_182870": coverage["rows"] == 182870
                and coverage["unique"] == 182870,
                "five_teacher_exact_coverage": exact_coverage,
                "train_audit_disjoint": coverage["train_audit_disjoint"],
                "no_labeled_ground_truth_read": True,
            },
        },
    )
    common.output_manifest(run_dir, hash_large_files=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
