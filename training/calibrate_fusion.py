from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_PAIRS: dict[str, list[tuple[int, int]]] = {
    "A4C": list(zip(range(0, 16, 2), range(1, 16, 2))),
    "FUGC": [(0, 1)],
    "IVC": [(0, 1)],
    "PLAX": list(zip(range(0, 22, 2), range(1, 22, 2))),
    "PSAX": [(0, 1), (2, 3)],
    "fetal_femur": [(0, 1)],
}
OFFICIAL_SORT_TASKS = {"A4C", "PSAX"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fold DenseFusion residual-scale calibration.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def interpolate_logits(anchor: np.ndarray, fusion: np.ndarray, scale: float) -> np.ndarray:
    if anchor.shape != fusion.shape:
        raise ValueError(f"Logit shape mismatch: {anchor.shape} != {fusion.shape}")
    return anchor.astype(np.float32) + float(scale) * (
        fusion.astype(np.float32) - anchor.astype(np.float32)
    )


def decode_topk(logits: np.ndarray, topk: int, beta: float) -> np.ndarray:
    batch, points, height, width = logits.shape
    flat = logits.reshape(batch, points, -1).astype(np.float32)
    count = min(max(int(topk), 1), flat.shape[-1])
    indices = np.argpartition(flat, -count, axis=-1)[..., -count:]
    values = np.take_along_axis(flat, indices, axis=-1) * float(beta)
    values -= values.max(axis=-1, keepdims=True)
    weights = np.exp(values)
    weights /= weights.sum(axis=-1, keepdims=True).clip(min=1e-12)
    xs = (indices % width).astype(np.float32) / float(max(width - 1, 1))
    ys = (indices // width).astype(np.float32) / float(max(height - 1, 1))
    return np.stack(((weights * xs).sum(-1), (weights * ys).sum(-1)), axis=-1)


def sort_official_vertical(points: np.ndarray, task_id: str) -> np.ndarray:
    output = points.copy()
    if str(task_id) not in OFFICIAL_SORT_TASKS:
        return output
    for first in range(0, output.shape[1], 2):
        second = first + 1
        swap = output[:, first, 1] > output[:, second, 1]
        first_value = output[:, first].copy()
        second_value = output[:, second].copy()
        output[swap, first] = second_value[swap]
        output[swap, second] = first_value[swap]
    return output


def canvas_to_original(
    points_norm: np.ndarray, metadata: list[dict[str, Any]], input_size: int
) -> np.ndarray:
    output = points_norm.astype(np.float32).copy() * float(input_size - 1)
    for index, item in enumerate(metadata):
        letterbox = item["letterbox"]
        scale = max(float(letterbox["scale"]), 1e-8)
        output[index, :, 0] = (output[index, :, 0] - float(letterbox["pad_x"])) / scale
        output[index, :, 1] = (output[index, :, 1] - float(letterbox["pad_y"])) / scale
        output[index, :, 0] = np.clip(
            output[index, :, 0], 0.0, float(letterbox["original_w"]) - 1.0
        )
        output[index, :, 1] = np.clip(
            output[index, :, 1], 0.0, float(letterbox["original_h"]) - 1.0
        )
    return output


def _length(points: torch.Tensor, first: int, second: int) -> torch.Tensor:
    return torch.linalg.norm(points[:, first] - points[:, second], dim=-1).clamp_min(1e-6)


def _ellipse_circumference(points: torch.Tensor) -> torch.Tensor:
    semi_a = 0.5 * _length(points, 0, 1)
    semi_b = 0.5 * _length(points, 2, 3)
    h = ((semi_a - semi_b) / (semi_a + semi_b).clamp_min(1e-6)).square()
    return torch.pi * (semi_a + semi_b) * (
        1.0 + 3.0 * h / (10.0 + torch.sqrt((4.0 - 3.0 * h).clamp_min(1e-6)))
    )


def measurement_values(points: np.ndarray, task_id: str) -> np.ndarray:
    value = torch.from_numpy(points.astype(np.float32))
    task = str(task_id)
    if task == "FA":
        result = _ellipse_circumference(value).unsqueeze(-1)
    elif task == "HC":
        result = torch.stack((_length(value, 0, 1), _ellipse_circumference(value)), dim=-1)
    elif task == "AOP":
        first = F.normalize(value[:, 1] - value[:, 0], dim=-1, eps=1e-6)
        second = F.normalize(value[:, 3] - value[:, 0], dim=-1, eps=1e-6)
        angle = torch.rad2deg(torch.acos((first * second).sum(-1).clamp(-1.0, 1.0)))
        result = torch.stack((angle, _length(value, 0, 2)), dim=-1)
    else:
        result = torch.stack(
            [_length(value, first, second) for first, second in TASK_PAIRS[task]], dim=-1
        )
    return result.numpy()


def _load_cache(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    payload = np.load(path, allow_pickle=False)
    logits = payload["logits"]
    metadata = [json.loads(str(value)) for value in payload["metadata_json"]]
    if len(logits) != len(metadata):
        raise RuntimeError(f"Cache length mismatch: {path}")
    return logits, metadata


def _aligned_caches(
    anchor_path: Path, fusion_path: Path
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    anchor_logits, anchor_metadata = _load_cache(anchor_path)
    fusion_logits, fusion_metadata = _load_cache(fusion_path)
    anchor_index = {str(item["source_image_path"]): index for index, item in enumerate(anchor_metadata)}
    fusion_sources = [str(item["source_image_path"]) for item in fusion_metadata]
    if len(anchor_index) != len(anchor_metadata) or len(set(fusion_sources)) != len(fusion_sources):
        raise RuntimeError("Duplicate source_image_path in OOF cache.")
    if set(anchor_index) != set(fusion_sources):
        raise RuntimeError(f"OOF source mismatch: {anchor_path} vs {fusion_path}")
    order = [anchor_index[source] for source in fusion_sources]
    aligned_anchor = anchor_logits[np.asarray(order)]
    aligned_metadata = [anchor_metadata[index] for index in order]
    for anchor_item, fusion_item in zip(aligned_metadata, fusion_metadata):
        for key in ("internal_target_points", "official_target_points", "letterbox"):
            if anchor_item[key] != fusion_item[key]:
                raise RuntimeError(f"Metadata mismatch for {fusion_item['source_image_path']}: {key}")
    return aligned_anchor, fusion_logits, fusion_metadata


def evaluate_task(
    fold: int,
    task: str,
    scale: float,
    anchor_logits: np.ndarray,
    fusion_logits: np.ndarray,
    metadata: list[dict[str, Any]],
    input_size: int,
    topk: int,
    beta: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    logits = interpolate_logits(anchor_logits, fusion_logits, scale)
    internal_norm = decode_topk(logits, topk, beta)
    official_norm = sort_official_vertical(internal_norm, task)
    internal_prediction = canvas_to_original(internal_norm, metadata, input_size)
    official_prediction = canvas_to_original(official_norm, metadata, input_size)
    internal_target = np.asarray(
        [item["internal_target_points"] for item in metadata], dtype=np.float32
    )
    official_target = np.asarray(
        [item["official_target_points"] for item in metadata], dtype=np.float32
    )
    internal_error = np.linalg.norm(internal_prediction - internal_target, axis=-1)
    official_error = np.linalg.norm(official_prediction - official_target, axis=-1)
    internal_measurement_error = np.abs(
        measurement_values(internal_prediction, task) - measurement_values(internal_target, task)
    ).mean(axis=-1)
    official_measurement_error = np.abs(
        measurement_values(official_prediction, task) - measurement_values(official_target, task)
    ).mean(axis=-1)
    mre = float(internal_error.mean(axis=-1).mean())
    mae = float(internal_measurement_error.mean())
    official_mre = float(official_error.mean(axis=-1).mean())
    official_mae = float(official_measurement_error.mean())
    row = {
        "fold": int(fold),
        "task_id": str(task),
        "scale": float(scale),
        "num_images": int(len(metadata)),
        "mre_original_px": mre,
        "measurement_proxy_mae": mae,
        "final_proxy_score": 0.5 * (mre + mae),
        "point_error_p90": float(np.quantile(internal_error, 0.90)),
        "point_error_p95": float(np.quantile(internal_error, 0.95)),
        "point_error_gt40_rate": float(np.mean(internal_error > 40.0)),
        "official_mre_original_px": official_mre,
        "official_measurement_proxy_mae": official_mae,
        "official_final_proxy_score": 0.5 * (official_mre + official_mae),
        "official_point_error_p90": float(np.quantile(official_error, 0.90)),
        "official_point_error_p95": float(np.quantile(official_error, 0.95)),
        "official_point_error_gt40_rate": float(np.mean(official_error > 40.0)),
    }
    image_rows = pd.DataFrame(
        {
            "fold": int(fold),
            "task_id": str(task),
            "scale": float(scale),
            "source_image_path": [str(item["source_image_path"]) for item in metadata],
            "mre_original_px": internal_error.mean(axis=-1),
            "measurement_proxy_mae": internal_measurement_error,
            "official_mre_original_px": official_error.mean(axis=-1),
            "official_measurement_proxy_mae": official_measurement_error,
        }
    )
    point_rows = pd.DataFrame(
        {
            "fold": int(fold),
            "task_id": str(task),
            "scale": float(scale),
            "error_px": internal_error.reshape(-1),
            "official_error_px": official_error.reshape(-1),
        }
    )
    return row, image_rows, point_rows


def select_scales(
    task_metrics: pd.DataFrame,
    development_folds: list[int],
    scales: list[float],
    metric: str,
) -> pd.DataFrame:
    rows = []
    for task, frame in task_metrics[task_metrics["fold"].isin(development_folds)].groupby("task_id"):
        baseline = frame[frame["scale"] == 0.0].set_index("fold")[metric]
        candidates = []
        for scale in scales:
            values = frame[frame["scale"] == float(scale)].set_index("fold")[metric]
            delta = values - baseline
            if float(scale) == 0.0 or bool((delta < 0.0).all()):
                candidates.append((float(delta.mean()), float(scale), delta))
        candidates.sort(key=lambda item: (item[0], item[1]))
        mean_delta, selected, deltas = candidates[0]
        row: dict[str, Any] = {
            "task_id": str(task),
            "selected_scale": selected,
            "mean_development_delta": mean_delta,
            "improves_in_every_development_fold": bool(selected == 0.0 or (deltas < 0).all()),
        }
        for fold in development_folds:
            row[f"fold{fold}_delta"] = float(deltas.loc[fold])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def summarize_selected(
    selected_task_metrics: pd.DataFrame, selected_point_rows: pd.DataFrame
) -> dict[str, float]:
    internal = selected_point_rows["error_px"].to_numpy(np.float64)
    official = selected_point_rows["official_error_px"].to_numpy(np.float64)
    return {
        "task_macro_mre_original_px": float(selected_task_metrics["mre_original_px"].mean()),
        "task_macro_parameter_mae_proxy": float(
            selected_task_metrics["measurement_proxy_mae"].mean()
        ),
        "final_proxy_score": float(selected_task_metrics["final_proxy_score"].mean()),
        "point_error_p90": float(np.quantile(internal, 0.90)),
        "point_error_p95": float(np.quantile(internal, 0.95)),
        "point_error_gt40_rate": float(np.mean(internal > 40.0)),
        "official_task_macro_mre_original_px": float(
            selected_task_metrics["official_mre_original_px"].mean()
        ),
        "official_task_macro_parameter_mae_proxy": float(
            selected_task_metrics["official_measurement_proxy_mae"].mean()
        ),
        "official_final_proxy_score": float(
            selected_task_metrics["official_final_proxy_score"].mean()
        ),
        "official_point_error_p90": float(np.quantile(official, 0.90)),
        "official_point_error_p95": float(np.quantile(official, 0.95)),
        "official_point_error_gt40_rate": float(np.mean(official > 40.0)),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if settings["selection_rule"] != "improves_in_every_development_fold_then_best_mean":
        raise RuntimeError("Unexpected selection rule.")
    scales = sorted({float(value) for value in settings["scales"]})
    if scales != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError(f"FusedTeacher fixed scale grid changed: {scales}")
    development_folds = sorted(map(int, settings["development_folds"]))
    confirmation_folds = sorted(map(int, settings["confirmation_folds"]))
    if development_folds != [0, 1] or confirmation_folds != [2, 3, 4]:
        raise RuntimeError("FusedTeacher requires development folds 0-1 and confirmation folds 2-4.")
    run_dir = resolve(settings["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    image_frames: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    expected_tasks: set[str] | None = None
    for fold_entry in settings["fold_runs"]:
        fold = int(fold_entry["fold"])
        anchor_dir = resolve(fold_entry["anchor_run"]) / "oof_heatmaps_selected_best_final_proxy"
        fusion_dir = resolve(fold_entry["fusion_run"]) / "oof_heatmaps_selected_best_final_proxy"
        anchor_tasks = {path.stem for path in anchor_dir.glob("*.npz")}
        fusion_tasks = {path.stem for path in fusion_dir.glob("*.npz")}
        if anchor_tasks != fusion_tasks:
            raise RuntimeError(f"Fold {fold} task cache mismatch.")
        if expected_tasks is None:
            expected_tasks = anchor_tasks
        elif expected_tasks != anchor_tasks:
            raise RuntimeError(f"Fold {fold} task set differs.")
        for task in sorted(anchor_tasks):
            anchor_path = anchor_dir / f"{task}.npz"
            fusion_path = fusion_dir / f"{task}.npz"
            anchor_logits, fusion_logits, metadata = _aligned_caches(anchor_path, fusion_path)
            for scale in scales:
                row, images, points = evaluate_task(
                    fold,
                    task,
                    scale,
                    anchor_logits,
                    fusion_logits,
                    metadata,
                    int(settings["input_size"]),
                    int(settings["decode_topk"]),
                    float(settings["decode_beta"]),
                )
                metric_rows.append(row)
                image_frames.append(images)
                point_frames.append(points)
            for path in (anchor_path, fusion_path):
                inputs.append(
                    {
                        "fold": fold,
                        "task_id": task,
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )

    task_metrics = pd.DataFrame(metric_rows)
    per_image = pd.concat(image_frames, ignore_index=True)
    per_point = pd.concat(point_frames, ignore_index=True)
    task_metrics.to_csv(run_dir / "candidate_metrics_by_fold_task.csv", index=False)
    per_image.to_csv(run_dir / "candidate_per_image.csv.gz", index=False, compression="gzip")
    selections = select_scales(
        task_metrics,
        development_folds,
        scales,
        str(settings["selection_metric"]),
    )
    selections.to_csv(run_dir / "locked_task_scales.csv", index=False)
    scale_map = dict(zip(selections["task_id"], selections["selected_scale"]))

    comparisons: dict[str, Any] = {}
    selected_task_frames = []
    selected_point_frames = []
    for fold in confirmation_folds:
        selected_rows = pd.concat(
            [
                task_metrics[
                    (task_metrics["fold"] == fold)
                    & (task_metrics["task_id"] == task)
                    & (task_metrics["scale"] == float(scale))
                ]
                for task, scale in sorted(scale_map.items())
            ],
            ignore_index=True,
        )
        anchor_rows = task_metrics[
            (task_metrics["fold"] == fold) & (task_metrics["scale"] == 0.0)
        ].sort_values("task_id")
        selected_points = pd.concat(
            [
                per_point[
                    (per_point["fold"] == fold)
                    & (per_point["task_id"] == task)
                    & (per_point["scale"] == float(scale))
                ]
                for task, scale in sorted(scale_map.items())
            ],
            ignore_index=True,
        )
        anchor_points = per_point[
            (per_point["fold"] == fold) & (per_point["scale"] == 0.0)
        ]
        selected_summary = summarize_selected(selected_rows, selected_points)
        anchor_summary = summarize_selected(anchor_rows, anchor_points)
        comparisons[str(fold)] = {
            key: {
                "anchor": float(anchor_summary[key]),
                "calibrated": float(selected_summary[key]),
                "delta": float(selected_summary[key] - anchor_summary[key]),
            }
            for key in selected_summary
        }
        selected_rows.assign(selected_scale=selected_rows["scale"]).to_csv(
            run_dir / f"fold{fold}_selected_per_task.csv", index=False
        )
        selected_task_frames.append(selected_rows)
        selected_point_frames.append(selected_points)

    pooled_tasks = pd.concat(selected_task_frames, ignore_index=True)
    mean_task = (
        pooled_tasks.groupby("task_id", sort=True)
        .agg(
            mre_original_px=("mre_original_px", "mean"),
            measurement_proxy_mae=("measurement_proxy_mae", "mean"),
            final_proxy_score=("final_proxy_score", "mean"),
            official_mre_original_px=("official_mre_original_px", "mean"),
            official_measurement_proxy_mae=("official_measurement_proxy_mae", "mean"),
            official_final_proxy_score=("official_final_proxy_score", "mean"),
        )
        .reset_index()
    )
    pooled_points = pd.concat(selected_point_frames, ignore_index=True)
    confirmation_summary = summarize_selected(mean_task, pooled_points)
    anchor_confirmation = task_metrics[
        task_metrics["fold"].isin(confirmation_folds) & (task_metrics["scale"] == 0.0)
    ]
    anchor_mean_task = (
        anchor_confirmation.groupby("task_id", sort=True)
        .agg(
            mre_original_px=("mre_original_px", "mean"),
            measurement_proxy_mae=("measurement_proxy_mae", "mean"),
            final_proxy_score=("final_proxy_score", "mean"),
            official_mre_original_px=("official_mre_original_px", "mean"),
            official_measurement_proxy_mae=("official_measurement_proxy_mae", "mean"),
            official_final_proxy_score=("official_final_proxy_score", "mean"),
        )
        .reset_index()
    )
    anchor_points = per_point[
        per_point["fold"].isin(confirmation_folds) & (per_point["scale"] == 0.0)
    ]
    anchor_summary = summarize_selected(anchor_mean_task, anchor_points)
    summary = {
        "status": "complete",
        "protocol": "folds0-1 locked task residual scales; one-pass confirmation on folds2-4",
        "development_folds": development_folds,
        "confirmation_folds": confirmation_folds,
        "locked_task_scales": {task: float(scale) for task, scale in sorted(scale_map.items())},
        "confirmation": confirmation_summary,
        "confirmation_anchor": anchor_summary,
        "confirmation_delta": {
            key: float(confirmation_summary[key] - anchor_summary[key])
            for key in confirmation_summary
        },
        "per_fold_comparison": comparisons,
        "target_fold_grid_search": False,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.resolved.json", settings)
    write_json(run_dir / "input_manifest.json", inputs)
    write_json(
        run_dir / "command.json",
        {"argv": sys.argv, "cwd": str(Path.cwd()), "config": str(config_path.relative_to(PROJECT_ROOT))},
    )
    output_paths = sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path.name != "output_manifest.json"
    )
    write_json(
        run_dir / "output_manifest.json",
        [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
