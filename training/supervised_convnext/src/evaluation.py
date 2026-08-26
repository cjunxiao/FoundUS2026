from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import model


TASK_PAIRS: dict[str, list[tuple[int, int]]] = {
    "A4C": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
    "FUGC": [(0, 1)],
    "IVC": [(0, 1)],
    "PLAX": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)],
    "PSAX": [(0, 1), (2, 3)],
    "fetal_femur": [(0, 1)],
}

MEASUREMENT_NAMES: dict[str, list[str]] = {
    "A4C": [
        "LV_vertical", "LV_horizontal", "RV_vertical", "RV_horizontal",
        "LA_vertical", "LA_horizontal", "RA_vertical", "RA_horizontal",
    ],
    "AOP": ["AOP_angle", "HSD"],
    "FA": ["FA_circumference"],
    "FUGC": ["CL"],
    "HC": ["BPD", "HC_circumference"],
    "IVC": ["IVC"],
    "PLAX": ["LV", "RV", "IVS", "LVPW", "VAO", "STJ", "AAO", "AV", "LVOT", "LA", "RVOT"],
    "PSAX": ["RVOT", "PA"],
    "fetal_femur": ["FL"],
}


def _length(points: torch.Tensor, first: int, second: int) -> torch.Tensor:
    return torch.linalg.norm(points[:, first] - points[:, second], dim=-1).clamp_min(1e-6)


def ellipse_circumference(points: torch.Tensor) -> torch.Tensor:
    axis_a = _length(points, 0, 1)
    axis_b = _length(points, 2, 3)
    semi_a, semi_b = 0.5 * axis_a, 0.5 * axis_b
    h = ((semi_a - semi_b) / (semi_a + semi_b).clamp_min(1e-6)).square()
    return torch.pi * (semi_a + semi_b) * (
        1.0 + 3.0 * h / (10.0 + torch.sqrt((4.0 - 3.0 * h).clamp_min(1e-6)))
    )


def measurement_values(points: torch.Tensor, task_id: str) -> torch.Tensor:
    task = str(task_id)
    if task == "FA":
        return ellipse_circumference(points).unsqueeze(-1)
    if task == "HC":
        return torch.stack([_length(points, 0, 1), ellipse_circumference(points)], dim=-1)
    if task == "AOP":
        first = F.normalize(points[:, 1] - points[:, 0], dim=-1, eps=1e-6)
        second = F.normalize(points[:, 3] - points[:, 0], dim=-1, eps=1e-6)
        angle = torch.rad2deg(torch.acos((first * second).sum(-1).clamp(-1.0, 1.0)))
        return torch.stack([angle, _length(points, 0, 2)], dim=-1)
    return torch.stack(
        [_length(points, first, second) for first, second in TASK_PAIRS[task]], dim=-1
    )


def canvas_to_original(
    points_norm: torch.Tensor,
    metas: list[dict[str, float]],
    input_size: int,
) -> torch.Tensor:
    points = points_norm.clone() * float(input_size - 1)
    output = []
    for index, meta in enumerate(metas):
        value = points[index].clone()
        value[:, 0] = (value[:, 0] - float(meta["pad_x"])) / max(float(meta["scale"]), 1e-8)
        value[:, 1] = (value[:, 1] - float(meta["pad_y"])) / max(float(meta["scale"]), 1e-8)
        value[:, 0].clamp_(0.0, float(meta["original_w"]) - 1.0)
        value[:, 1].clamp_(0.0, float(meta["original_h"]) - 1.0)
        output.append(value)
    return torch.stack(output)


def _tail_metrics(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}point_error_p50": float(np.quantile(values, 0.50)),
        f"{prefix}point_error_p90": float(np.quantile(values, 0.90)),
        f"{prefix}point_error_p95": float(np.quantile(values, 0.95)),
        f"{prefix}point_error_gt20_rate": float(np.mean(values > 20.0)),
        f"{prefix}point_error_gt40_rate": float(np.mean(values > 40.0)),
    }


def _internal_target(
    official_target: torch.Tensor,
    task_id: str,
    settings: dict[str, Any],
) -> torch.Tensor:
    tasks = {str(value) for value in settings.get("internal_identity_tasks", [])}
    if str(task_id) not in tasks:
        return official_target
    return model.canonicalize_internal_points(official_target, task_id)


@torch.no_grad()
def evaluate(
    network: torch.nn.Module,
    loader: DataLoader,
    settings: dict[str, Any],
    device: torch.device,
    cache_logits_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    network.eval()
    image_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    logits_cache: dict[str, list[np.ndarray]] = defaultdict(list)
    cache_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for batch in tqdm(loader, desc="[Validation internal+official]", leave=False):
        task = str(batch["task_id"])
        images = batch["image"].to(device, non_blocking=True)
        output = model.forward_inference(network, images, task, settings)
        internal_norm = output["internal_coords_norm"].float().clamp(0.0, 1.0)
        official_norm = output["coords_norm"].float().clamp(0.0, 1.0)
        internal_prediction = canvas_to_original(
            internal_norm, batch["letterbox"], int(settings["input_size"])
        )
        official_prediction = canvas_to_original(
            official_norm, batch["letterbox"], int(settings["input_size"])
        )
        official_target = batch["points_original"].to(device).float()
        internal_target = _internal_target(official_target, task, settings)

        internal_errors = torch.linalg.norm(internal_prediction - internal_target, dim=-1)
        official_errors = torch.linalg.norm(official_prediction - official_target, dim=-1)
        internal_measurements = measurement_values(internal_prediction, task)
        official_measurements = measurement_values(official_prediction, task)
        internal_target_measurements = measurement_values(internal_target, task)
        official_target_measurements = measurement_values(official_target, task)
        internal_measurement_errors = (
            internal_measurements - internal_target_measurements
        ).abs()
        official_measurement_errors = (
            official_measurements - official_target_measurements
        ).abs()

        if cache_logits_dir is not None:
            logits_cache[task].append(
                output["heatmap_logits"].detach().float().cpu().numpy().astype(np.float16)
            )

        for image_index, source in enumerate(batch["source_image_path"]):
            internal_pred_list = internal_prediction[image_index].cpu().tolist()
            official_pred_list = official_prediction[image_index].cpu().tolist()
            internal_target_list = internal_target[image_index].cpu().tolist()
            official_target_list = official_target[image_index].cpu().tolist()
            changed = not torch.equal(
                internal_prediction[image_index], official_prediction[image_index]
            )
            image_rows.append(
                {
                    "task_id": task,
                    "source_image_path": str(source),
                    "group_id": str(batch["group_id"][image_index]),
                    "num_points": int(internal_errors[image_index].numel()),
                    "mre_original_px": float(internal_errors[image_index].mean().cpu()),
                    "measurement_proxy_mae": float(
                        internal_measurement_errors[image_index].mean().cpu()
                    ),
                    "official_mre_original_px": float(
                        official_errors[image_index].mean().cpu()
                    ),
                    "official_measurement_proxy_mae": float(
                        official_measurement_errors[image_index].mean().cpu()
                    ),
                    "official_sort_changed": bool(changed),
                    "predicted_points_json": json.dumps(
                        internal_pred_list, separators=(",", ":")
                    ),
                    "target_points_json": json.dumps(
                        internal_target_list, separators=(",", ":")
                    ),
                    "official_predicted_points_json": json.dumps(
                        official_pred_list, separators=(",", ":")
                    ),
                    "official_target_points_json": json.dumps(
                        official_target_list, separators=(",", ":")
                    ),
                }
            )
            for point_index, (internal_error, official_error) in enumerate(
                zip(
                    internal_errors[image_index].cpu().tolist(),
                    official_errors[image_index].cpu().tolist(),
                )
            ):
                point_rows.append(
                    {
                        "task_id": task,
                        "source_image_path": str(source),
                        "point_index": int(point_index),
                        "error_px": float(internal_error),
                        "official_error_px": float(official_error),
                    }
                )
            for name, internal_error, official_error in zip(
                MEASUREMENT_NAMES[task],
                internal_measurement_errors[image_index].cpu().tolist(),
                official_measurement_errors[image_index].cpu().tolist(),
            ):
                measurement_rows.append(
                    {
                        "task_id": task,
                        "source_image_path": str(source),
                        "measurement_name": str(name),
                        "absolute_error": float(internal_error),
                        "official_absolute_error": float(official_error),
                    }
                )
            if cache_logits_dir is not None:
                cache_metadata[task].append(
                    {
                        "source_image_path": str(source),
                        "group_id": str(batch["group_id"][image_index]),
                        "letterbox": {
                            key: float(value)
                            for key, value in batch["letterbox"][image_index].items()
                        },
                        "internal_target_points": internal_target_list,
                        "official_target_points": official_target_list,
                    }
                )

    per_image = pd.DataFrame(image_rows)
    per_point = pd.DataFrame(point_rows)
    per_measurement_image = pd.DataFrame(measurement_rows)
    task_rows: list[dict[str, Any]] = []
    for task, task_images in per_image.groupby("task_id", sort=True):
        task_points = per_point[per_point["task_id"] == task]
        internal_values = task_points["error_px"].to_numpy(dtype=np.float64)
        official_values = task_points["official_error_px"].to_numpy(dtype=np.float64)
        task_rows.append(
            {
                "task_id": str(task),
                "num_images": int(len(task_images)),
                "num_points": int(len(internal_values)),
                "mre_original_px": float(task_images["mre_original_px"].mean()),
                "measurement_proxy_mae": float(
                    task_images["measurement_proxy_mae"].mean()
                ),
                **_tail_metrics(internal_values),
                "official_mre_original_px": float(
                    task_images["official_mre_original_px"].mean()
                ),
                "official_measurement_proxy_mae": float(
                    task_images["official_measurement_proxy_mae"].mean()
                ),
                **_tail_metrics(official_values, prefix="official_"),
                "official_sort_changed_image_rate": float(
                    task_images["official_sort_changed"].mean()
                ),
            }
        )
    per_task = pd.DataFrame(task_rows)
    per_task["final_proxy_score"] = 0.5 * (
        per_task["mre_original_px"] + per_task["measurement_proxy_mae"]
    )
    per_task["official_final_proxy_score"] = 0.5 * (
        per_task["official_mre_original_px"]
        + per_task["official_measurement_proxy_mae"]
    )
    per_measurement = (
        per_measurement_image.groupby(["task_id", "measurement_name"], sort=True)
        .agg(
            num_images=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            official_mae=("official_absolute_error", "mean"),
        )
        .reset_index()
    )

    internal_values = per_point["error_px"].to_numpy(dtype=np.float64)
    official_values = per_point["official_error_px"].to_numpy(dtype=np.float64)
    summary = {
        "task_macro_mre_original_px": float(per_task["mre_original_px"].mean()),
        "task_macro_parameter_mae_proxy": float(
            per_task["measurement_proxy_mae"].mean()
        ),
        "final_proxy_score": float(per_task["final_proxy_score"].mean()),
        **_tail_metrics(internal_values),
        "official_task_macro_mre_original_px": float(
            per_task["official_mre_original_px"].mean()
        ),
        "official_task_macro_parameter_mae_proxy": float(
            per_task["official_measurement_proxy_mae"].mean()
        ),
        "official_final_proxy_score": float(
            per_task["official_final_proxy_score"].mean()
        ),
        **_tail_metrics(official_values, prefix="official_"),
    }
    psax = per_task[per_task["task_id"] == "PSAX"]
    if len(psax) == 1:
        row = psax.iloc[0]
        for column in (
            "mre_original_px",
            "measurement_proxy_mae",
            "final_proxy_score",
            "point_error_p90",
            "point_error_p95",
            "point_error_gt20_rate",
            "point_error_gt40_rate",
            "official_mre_original_px",
            "official_measurement_proxy_mae",
            "official_final_proxy_score",
            "official_point_error_p90",
            "official_point_error_p95",
            "official_point_error_gt20_rate",
            "official_point_error_gt40_rate",
        ):
            summary[f"psax_{column}"] = float(row[column])

    if cache_logits_dir is not None:
        cache_logits_dir.mkdir(parents=True, exist_ok=True)
        for task, arrays in logits_cache.items():
            np.savez_compressed(
                cache_logits_dir / f"{task}.npz",
                logits=np.concatenate(arrays, axis=0),
                metadata_json=np.asarray(
                    [
                        json.dumps(item, separators=(",", ":"))
                        for item in cache_metadata[task]
                    ],
                    dtype=np.str_,
                ),
            )
    return per_image, per_point, per_task, per_measurement, summary
