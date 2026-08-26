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

import model as exp152_model


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

    for batch in tqdm(loader, desc="[Validation]", leave=False):
        task = str(batch["task_id"])
        images = batch["image"].to(device, non_blocking=True)
        output = exp152_model.forward_inference(network, images, task, settings)
        endpoint_output_policy = str(
            settings.get("endpoint_output_policy", "official_vertical")
        )
        if endpoint_output_policy == "fixed_internal":
            if "internal_coords_norm" not in output:
                raise RuntimeError(
                    "fixed_internal evaluation requires internal_coords_norm"
                )
            primary_norm = output["internal_coords_norm"].float().clamp(0.0, 1.0)
            structured_norm = primary_norm
        elif endpoint_output_policy == "official_vertical":
            primary_norm = output["coords_norm"].float().clamp(0.0, 1.0)
            structured_norm = output["structured_coords_norm"].float().clamp(0.0, 1.0)
        else:
            raise ValueError(
                f"Unknown endpoint_output_policy: {endpoint_output_policy}"
            )
        primary = canvas_to_original(primary_norm, batch["letterbox"], int(settings["input_size"]))
        structured = canvas_to_original(structured_norm, batch["letterbox"], int(settings["input_size"]))
        target = batch["points_original"].to(device).float()
        errors = torch.linalg.norm(primary - target, dim=-1)
        structured_errors = torch.linalg.norm(structured - target, dim=-1)
        predicted_measurements = measurement_values(primary, task)
        structured_measurements = measurement_values(structured, task)
        target_measurements = measurement_values(target, task)
        measurement_errors = (predicted_measurements - target_measurements).abs()
        structured_measurement_errors = (structured_measurements - target_measurements).abs()

        if cache_logits_dir is not None:
            logits_cache[task].append(
                output["heatmap_logits"].detach().float().cpu().numpy().astype(np.float16)
            )

        for image_index, source in enumerate(batch["source_image_path"]):
            predicted_list = primary[image_index].cpu().tolist()
            structured_list = structured[image_index].cpu().tolist()
            target_list = target[image_index].cpu().tolist()
            image_rows.append(
                {
                    "task_id": task,
                    "source_image_path": str(source),
                    "group_id": str(batch["group_id"][image_index]),
                    "num_points": int(errors[image_index].numel()),
                    "mre_original_px": float(errors[image_index].mean().cpu()),
                    "measurement_proxy_mae": float(measurement_errors[image_index].mean().cpu()),
                    "structured_mre_original_px": float(structured_errors[image_index].mean().cpu()),
                    "structured_measurement_proxy_mae": float(
                        structured_measurement_errors[image_index].mean().cpu()
                    ),
                    "predicted_points_json": json.dumps(predicted_list, separators=(",", ":")),
                    "structured_points_json": json.dumps(structured_list, separators=(",", ":")),
                    "target_points_json": json.dumps(target_list, separators=(",", ":")),
                }
            )
            for point_index, (error, structured_error) in enumerate(
                zip(errors[image_index].cpu().tolist(), structured_errors[image_index].cpu().tolist())
            ):
                point_rows.append(
                    {
                        "task_id": task,
                        "source_image_path": str(source),
                        "point_index": int(point_index),
                        "error_px": float(error),
                        "structured_error_px": float(structured_error),
                    }
                )
            for name, error, structured_error, predicted_value, target_value in zip(
                MEASUREMENT_NAMES[task],
                measurement_errors[image_index].cpu().tolist(),
                structured_measurement_errors[image_index].cpu().tolist(),
                predicted_measurements[image_index].cpu().tolist(),
                target_measurements[image_index].cpu().tolist(),
            ):
                measurement_rows.append(
                    {
                        "task_id": task,
                        "source_image_path": str(source),
                        "measurement_name": str(name),
                        "predicted_value": float(predicted_value),
                        "target_value": float(target_value),
                        "absolute_error": float(error),
                        "structured_absolute_error": float(structured_error),
                    }
                )
            if cache_logits_dir is not None:
                cache_metadata[task].append(
                    {
                        "source_image_path": str(source),
                        "group_id": str(batch["group_id"][image_index]),
                        "letterbox": {
                            key: float(value) for key, value in batch["letterbox"][image_index].items()
                        },
                        "target_points": target_list,
                    }
                )

    per_image = pd.DataFrame(image_rows)
    per_point = pd.DataFrame(point_rows)
    per_measurement_image = pd.DataFrame(measurement_rows)
    task_rows = []
    for task, task_images in per_image.groupby("task_id", sort=True):
        task_points = per_point[per_point["task_id"] == task]
        primary_values = task_points["error_px"].to_numpy(dtype=np.float64)
        structured_values = task_points["structured_error_px"].to_numpy(dtype=np.float64)
        task_rows.append(
            {
                "task_id": str(task),
                "num_images": int(len(task_images)),
                "num_points": int(len(primary_values)),
                "mre_original_px": float(task_images["mre_original_px"].mean()),
                "measurement_proxy_mae": float(task_images["measurement_proxy_mae"].mean()),
                **_tail_metrics(primary_values),
                "structured_mre_original_px": float(task_images["structured_mre_original_px"].mean()),
                "structured_measurement_proxy_mae": float(
                    task_images["structured_measurement_proxy_mae"].mean()
                ),
                **_tail_metrics(structured_values, prefix="structured_"),
            }
        )
    per_task = pd.DataFrame(task_rows)
    per_task["final_proxy_score"] = 0.5 * (
        per_task["mre_original_px"] + per_task["measurement_proxy_mae"]
    )
    per_task["structured_final_proxy_score"] = 0.5 * (
        per_task["structured_mre_original_px"] + per_task["structured_measurement_proxy_mae"]
    )
    per_measurement = (
        per_measurement_image.groupby(["task_id", "measurement_name"], sort=True)
        .agg(
            num_images=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            structured_mae=("structured_absolute_error", "mean"),
        )
        .reset_index()
    )
    primary_values = per_point["error_px"].to_numpy(dtype=np.float64)
    structured_values = per_point["structured_error_px"].to_numpy(dtype=np.float64)
    summary = {
        "task_macro_mre_original_px": float(per_task["mre_original_px"].mean()),
        "task_macro_parameter_mae_proxy": float(per_task["measurement_proxy_mae"].mean()),
        "final_proxy_score": float(per_task["final_proxy_score"].mean()),
        **_tail_metrics(primary_values),
        "structured_task_macro_mre_original_px": float(per_task["structured_mre_original_px"].mean()),
        "structured_task_macro_parameter_mae_proxy": float(
            per_task["structured_measurement_proxy_mae"].mean()
        ),
        "structured_final_proxy_score": float(per_task["structured_final_proxy_score"].mean()),
        **_tail_metrics(structured_values, prefix="structured_"),
    }
    psax = per_task[per_task["task_id"] == "PSAX"]
    if len(psax) == 1:
        row = psax.iloc[0]
        for column in (
            "mre_original_px", "measurement_proxy_mae", "final_proxy_score",
            "point_error_p90", "point_error_p95", "point_error_gt20_rate", "point_error_gt40_rate",
            "structured_mre_original_px", "structured_measurement_proxy_mae", "structured_final_proxy_score",
            "structured_point_error_p90", "structured_point_error_p95",
            "structured_point_error_gt20_rate", "structured_point_error_gt40_rate",
        ):
            summary[f"psax_{column}"] = float(row[column])

    if cache_logits_dir is not None:
        cache_logits_dir.mkdir(parents=True, exist_ok=True)
        for task, arrays in logits_cache.items():
            np.savez_compressed(
                cache_logits_dir / f"{task}.npz",
                logits=np.concatenate(arrays, axis=0),
                metadata_json=np.asarray(
                    [json.dumps(item, separators=(",", ":")) for item in cache_metadata[task]],
                    dtype=np.str_,
                ),
            )
    return per_image, per_point, per_task, per_measurement, summary
