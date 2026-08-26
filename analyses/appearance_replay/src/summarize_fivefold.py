from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT_ROOT / "2-code/164-task-conditioned-unlabeled-appearance-replay"
DEFAULT_OUTPUT = PROJECT_ROOT / "4-runs/164-task-conditioned-appearance-replay-5fold-summary"
EXPECTED_TASKS = {"A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"}
EXPECTED_UNLABELED = 182_870


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed Exp164 grouped five-fold OOF results.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def branch_dir(fold: int, branch: str) -> Path:
    return PROJECT_ROOT / (
        f"4-runs/164-task-conditioned-appearance-replay-fold{fold}-"
        f"{branch.lower()}-seed42"
    )


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"run_dir", "branch", "style_donor", "require_full_unlabeled_coverage"}
    path_keys = {"anchor_checkpoint", "unlabeled_manifest", "unlabeled_summary"}

    def project_relative(value: str) -> str:
        parts = Path(value).parts
        for marker in ("3-data", "4-runs"):
            if marker in parts:
                return str(Path(*parts[parts.index(marker) :]))
        return str(value)

    comparable = {key: value for key, value in config.items() if key not in ignored}
    for key in path_keys:
        comparable[key] = project_relative(str(comparable[key]))
    comparable["bad_image_paths"] = [
        project_relative(str(value)) for value in comparable["bad_image_paths"]
    ]
    return comparable


def aggregate_metrics(
    images: pd.DataFrame,
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for task_id, task_images in images.groupby("task_id", sort=True):
        task_points = points[points["task_id"] == task_id]
        errors = task_points["official_error_px"].to_numpy(dtype=np.float64)
        mre = float(task_images["official_mre_original_px"].mean())
        measurement = float(task_images["official_measurement_proxy_mae"].mean())
        rows.append(
            {
                "task_id": str(task_id),
                "num_images": int(len(task_images)),
                "num_points": int(len(task_points)),
                "mre": mre,
                "measurement_proxy": measurement,
                "final_proxy": 0.5 * (mre + measurement),
                "p50": float(np.quantile(errors, 0.50)),
                "p90": float(np.quantile(errors, 0.90)),
                "p95": float(np.quantile(errors, 0.95)),
                "gt20_rate": float(np.mean(errors > 20.0)),
                "gt40_rate": float(np.mean(errors > 40.0)),
            }
        )
    per_task = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    if set(per_task["task_id"]) != EXPECTED_TASKS:
        raise RuntimeError(f"Unexpected task set: {sorted(per_task['task_id'])}")
    all_errors = points["official_error_px"].to_numpy(dtype=np.float64)
    summary = {
        "task_macro_mre": float(per_task["mre"].mean()),
        "task_macro_measurement_proxy": float(per_task["measurement_proxy"].mean()),
        "task_macro_final_proxy": float(per_task["final_proxy"].mean()),
        "task_macro_p90": float(per_task["p90"].mean()),
        "task_macro_gt40_rate": float(per_task["gt40_rate"].mean()),
        "pooled_point_p50": float(np.quantile(all_errors, 0.50)),
        "pooled_point_p90": float(np.quantile(all_errors, 0.90)),
        "pooled_point_p95": float(np.quantile(all_errors, 0.95)),
        "pooled_point_gt20_rate": float(np.mean(all_errors > 20.0)),
        "pooled_point_gt40_rate": float(np.mean(all_errors > 40.0)),
        "num_images": int(len(images)),
        "num_points": int(len(points)),
    }
    return per_task, summary


def delta_metrics(right: dict[str, float], left: dict[str, float]) -> dict[str, float]:
    return {
        key: float(right[key] - left[key])
        for key in right
        if key not in {"num_images", "num_points"}
    }


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.refresh:
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    pooled_images: dict[str, list[pd.DataFrame]] = {"ANCHOR": [], "B0": [], "B1": []}
    pooled_points: dict[str, list[pd.DataFrame]] = {"ANCHOR": [], "B0": [], "B1": []}
    schedule_qc: list[dict[str, Any]] = []
    coverage_qc: list[dict[str, Any]] = []
    input_paths: set[Path] = set()

    for fold in range(5):
        fold_data: dict[str, dict[str, Any]] = {}
        configs: dict[str, dict[str, Any]] = {}
        for branch in ("B0", "B1"):
            directory = branch_dir(fold, branch)
            summary_path = directory / "run_summary.json"
            if not summary_path.exists() or read_json(summary_path).get("status") != "complete":
                raise RuntimeError(f"Incomplete run: {directory}")
            config_path = directory / "resolved_config.json"
            image_path = directory / "val_per_image_epoch_008_scale_1p00.csv"
            point_path = directory / "val_per_point_epoch_008_scale_1p00.csv"
            config = read_json(config_path)
            if int(config["fold"]) != fold or int(config["epochs"]) != 8:
                raise RuntimeError(f"Wrong fixed protocol in {directory}")
            images = pd.read_csv(image_path)
            points = pd.read_csv(point_path)
            images["fold"] = fold
            images["branch"] = branch
            points["fold"] = fold
            points["branch"] = branch
            per_task, metrics = aggregate_metrics(images, points)
            fold_data[branch] = {
                "images": images,
                "points": points,
                "per_task": per_task,
                "metrics": metrics,
            }
            configs[branch] = config
            pooled_images[branch].append(images)
            pooled_points[branch].append(points)
            input_paths.update({summary_path, config_path, image_path, point_path})

        if comparable_config(configs["B0"]) != comparable_config(configs["B1"]):
            raise RuntimeError(f"Fold {fold} B0/B1 configs are not matched.")
        b0_images = fold_data["B0"]["images"]
        b1_images = fold_data["B1"]["images"]
        join_keys = ["task_id", "source_image_path", "group_id", "fold"]
        paired = b1_images.merge(
            b0_images,
            on=join_keys,
            suffixes=("_b1", "_b0"),
            validate="one_to_one",
        )
        if len(paired) != len(b0_images) or len(paired) != len(b1_images):
            raise RuntimeError(f"Fold {fold} B0/B1 validation image mismatch.")
        paired["delta_mre"] = (
            paired["official_mre_original_px_b1"]
            - paired["official_mre_original_px_b0"]
        )
        paired["delta_measurement_proxy"] = (
            paired["official_measurement_proxy_mae_b1"]
            - paired["official_measurement_proxy_mae_b0"]
        )
        paired["delta_final_proxy"] = 0.5 * (
            paired["delta_mre"] + paired["delta_measurement_proxy"]
        )
        paired_frames.append(paired)

        anchor_b0_image_path = branch_dir(fold, "B0") / "val_per_image_epoch_000_scale_0p00.csv"
        anchor_b1_image_path = branch_dir(fold, "B1") / "val_per_image_epoch_000_scale_0p00.csv"
        anchor_point_path = branch_dir(fold, "B0") / "val_per_point_epoch_000_scale_0p00.csv"
        anchor_b0 = pd.read_csv(anchor_b0_image_path)
        anchor_b1 = pd.read_csv(anchor_b1_image_path)
        anchor_check = anchor_b0.merge(
            anchor_b1,
            on=["task_id", "source_image_path", "group_id"],
            suffixes=("_b0", "_b1"),
            validate="one_to_one",
        )
        if not np.allclose(
            anchor_check["official_mre_original_px_b0"],
            anchor_check["official_mre_original_px_b1"],
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(f"Fold {fold} B0/B1 anchors differ.")
        anchor_b0["fold"] = fold
        anchor_b0["branch"] = "ANCHOR"
        anchor_points = pd.read_csv(anchor_point_path)
        anchor_points["fold"] = fold
        anchor_points["branch"] = "ANCHOR"
        pooled_images["ANCHOR"].append(anchor_b0)
        pooled_points["ANCHOR"].append(anchor_points)
        input_paths.update({anchor_b0_image_path, anchor_b1_image_path, anchor_point_path})

        b0_schedule_path = branch_dir(fold, "B0") / "labeled_schedule_audit.json"
        b1_schedule_path = branch_dir(fold, "B1") / "labeled_schedule_audit.json"
        b0_schedule = read_json(b0_schedule_path)
        b1_schedule = read_json(b1_schedule_path)
        schedule_match = len(b0_schedule) == len(b1_schedule) == 8 and all(
            left[key] == right[key]
            for left, right in zip(b0_schedule, b1_schedule)
            for key in ("epoch", "steps", "content_schedule_sha256", "style_parameter_schedule_sha256")
        )
        if not schedule_match:
            raise RuntimeError(f"Fold {fold} B0/B1 schedule mismatch.")
        schedule_qc.append({"fold": fold, "paired_schedule_exact_match": True})
        input_paths.update({b0_schedule_path, b1_schedule_path})

        coverage_path = branch_dir(fold, "B1") / "unlabeled_coverage.json"
        coverage = read_json(coverage_path)
        coverage_pass = (
            int(coverage["unlabeled_total_unique"]) == EXPECTED_UNLABELED
            and int(coverage["unlabeled_seen_unique"]) == EXPECTED_UNLABELED
            and int(coverage["unlabeled_total_draws"]) == EXPECTED_UNLABELED
            and int(coverage["unlabeled_effective_style_unique"]) == EXPECTED_UNLABELED
            and int(coverage["unlabeled_effective_style_draws"]) == EXPECTED_UNLABELED
            and int(coverage["unlabeled_repeated_unique"]) == 0
            and int(coverage["unlabeled_effective_style_repeated_unique"]) == 0
        )
        if not coverage_pass:
            raise RuntimeError(f"Fold {fold} effective unlabeled coverage failed.")
        coverage_qc.append({"fold": fold, "effective_unique_exactly_once": True})
        input_paths.add(coverage_path)

        b0_metrics = fold_data["B0"]["metrics"]
        b1_metrics = fold_data["B1"]["metrics"]
        row: dict[str, Any] = {"fold": fold, "num_images": int(len(paired))}
        for branch, values in (("b0", b0_metrics), ("b1", b1_metrics)):
            for key, value in values.items():
                row[f"{branch}_{key}"] = value
        for key, value in delta_metrics(b1_metrics, b0_metrics).items():
            row[f"delta_{key}"] = value
        row["final_proxy_improved"] = bool(row["delta_task_macro_final_proxy"] < 0.0)
        fold_rows.append(row)

    combined_images: dict[str, pd.DataFrame] = {}
    combined_points: dict[str, pd.DataFrame] = {}
    per_task_tables: dict[str, pd.DataFrame] = {}
    pooled_metrics: dict[str, dict[str, float]] = {}
    for branch in ("ANCHOR", "B0", "B1"):
        images = pd.concat(pooled_images[branch], ignore_index=True)
        points = pd.concat(pooled_points[branch], ignore_index=True)
        if images.duplicated(["task_id", "source_image_path"]).any():
            raise RuntimeError(f"Duplicate OOF image in {branch}.")
        combined_images[branch] = images
        combined_points[branch] = points
        per_task_tables[branch], pooled_metrics[branch] = aggregate_metrics(images, points)

    per_task = per_task_tables["ANCHOR"].add_suffix("_anchor").rename(
        columns={"task_id_anchor": "task_id"}
    )
    for branch in ("B0", "B1"):
        suffix = branch.lower()
        table = per_task_tables[branch].add_suffix(f"_{suffix}").rename(
            columns={f"task_id_{suffix}": "task_id"}
        )
        per_task = per_task.merge(table, on="task_id", validate="one_to_one")
    for metric in ("mre", "measurement_proxy", "final_proxy", "p90", "gt40_rate"):
        per_task[f"delta_{metric}_b1_minus_b0"] = (
            per_task[f"{metric}_b1"] - per_task[f"{metric}_b0"]
        )
        per_task[f"delta_{metric}_b1_minus_anchor"] = (
            per_task[f"{metric}_b1"] - per_task[f"{metric}_anchor"]
        )

    deltas_b1_b0 = delta_metrics(pooled_metrics["B1"], pooled_metrics["B0"])
    deltas_b1_anchor = delta_metrics(pooled_metrics["B1"], pooled_metrics["ANCHOR"])
    fold_frame = pd.DataFrame(fold_rows).sort_values("fold")
    paired = pd.concat(paired_frames, ignore_index=True)

    leave_one_fold_out = []
    for excluded_fold in range(5):
        branch_metrics = {}
        for branch in ("B0", "B1"):
            images = combined_images[branch][combined_images[branch]["fold"] != excluded_fold]
            points = combined_points[branch][combined_points[branch]["fold"] != excluded_fold]
            _, branch_metrics[branch] = aggregate_metrics(images, points)
        delta = delta_metrics(branch_metrics["B1"], branch_metrics["B0"])
        leave_one_fold_out.append({"excluded_fold": excluded_fold, **delta})

    leave_one_task_out = []
    for excluded_task in sorted(EXPECTED_TASKS):
        subset = per_task[per_task["task_id"] != excluded_task]
        leave_one_task_out.append(
            {
                "excluded_task": excluded_task,
                "delta_mre": float(subset["delta_mre_b1_minus_b0"].mean()),
                "delta_measurement_proxy": float(
                    subset["delta_measurement_proxy_b1_minus_b0"].mean()
                ),
                "delta_final_proxy": float(
                    subset["delta_final_proxy_b1_minus_b0"].mean()
                ),
            }
        )

    checks = {
        "pooled_final_proxy_improves_0p10": deltas_b1_b0["task_macro_final_proxy"] <= -0.10,
        "at_least_three_folds_improve": int(fold_frame["final_proxy_improved"].sum()) >= 3,
        "leave_one_task_out_direction_robust": all(
            row["delta_final_proxy"] < 0.0 for row in leave_one_task_out
        ),
        "paired_schedules_exact": all(row["paired_schedule_exact_match"] for row in schedule_qc),
        "effective_unlabeled_coverage_exact": all(
            row["effective_unique_exactly_once"] for row in coverage_qc
        ),
    }
    decision = {
        "status": "complete",
        "protocol": "grouped five-fold OOF; fixed epoch 8; fixed Adapter scale 1.0; official endpoint space",
        "pooled": pooled_metrics,
        "delta_b1_minus_b0": deltas_b1_b0,
        "delta_b1_minus_anchor": deltas_b1_anchor,
        "improved_folds": int(fold_frame["final_proxy_improved"].sum()),
        "leave_one_fold_out": leave_one_fold_out,
        "leave_one_task_out": leave_one_task_out,
        "checks": checks,
        "positive_ssl_result": bool(all(checks.values())),
        "interpretation": (
            "The unlabeled contribution is identified only by the fixed-protocol "
            "B1-minus-matched-B0 comparison; B1 versus anchor is secondary."
        ),
    }

    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    per_task.to_csv(output / "pooled_task_metrics.csv", index=False)
    paired.to_csv(
        output / "paired_per_image.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(leave_one_fold_out).to_csv(output / "leave_one_fold_out.csv", index=False)
    pd.DataFrame(leave_one_task_out).to_csv(output / "leave_one_task_out.csv", index=False)
    write_json(output / "schedule_qc.json", schedule_qc)
    write_json(output / "coverage_qc.json", coverage_qc)
    write_json(output / "metrics_summary.json", decision)
    write_json(output / "decision.json", decision)
    write_json(
        output / "command.json",
        {"cwd": str(PROJECT_ROOT), "argv": [str(value) for value in sys.argv], "python": sys.executable},
    )
    package_versions = {}
    for package in ("numpy", "pandas"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    write_json(
        output / "environment.json",
        {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": package_versions,
        },
    )
    source_paths = [
        Path(__file__).resolve(),
        EXPERIMENT / "src/train_exp164.py",
        EXPERIMENT / "src/run_exp164_fivefold_queue.py",
        *sorted((EXPERIMENT / "configs").glob("exp164_fold*_*.json")),
    ]
    write_json(output / "source_status.json", [file_record(path) for path in source_paths])
    write_json(output / "input_manifest.json", [file_record(path) for path in sorted(input_paths)])
    write_json(
        output / "qc_summary.json",
        {
            "status": "pass",
            "folds_complete": 5,
            "fixed_epoch": 8,
            "fixed_adapter_scale": 1.0,
            "validation_selection_performed": False,
            "paired_schedules_exact": checks["paired_schedules_exact"],
            "effective_unlabeled_coverage_exact": checks["effective_unlabeled_coverage_exact"],
        },
    )

    lines = [
        "# Exp164 Five-fold OOF Result",
        "",
        "Fixed epoch 8 and Adapter scale 1.0. Deltas are B1 minus matched B0.",
        "",
        "| Fold | Images | Delta MRE | Delta MAE proxy | Delta Final Proxy | Improved |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['num_images']} | {row['delta_task_macro_mre']:+.4f} | "
            f"{row['delta_task_macro_measurement_proxy']:+.4f} | "
            f"{row['delta_task_macro_final_proxy']:+.4f} | "
            f"{'yes' if row['final_proxy_improved'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Pooled delta MRE: `{deltas_b1_b0['task_macro_mre']:+.4f}`",
            f"Pooled delta MAE proxy: `{deltas_b1_b0['task_macro_measurement_proxy']:+.4f}`",
            f"Pooled delta Final Proxy: `{deltas_b1_b0['task_macro_final_proxy']:+.4f}`",
            f"Pooled point P90 delta: `{deltas_b1_b0['pooled_point_p90']:+.4f}`",
            f"Improved folds: `{decision['improved_folds']}/5`",
            f"Strict SSL conclusion: `{'positive' if decision['positive_ssl_result'] else 'negative'}`",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    outputs = [
        file_record(path)
        for path in sorted(value for value in output.iterdir() if value.is_file())
        if path.name != "output_manifest.json"
    ]
    write_json(output / "output_manifest.json", outputs)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
