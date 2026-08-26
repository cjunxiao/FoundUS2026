from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "2-code/308-endpoint-identity-audit/src/audit_endpoint_identity.py",
    "2-code/308-endpoint-identity-audit/configs/exp308_endpoint_identity_audit.json",
    "2-code/308-endpoint-identity-audit/README.md",
    "2-code/308-endpoint-identity-audit/COMMAND.md",
    "1-docs/308-endpoint-identity-audit-literature-rationale.md",
    "2-code/161-stable-internal-exp152/src/evaluation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Exp161 endpoint identity audit.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def parse_points(value: str) -> np.ndarray:
    points = np.asarray(json.loads(str(value)), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid coordinate array: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Non-finite coordinate encountered.")
    return points


def length(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def circumference(points: np.ndarray) -> float:
    semi_a = 0.5 * length(points, 0, 1)
    semi_b = 0.5 * length(points, 2, 3)
    denominator = max(semi_a + semi_b, 1e-12)
    h = ((semi_a - semi_b) / denominator) ** 2
    return float(
        np.pi
        * (semi_a + semi_b)
        * (1.0 + 3.0 * h / (10.0 + np.sqrt(max(4.0 - 3.0 * h, 1e-12))))
    )


def measurement_values(
    points: np.ndarray, task_id: str, task_pairs: dict[str, list[dict[str, Any]]]
) -> np.ndarray:
    task = str(task_id)
    if task == "FA":
        return np.asarray([circumference(points)], dtype=np.float64)
    if task == "HC":
        return np.asarray(
            [length(points, 0, 1), circumference(points)], dtype=np.float64
        )
    if task == "AOP":
        raise ValueError("AOP is intentionally N/A in the endpoint-pair audit.")
    return np.asarray(
        [length(points, *pair["indices"]) for pair in task_pairs[task]],
        dtype=np.float64,
    )


def official_targets_from_validation(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame = pd.read_csv(path)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        task = str(row.task_id)
        source = str(row.source_image_path)
        count = int(row.num_classes)
        points = []
        for point_index in range(1, count + 1):
            parsed = ast.literal_eval(str(getattr(row, f"point_{point_index}_xy")))
            points.append([float(parsed[0]), float(parsed[1])])
        key = (task, source)
        if key in output:
            raise RuntimeError(f"Duplicate validation target: {key}")
        output[key] = {
            "points": np.asarray(points, dtype=np.float64),
            "width": int(row.width),
            "height": int(row.height),
        }
    return output


def summarize(values: pd.DataFrame, label: str, row_type: str) -> dict[str, Any]:
    if values.empty:
        raise RuntimeError(f"Cannot summarize empty subset: {label}")
    return {
        "task_pair": label,
        "row_type": row_type,
        "n": int(len(values)),
        "median_gt_pair_length_px": float(values.gt_pair_length_px.median()),
        "median_ordering_margin": float(values.ordering_margin.median()),
        "flip_rate": float(values.vertical_order_flip.mean()),
        "gt_tie_rate": float(values.gt_vertical_tie.mean()),
        "prediction_tie_rate": float(values.prediction_vertical_tie.mean()),
        "correspondence_swap_rate": float(values.correspondence_swap_better.mean()),
        "direct_mre_px": float(values.direct_mre_px.mean()),
        "pair_swapped_mre_px": float(values.pair_swapped_mre_px.mean()),
        "swap_oracle_mre_px": float(values.swap_oracle_mre_px.mean()),
        "oracle_gain_px": float(values.oracle_gain_px.mean()),
        "measurement_value_change_mean_abs": float(
            values.measurement_value_change_mean_abs.mean()
        ),
        "measurement_value_change_max_abs": float(
            values.measurement_value_change_max_abs.max()
        ),
        "measurement_mae_direct": float(values.measurement_mae_direct.mean()),
        "measurement_mae_swapped": float(values.measurement_mae_swapped.mean()),
        "measurement_mae_change": float(values.measurement_mae_change.mean()),
    }


def build_summary_tables(
    details: pd.DataFrame, config: dict[str, Any], aop_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for task, pair_specs in config["tasks"].items():
        task_values = details[details.task_id == task]
        for pair in pair_specs:
            subset = task_values[task_values.pair_name == pair["name"]]
            rows.append(summarize(subset, f"{task} / {pair['name']}", "pair"))
        if task == "A4C":
            for category in ("vertical", "horizontal"):
                subset = task_values[task_values.pair_category == category]
                rows.append(
                    summarize(subset, f"A4C / {category} aggregate", "category")
                )
        if len(pair_specs) > 1:
            rows.append(summarize(task_values, f"{task} / all pairs", "task"))

    controls = details[details.task_id.isin(config["control_tasks"])]
    rows.append(summarize(controls, "Controls / pooled pairs", "control"))
    main = pd.DataFrame(rows)
    main = pd.concat(
        [
            main,
            pd.DataFrame(
                [
                    {
                        "task_pair": "AOP / N/A",
                        "row_type": "n_a",
                        "n": int(aop_count),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    fold_rows: list[dict[str, Any]] = []
    for (fold, task, pair_name), subset in details.groupby(
        ["fold", "task_id", "pair_name"], sort=True
    ):
        record = summarize(subset, f"{task} / {pair_name}", "pair")
        record["fold"] = int(fold)
        record["task_id"] = str(task)
        record["pair_name"] = str(pair_name)
        fold_rows.append(record)
    for fold, fold_values in details.groupby("fold", sort=True):
        fixed_groups = (
            (
                "A4C / horizontal aggregate",
                fold_values[
                    (fold_values.task_id == "A4C")
                    & (fold_values.pair_category == "horizontal")
                ],
                "category",
            ),
            (
                "A4C / vertical aggregate",
                fold_values[
                    (fold_values.task_id == "A4C")
                    & (fold_values.pair_category == "vertical")
                ],
                "category",
            ),
            (
                "PSAX / all pairs",
                fold_values[fold_values.task_id == "PSAX"],
                "task",
            ),
            (
                "Controls / pooled pairs",
                fold_values[fold_values.task_id.isin(config["control_tasks"])],
                "control",
            ),
        )
        for label, subset, row_type in fixed_groups:
            record = summarize(subset, label, row_type)
            record["fold"] = int(fold)
            record["task_id"] = label.split(" / ")[0]
            record["pair_name"] = label.split(" / ")[1]
            fold_rows.append(record)
    return main, pd.DataFrame(fold_rows)


def row_by_label(frame: pd.DataFrame, label: str) -> pd.Series:
    selected = frame[frame.task_pair == label]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one summary row for {label}, found {len(selected)}")
    return selected.iloc[0]


def evidence_summary(
    main: pd.DataFrame, details: pd.DataFrame, tolerance: float
) -> dict[str, Any]:
    horizontal = row_by_label(main, "A4C / horizontal aggregate")
    vertical = row_by_label(main, "A4C / vertical aggregate")
    psax = row_by_label(main, "PSAX / all pairs")
    controls = row_by_label(main, "Controls / pooled pairs")

    control_tasks = []
    for task in sorted(set(details.task_id) - {"A4C", "PSAX"}):
        subset = details[details.task_id == task]
        control_tasks.append(summarize(subset, f"{task} / all pairs", "task"))
    control_task_frame = pd.DataFrame(control_tasks).sort_values(
        ["flip_rate", "oracle_gain_px"], ascending=False
    )
    top_flip = control_task_frame.iloc[0]
    top_gain = control_task_frame.sort_values("oracle_gain_px", ascending=False).iloc[0]

    max_measurement_change = float(details.measurement_value_change_max_abs.max())
    conditions = {
        "a4c_horizontal_exceeds_vertical_flip": bool(
            horizontal.flip_rate > vertical.flip_rate
        ),
        "a4c_horizontal_exceeds_vertical_oracle_gain": bool(
            horizontal.oracle_gain_px > vertical.oracle_gain_px
        ),
        "a4c_horizontal_exceeds_controls_flip": bool(
            horizontal.flip_rate > controls.flip_rate
        ),
        "a4c_horizontal_exceeds_controls_oracle_gain": bool(
            horizontal.oracle_gain_px > controls.oracle_gain_px
        ),
        "psax_exceeds_controls_flip": bool(psax.flip_rate > controls.flip_rate),
        "psax_exceeds_controls_oracle_gain": bool(
            psax.oracle_gain_px > controls.oracle_gain_px
        ),
        "measurement_invariant": bool(max_measurement_change <= tolerance),
        "a4c_and_psax_have_positive_oracle_gain": bool(
            horizontal.oracle_gain_px > 0.0 and psax.oracle_gain_px > 0.0
        ),
    }
    fold_directions = []
    for fold, fold_values in details.groupby("fold", sort=True):
        fold_horizontal = summarize(
            fold_values[
                (fold_values.task_id == "A4C")
                & (fold_values.pair_category == "horizontal")
            ],
            "A4C horizontal",
            "category",
        )
        fold_vertical = summarize(
            fold_values[
                (fold_values.task_id == "A4C")
                & (fold_values.pair_category == "vertical")
            ],
            "A4C vertical",
            "category",
        )
        fold_psax = summarize(
            fold_values[fold_values.task_id == "PSAX"], "PSAX", "task"
        )
        fold_controls = summarize(
            fold_values[
                ~fold_values.task_id.isin(["A4C", "PSAX"])
            ],
            "Controls",
            "control",
        )
        fold_directions.append(
            {
                "fold": int(fold),
                "a4c_horizontal_flip_rate": fold_horizontal["flip_rate"],
                "a4c_vertical_flip_rate": fold_vertical["flip_rate"],
                "a4c_horizontal_oracle_gain_px": fold_horizontal["oracle_gain_px"],
                "a4c_vertical_oracle_gain_px": fold_vertical["oracle_gain_px"],
                "a4c_two_metric_direction": bool(
                    fold_horizontal["flip_rate"] > fold_vertical["flip_rate"]
                    and fold_horizontal["oracle_gain_px"]
                    > fold_vertical["oracle_gain_px"]
                ),
                "psax_flip_rate": fold_psax["flip_rate"],
                "control_flip_rate": fold_controls["flip_rate"],
                "psax_oracle_gain_px": fold_psax["oracle_gain_px"],
                "control_oracle_gain_px": fold_controls["oracle_gain_px"],
                "psax_two_metric_direction": bool(
                    fold_psax["flip_rate"] > fold_controls["flip_rate"]
                    and fold_psax["oracle_gain_px"]
                    > fold_controls["oracle_gain_px"]
                ),
            }
        )
    title_problem_support = bool(all(conditions.values()))
    return {
        "a4c_horizontal": horizontal.to_dict(),
        "a4c_vertical": vertical.to_dict(),
        "psax": psax.to_dict(),
        "pooled_controls": controls.to_dict(),
        "highest_control_flip_task": top_flip.to_dict(),
        "highest_control_oracle_gain_task": top_gain.to_dict(),
        "max_measurement_value_change": max_measurement_change,
        "conditions": conditions,
        "fold_directions": fold_directions,
        "title_level_problem_motivation_support": title_problem_support,
        "method_improvement_support": False,
        "control_task_aggregates": control_task_frame.to_dict(orient="records"),
    }


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def markdown_table(main: pd.DataFrame) -> list[str]:
    lines = [
        "| Task / Pair | N | Median ordering margin | Flip rate | Direct MRE | Swap oracle MRE | Oracle gain | Measurement change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main.itertuples(index=False):
        if row.row_type == "n_a":
            lines.append(
                f"| {row.task_pair} | {int(row.n)} | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        change = (
            f"value {row.measurement_value_change_mean_abs:.2e}; "
            f"MAE {row.measurement_mae_change:+.2e}"
        )
        lines.append(
            f"| {row.task_pair} | {int(row.n)} | {row.median_ordering_margin:.4f} | "
            f"{percent(row.flip_rate)} | {row.direct_mre_px:.4f} | "
            f"{row.swap_oracle_mre_px:.4f} | {row.oracle_gain_px:.4f} | {change} |"
        )
    return lines


def build_report(
    main: pd.DataFrame,
    evidence: dict[str, Any],
    qc: dict[str, Any],
    config_record: dict[str, Any],
) -> str:
    horizontal = evidence["a4c_horizontal"]
    vertical = evidence["a4c_vertical"]
    psax = evidence["psax"]
    controls = evidence["pooled_controls"]
    top_flip = evidence["highest_control_flip_task"]
    top_gain = evidence["highest_control_oracle_gain_task"]
    conditions = evidence["conditions"]
    a4c_clear = all(
        conditions[name]
        for name in (
            "a4c_horizontal_exceeds_vertical_flip",
            "a4c_horizontal_exceeds_vertical_oracle_gain",
            "a4c_horizontal_exceeds_controls_flip",
            "a4c_horizontal_exceeds_controls_oracle_gain",
        )
    )
    psax_clear = all(
        conditions[name]
        for name in (
            "psax_exceeds_controls_flip",
            "psax_exceeds_controls_oracle_gain",
        )
    )
    comparable_control_tasks = [
        row["task_pair"]
        for row in evidence["control_task_aggregates"]
        if row["flip_rate"] >= min(horizontal["flip_rate"], psax["flip_rate"])
        and row["oracle_gain_px"]
        >= min(horizontal["oracle_gain_px"], psax["oracle_gain_px"])
    ]
    comparable_control = bool(comparable_control_tasks)
    a4c_supporting_folds = sum(
        int(row["a4c_two_metric_direction"]) for row in evidence["fold_directions"]
    )
    psax_supporting_folds = sum(
        int(row["psax_two_metric_direction"]) for row in evidence["fold_directions"]
    )
    lines = [
        "# Exp308 Endpoint Identity Audit Results",
        "",
        "## Scope",
        "",
        "Read-only audit of the fixed Exp161 five-fold OOF predictions. No model was trained, no checkpoint was reselected, and no prediction was sorted or remapped. Direct channel predictions are compared with raw current official GT coordinates.",
        "",
        f"- Selected epochs: `{qc['selected_epochs']}`.",
        f"- Complete OOF images: `{qc['oof_rows']}`; unique keys: `{qc['unique_oof_keys']}`.",
        f"- Pair instances: `{qc['pair_instances']}`.",
        f"- Input config: `{config_record['path']}` (`{config_record['sha256']}`).",
        f"- Maximum endpoint-swap measurement value change: `{evidence['max_measurement_value_change']:.3e}`.",
        "",
        "`Flip rate` is a disagreement between predicted and GT vertical order. `Swap oracle MRE` is the per-sample minimum of direct and pair-swapped correspondence errors. The two statistics intentionally measure different aspects of instability.",
        "",
        "## Main Table",
        "",
        *markdown_table(main),
        "",
        "## Fixed Comparison",
        "",
        "| Group | Flip rate | Oracle gain | Direct MRE | Oracle MRE |",
        "|---|---:|---:|---:|---:|",
        f"| A4C horizontal | {percent(horizontal['flip_rate'])} | {horizontal['oracle_gain_px']:.4f} | {horizontal['direct_mre_px']:.4f} | {horizontal['swap_oracle_mre_px']:.4f} |",
        f"| A4C vertical | {percent(vertical['flip_rate'])} | {vertical['oracle_gain_px']:.4f} | {vertical['direct_mre_px']:.4f} | {vertical['swap_oracle_mre_px']:.4f} |",
        f"| PSAX | {percent(psax['flip_rate'])} | {psax['oracle_gain_px']:.4f} | {psax['direct_mre_px']:.4f} | {psax['swap_oracle_mre_px']:.4f} |",
        f"| Other-task controls | {percent(controls['flip_rate'])} | {controls['oracle_gain_px']:.4f} | {controls['direct_mre_px']:.4f} | {controls['swap_oracle_mre_px']:.4f} |",
        "",
        "## Fold Direction",
        "",
        "| Fold | A4C-H flip | A4C-V flip | A4C-H gain | A4C-V gain | PSAX flip | Control flip | PSAX gain | Control gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['fold']} | {percent(row['a4c_horizontal_flip_rate'])} | {percent(row['a4c_vertical_flip_rate'])} | {row['a4c_horizontal_oracle_gain_px']:.4f} | {row['a4c_vertical_oracle_gain_px']:.4f} | {percent(row['psax_flip_rate'])} | {percent(row['control_flip_rate'])} | {row['psax_oracle_gain_px']:.4f} | {row['control_oracle_gain_px']:.4f} |"
            for row in evidence["fold_directions"]
        ],
        "",
        "## Conclusion",
        "",
        f"- **A4C horizontal pairs:** {'Yes' if a4c_clear else 'No'}, they {'are' if a4c_clear else 'are not'} consistently more identity-unstable than both A4C vertical pairs and pooled controls under the preregistered flip-rate and oracle-gain comparisons (horizontal {percent(horizontal['flip_rate'])}/{horizontal['oracle_gain_px']:.4f} px; vertical {percent(vertical['flip_rate'])}/{vertical['oracle_gain_px']:.4f} px; same direction in {a4c_supporting_folds}/5 folds).",
        f"- **PSAX:** {'Yes' if psax_clear else 'No'}, it {'shows' if psax_clear else 'does not show'} the same two-metric excess over pooled controls (PSAX {percent(psax['flip_rate'])}/{psax['oracle_gain_px']:.4f} px; controls {percent(controls['flip_rate'])}/{controls['oracle_gain_px']:.4f} px; same direction in {psax_supporting_folds}/5 folds).",
        f"- **Other tasks:** {'A comparable control problem exists in ' + ', '.join(comparable_control_tasks) if comparable_control else 'No control task is comparable on both prespecified statistics'}. The highest control flip rate is {top_flip['task_pair']} at {percent(top_flip['flip_rate'])}; the largest control oracle gain is {top_gain['task_pair']} at {top_gain['oracle_gain_px']:.4f} px.",
        f"- **Paper-title evidence:** {'Sufficient' if evidence['title_level_problem_motivation_support'] else 'Insufficient'} to present `Stable Landmark Identity` as a title-level problem mechanism or design motivation under the preregistered rule, but **insufficient by itself** to claim a performance-improving core method. This audit establishes only whether an order/identity problem is present in Exp161 OOF predictions; causal method benefit requires a separate matched ablation.",
        "",
        "## Artifacts",
        "",
        "- `endpoint_identity_pair_instances.csv`: every OOF image-pair statistic.",
        "- `endpoint_identity_main_table.csv`: pair, category, task, and pooled summaries.",
        "- `endpoint_identity_by_fold.csv`: fixed fold-level pair summaries.",
        "- `summary.json`: evidence conditions and machine-readable conclusion.",
        "- `qc_summary.json`: checksum, target, completeness, and invariance checks.",
        "",
    ]
    return "\n".join(lines)


def output_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [file_record(path) for path in sorted(paths)]


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    for task, pair_specs in config["tasks"].items():
        expected_measurements = 1 if task == "FA" else 2 if task == "HC" else len(pair_specs)
        if len(config["measurement_names"][task]) != expected_measurements:
            raise RuntimeError(f"Measurement-name count mismatch for {task}.")
    output_dir = resolve(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary_dir = resolve(config["source_summary_dir"])
    source_summary_path = source_summary_dir / "summary.json"
    source_oof_path = resolve(config["source_oof"])
    source_summary = read_json(source_summary_path)
    expected_epochs = [int(value) for value in config["expected_selected_epochs"]]
    if [int(value) for value in source_summary["selected_epochs"]] != expected_epochs:
        raise RuntimeError("Selected epochs differ from the preregistered values.")

    oof = pd.read_csv(source_oof_path)
    required_columns = {
        "fold",
        "task_id",
        "source_image_path",
        config["prediction_column"],
        config["cached_target_column"],
    }
    missing_columns = required_columns - set(oof.columns)
    if missing_columns:
        raise RuntimeError(f"Missing OOF columns: {sorted(missing_columns)}")
    if len(oof) != int(config["expected_oof_rows"]):
        raise RuntimeError(f"Unexpected OOF rows: {len(oof)}")
    if oof.duplicated(["task_id", "source_image_path"]).any():
        raise RuntimeError("Duplicate OOF task/image keys.")

    input_records = [file_record(config_path), file_record(source_summary_path), file_record(source_oof_path)]
    targets: dict[tuple[int, str, str], dict[str, Any]] = {}
    checkpoint_records = []
    for fold, expected_epoch in enumerate(expected_epochs):
        fold_run = resolve(str(config["fold_run_template"]).format(fold=fold))
        selected_path = fold_run / "selected_checkpoint.json"
        selected = read_json(selected_path)
        if int(selected["source_epoch"]) != expected_epoch:
            raise RuntimeError(f"Fold {fold} checkpoint epoch mismatch.")
        checkpoint_path = resolve(selected["path"])
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != str(selected["sha256"]):
            raise RuntimeError(f"Fold {fold} checkpoint checksum mismatch.")
        validation_path = fold_run / "input_data/validation.csv"
        for (task, source), target in official_targets_from_validation(validation_path).items():
            key = (fold, task, source)
            if key in targets:
                raise RuntimeError(f"Duplicate fold target: {key}")
            targets[key] = target
        checkpoint_records.append(
            {
                "fold": fold,
                "selected_epoch": expected_epoch,
                "checkpoint": file_record(checkpoint_path),
                "selection_record": file_record(selected_path),
                "validation": file_record(validation_path),
            }
        )
        input_records.extend([file_record(selected_path), file_record(validation_path)])

    details: list[dict[str, Any]] = []
    cache_target_max_difference = 0.0
    raw_boundary_adjustment_max = 0.0
    raw_boundary_adjusted_images = 0
    aop_count = 0
    tasks_seen = set()
    for row in oof.itertuples(index=False):
        fold = int(row.fold)
        task = str(row.task_id)
        source = str(row.source_image_path)
        tasks_seen.add(task)
        prediction = parse_points(getattr(row, config["prediction_column"]))
        cached_target = parse_points(getattr(row, config["cached_target_column"]))
        target_record = targets.get((fold, task, source))
        if target_record is None:
            raise RuntimeError(f"Missing raw official target: {(fold, task, source)}")
        target = np.asarray(target_record["points"], dtype=np.float64)
        if prediction.shape != target.shape or cached_target.shape != target.shape:
            raise RuntimeError(f"Point-count mismatch: {(fold, task, source)}")

        expected_cached = target.copy()
        expected_cached[:, 0] = np.clip(
            expected_cached[:, 0], 0.0, float(target_record["width"] - 1)
        )
        expected_cached[:, 1] = np.clip(
            expected_cached[:, 1], 0.0, float(target_record["height"] - 1)
        )
        expected_cached = expected_cached.astype(np.float32).astype(np.float64)
        cache_difference = float(np.max(np.abs(cached_target - expected_cached)))
        cache_target_max_difference = max(cache_target_max_difference, cache_difference)
        boundary_adjustment = float(np.max(np.abs(target - expected_cached)))
        raw_boundary_adjustment_max = max(raw_boundary_adjustment_max, boundary_adjustment)
        raw_boundary_adjusted_images += int(boundary_adjustment > 1e-3)

        if task in config["n_a_tasks"]:
            aop_count += 1
            continue
        if task not in config["tasks"]:
            raise RuntimeError(f"No pair definition for task: {task}")
        gt_measurements = measurement_values(target, task, config["tasks"])
        direct_measurements = measurement_values(prediction, task, config["tasks"])
        direct_measurement_mae = float(np.abs(direct_measurements - gt_measurements).mean())
        for pair_index, pair in enumerate(config["tasks"][task]):
            first, second = (int(value) for value in pair["indices"])
            gt_length = length(target, first, second)
            if gt_length <= 0.0:
                raise RuntimeError(f"Zero GT pair length: {(fold, task, source, pair['name'])}")
            gt_dy = float(target[first, 1] - target[second, 1])
            pred_dy = float(prediction[first, 1] - prediction[second, 1])
            direct = 0.5 * (
                float(np.linalg.norm(prediction[first] - target[first]))
                + float(np.linalg.norm(prediction[second] - target[second]))
            )
            swapped = 0.5 * (
                float(np.linalg.norm(prediction[first] - target[second]))
                + float(np.linalg.norm(prediction[second] - target[first]))
            )
            swapped_prediction = prediction.copy()
            swapped_prediction[[first, second]] = swapped_prediction[[second, first]]
            swapped_measurements = measurement_values(
                swapped_prediction, task, config["tasks"]
            )
            value_changes = np.abs(swapped_measurements - direct_measurements)
            direct_measurement_errors = np.abs(
                direct_measurements - gt_measurements
            )
            swapped_measurement_errors = np.abs(
                swapped_measurements - gt_measurements
            )
            swapped_measurement_mae = float(
                swapped_measurement_errors.mean()
            )
            details.append(
                {
                    "fold": fold,
                    "task_id": task,
                    "source_image_path": source,
                    "pair_index": pair_index,
                    "pair_name": str(pair["name"]),
                    "pair_category": str(pair["category"]),
                    "first_index": first,
                    "second_index": second,
                    "gt_pair_length_px": gt_length,
                    "gt_abs_dy_px": abs(gt_dy),
                    "ordering_margin": abs(gt_dy) / gt_length,
                    "gt_dy_px": gt_dy,
                    "prediction_dy_px": pred_dy,
                    "vertical_order_flip": bool(gt_dy * pred_dy < 0.0),
                    "gt_vertical_tie": bool(gt_dy == 0.0),
                    "prediction_vertical_tie": bool(pred_dy == 0.0),
                    "direct_mre_px": direct,
                    "pair_swapped_mre_px": swapped,
                    "swap_oracle_mre_px": min(direct, swapped),
                    "oracle_gain_px": direct - min(direct, swapped),
                    "correspondence_swap_better": bool(swapped < direct),
                    "measurement_value_change_mean_abs": float(value_changes.mean()),
                    "measurement_value_change_max_abs": float(value_changes.max()),
                    "measurement_names_json": json.dumps(
                        config["measurement_names"][task], separators=(",", ":")
                    ),
                    "measurement_target_values_json": json.dumps(
                        gt_measurements.tolist(), separators=(",", ":")
                    ),
                    "measurement_direct_values_json": json.dumps(
                        direct_measurements.tolist(), separators=(",", ":")
                    ),
                    "measurement_swapped_values_json": json.dumps(
                        swapped_measurements.tolist(), separators=(",", ":")
                    ),
                    "measurement_direct_absolute_errors_json": json.dumps(
                        direct_measurement_errors.tolist(), separators=(",", ":")
                    ),
                    "measurement_swapped_absolute_errors_json": json.dumps(
                        swapped_measurement_errors.tolist(), separators=(",", ":")
                    ),
                    "measurement_mae_direct": direct_measurement_mae,
                    "measurement_mae_swapped": swapped_measurement_mae,
                    "measurement_mae_change": swapped_measurement_mae
                    - direct_measurement_mae,
                }
            )

    if cache_target_max_difference > 1e-3:
        raise RuntimeError(
            f"Cached GT does not match loader-form raw GT: {cache_target_max_difference}"
        )
    expected_tasks = set(config["tasks"]) | set(config["n_a_tasks"])
    if tasks_seen != expected_tasks:
        raise RuntimeError(f"Task coverage mismatch: {sorted(tasks_seen)}")

    detail_frame = pd.DataFrame(details)
    main_table, fold_table = build_summary_tables(detail_frame, config, aop_count)
    tolerance = float(config["measurement_invariance_tolerance"])
    max_measurement_change = float(detail_frame.measurement_value_change_max_abs.max())
    max_measurement_mae_change = float(detail_frame.measurement_mae_change.abs().max())
    if max_measurement_change > tolerance or max_measurement_mae_change > tolerance:
        raise RuntimeError(
            "Endpoint swap changed an invariant measurement: "
            f"value={max_measurement_change}, MAE={max_measurement_mae_change}"
        )

    evidence = evidence_summary(main_table, detail_frame, tolerance)
    qc = {
        "status": "pass",
        "training_performed": False,
        "predictions_modified": False,
        "checkpoint_reselection": False,
        "threshold_search": False,
        "sample_filtering": False,
        "selected_epochs": expected_epochs,
        "checkpoint_hashes_verified": True,
        "oof_rows": int(len(oof)),
        "unique_oof_keys": int(
            oof[["task_id", "source_image_path"]].drop_duplicates().shape[0]
        ),
        "pair_instances": int(len(detail_frame)),
        "aop_n_a_images": int(aop_count),
        "tasks_seen": sorted(tasks_seen),
        "cache_target_max_difference": cache_target_max_difference,
        "raw_boundary_adjusted_images": int(raw_boundary_adjusted_images),
        "raw_boundary_adjustment_max": raw_boundary_adjustment_max,
        "measurement_invariance_tolerance": tolerance,
        "measurement_value_change_max": max_measurement_change,
        "measurement_mae_change_max_abs": max_measurement_mae_change,
    }

    config_record = file_record(config_path)
    report = build_report(main_table, evidence, qc, config_record)
    detail_path = output_dir / "endpoint_identity_pair_instances.csv"
    main_path = output_dir / "endpoint_identity_main_table.csv"
    fold_path = output_dir / "endpoint_identity_by_fold.csv"
    detail_frame.to_csv(detail_path, index=False)
    main_table.to_csv(main_path, index=False)
    fold_table.to_csv(fold_path, index=False)
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    results_doc_path = resolve(config["results_doc"])
    results_doc_path.write_text(report, encoding="utf-8")
    write_json(output_dir / "qc_summary.json", qc)
    write_json(
        output_dir / "summary.json",
        {
            "status": "complete",
            "experiment": config["experiment"],
            "scope": "read_only_full_exp161_fivefold_oof",
            "selected_epochs": expected_epochs,
            "evidence": evidence,
            "artifacts": {
                "details": str(detail_path.relative_to(PROJECT_ROOT)),
                "main_table": str(main_path.relative_to(PROJECT_ROOT)),
                "by_fold": str(fold_path.relative_to(PROJECT_ROOT)),
                "results": str((output_dir / "RESULTS.md").relative_to(PROJECT_ROOT)),
                "results_doc": config["results_doc"],
            },
        },
    )
    write_json(output_dir / "metric_summary.json", evidence)
    write_json(output_dir / "input_manifest.json", input_records)
    write_json(output_dir / "checkpoint_audit.json", checkpoint_records)
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    source_records = [file_record(resolve(path)) for path in SOURCE_FILES]
    write_json(
        output_dir / "source_status.json",
        {"git_repository": False, "source_files": source_records},
    )
    (output_dir / "command.txt").write_text(
        f"{sys.executable} 2-code/308-endpoint-identity-audit/src/audit_endpoint_identity.py --config {config_path.relative_to(PROJECT_ROOT)}\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "output_manifest.json"
    ] + [results_doc_path]
    write_json(output_dir / "output_manifest.json", output_manifest(outputs))
    print(report)


if __name__ == "__main__":
    main()
