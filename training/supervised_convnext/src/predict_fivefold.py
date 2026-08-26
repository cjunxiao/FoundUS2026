from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_INFERENCE = (
    PROJECT_ROOT
    / "2-code/159-canonical-internal-exp152/src/predict_official_5fold_ensemble.py"
)
INTERNAL_TASK_PAIRS = {
    "A4C": tuple((index, index + 1) for index in range(0, 16, 2)),
    "PSAX": ((0, 1), (2, 3)),
}


def load_inference_module():
    name = "exp159_inference_engine_for_exp161"
    spec = importlib.util.spec_from_file_location(name, LEGACY_INFERENCE)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY_INFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def argument_value(name: str) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(name, required=True)
    arguments, _ = parser.parse_known_args()
    return str(getattr(arguments, name.removeprefix("--").replace("-", "_")))


def official_vertical_once(prediction: np.ndarray, task_id: str) -> np.ndarray:
    points = np.asarray(prediction, dtype=np.float64).reshape(-1, 2).copy()
    for first, second in INTERNAL_TASK_PAIRS.get(str(task_id), ()):
        if points[first, 1] > points[second, 1]:
            points[[first, second]] = points[[second, first]]
    return points


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def repair_audit_metadata(engine: Any, preserve_fixed_channels: bool) -> None:
    run_dir = engine.resolve(argument_value("--run-dir"))
    submission_dir = engine.resolve(argument_value("--submission-dir"))
    if preserve_fixed_channels:
        method = (
            "five Exp161 models selected against stable internal endpoint identities; "
            "per-landmark coordinate median; preserve fixed output channels without "
            "value-dependent endpoint sorting"
        )
        endpoint_policy = "preserve_fixed_channels_after_ensemble"
        ensemble_order = (
            "five fixed-internal predictions -> coordinate median -> preserve channels"
        )
        diagnostic_submission = True
        official_result_used_for_policy_selection = True
        release_interpretation = (
            "Post-hoc fixed-channel diagnostic created after the scored sorted "
            "artifact exposed endpoint-swap damage; not an unbiased model-selection result."
        )
    else:
        method = (
            "five Exp161 models selected against stable internal endpoint identities; "
            "per-landmark coordinate median in stable channels; one official vertical "
            "conversion after ensemble aggregation"
        )
        endpoint_policy = (
            "fixed_internal_per_fold_then_official_vertical_after_ensemble"
        )
        ensemble_order = (
            "five fixed-internal predictions -> coordinate median -> one "
            "official vertical conversion"
        )
        diagnostic_submission = False
        official_result_used_for_policy_selection = False
        release_interpretation = (
            "Formal Exp161 stable-internal selection with one post-median vertical "
            "conversion; subsequently rejected after official scoring."
        )

    inference_config_path = run_dir / "inference_config.json"
    inference_config = read_json(inference_config_path)
    inference_config.update(
        {
            "method": method,
            "endpoint_output_policy": endpoint_policy,
            "ensemble_order": ensemble_order,
            "official_validation_labels_used": False,
            "official_result_used_for_policy_selection": (
                official_result_used_for_policy_selection
            ),
            "diagnostic_submission": diagnostic_submission,
            "release_interpretation": release_interpretation,
        }
    )
    write_json(inference_config_path, inference_config)

    input_manifest_path = run_dir / "input_manifest.json"
    input_manifest = read_json(input_manifest_path)
    input_manifest["producing_experiment"] = "161-stable-internal-exp152"
    input_manifest["aggregation_contract"] = inference_config["ensemble_order"]
    write_json(input_manifest_path, input_manifest)

    qc_path = run_dir / "qc_summary.json"
    qc = read_json(qc_path)
    qc["stable_internal_per_fold"] = True
    qc["single_official_vertical_conversion_after_median"] = (
        not preserve_fixed_channels
    )
    qc["value_dependent_endpoint_sorting_disabled"] = preserve_fixed_channels
    write_json(qc_path, qc)

    wrapper = Path(__file__).resolve()
    snapshot = run_dir / "source_snapshot" / "2-code__161-stable-internal-exp152__src__predict_official_5fold_ensemble.py"
    shutil.copy2(wrapper, snapshot)
    source_status_path = run_dir / "source_status.json"
    source_status = read_json(source_status_path)
    record = engine.file_record(wrapper)
    record["snapshot"] = engine.relative(snapshot)
    source_status["inference_sources"].append(record)
    write_json(source_status_path, source_status)

    submission_manifest_path = submission_dir / "submission_manifest.json"
    submission_manifest = read_json(submission_manifest_path)
    submission_manifest.update(
        {
            "producing_experiment": "161-stable-internal-exp152",
            "method": method,
            "official_result_used_for_policy_selection": (
                official_result_used_for_policy_selection
            ),
            "diagnostic_submission": diagnostic_submission,
            "release_interpretation": release_interpretation,
            "aggregation_contract": inference_config["ensemble_order"],
        }
    )
    write_json(submission_manifest_path, submission_manifest)
    write_json(run_dir / "output_manifest.json", engine.output_manifest(run_dir))
    print(json.dumps(submission_manifest, indent=2))


def main() -> None:
    preserve_fixed_channels = "--preserve-fixed-channels" in sys.argv
    if preserve_fixed_channels:
        sys.argv.remove("--preserve-fixed-channels")
    if "--endpoint-output-policy" not in sys.argv:
        sys.argv.extend(["--endpoint-output-policy", "fixed_internal"])
    else:
        position = sys.argv.index("--endpoint-output-policy")
        if position + 1 >= len(sys.argv) or sys.argv[position + 1] != "fixed_internal":
            raise ValueError("Exp161 inference requires per-fold fixed_internal output.")

    engine = load_inference_module()
    original_prediction_item = engine.prediction_item

    def prediction_item_after_single_sort(
        prediction: np.ndarray,
        *,
        image_path: str,
        task_id: str,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        prepared = np.asarray(prediction, dtype=np.float64).reshape(-1, 2)
        official = (
            prepared
            if preserve_fixed_channels
            else official_vertical_once(prepared, task_id)
        )
        return original_prediction_item(
            official.reshape(-1),
            image_path=image_path,
            task_id=task_id,
            width=width,
            height=height,
        )

    engine.prediction_item = prediction_item_after_single_sort
    engine.main()
    repair_audit_metadata(engine, preserve_fixed_channels)


if __name__ == "__main__":
    main()
