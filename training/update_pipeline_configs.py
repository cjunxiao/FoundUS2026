from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "training"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_file(value: str | Path, label: str) -> Path:
    path = resolve(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def checkpoint_record(fold: int, path: Path) -> dict[str, Any]:
    return {"fold": fold, "path": relative(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update FoundUS working configs with local paths and SHA256 values."
    )
    parser.add_argument("--labeled-csv-dir", default="data/labeled_csv")
    parser.add_argument("--fold-manifest", default="data/grouped_5fold_manifest.csv")
    parser.add_argument("--unlabeled-manifest", default="data/official_unlabeled_manifest.csv")
    parser.add_argument(
        "--convnext-weights",
        default="weights/convnext_small.in12k_ft_in1k.safetensors",
    )
    parser.add_argument("--usfm-weights", default="weights/USFM_latest.pth")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    labeled_dir = resolve(args.labeled_csv_dir)
    if not labeled_dir.is_dir():
        raise FileNotFoundError(f"labeled CSV directory: {labeled_dir}")
    fold_manifest = require_file(args.fold_manifest, "fold manifest")
    unlabeled_manifest = require_file(args.unlabeled_manifest, "unlabeled manifest")
    convnext_weights = require_file(args.convnext_weights, "ConvNeXt weights")
    usfm_weights = require_file(args.usfm_weights, "USFM weights")
    a4c_csv = require_file(labeled_dir / "A4C_train.csv", "A4C canonical CSV")
    psax_csv = require_file(labeled_dir / "PSAX_train.csv", "PSAX canonical CSV")
    output_root = resolve(args.output_root)

    protocol_sha = sha256_file(fold_manifest)
    active_hashes = {"A4C": sha256_file(a4c_csv), "PSAX": sha256_file(psax_csv)}
    usfm_sha = sha256_file(usfm_weights)
    unlabeled_sha = sha256_file(unlabeled_manifest)

    for stage in ("supervised_convnext", "usfm_heatmap", "dense_fusion"):
        for fold in range(5):
            path = CONFIG_ROOT / stage / "configs" / f"fold{fold}.json"
            config = read_json(path)
            config["run_dir"] = relative(output_root / stage / f"fold{fold}")
            config["train_csv_dir"] = relative(labeled_dir)
            config["protocol_manifest"] = relative(fold_manifest)
            config["protocol_manifest_sha256"] = protocol_sha
            config["active_label_canonical_sha256"] = active_hashes
            if stage == "supervised_convnext":
                config["encoder_weights"] = relative(convnext_weights)
            else:
                config["usfm_checkpoint"] = relative(usfm_weights)
                config["usfm_checkpoint_sha256"] = usfm_sha
            if stage == "dense_fusion":
                convnext = output_root / "supervised_convnext" / f"fold{fold}" / "checkpoints/best_final_proxy.pt"
                foundation = output_root / "usfm_heatmap" / f"fold{fold}" / "checkpoints/best_final_proxy.pt"
                config["convnext_checkpoint"] = relative(convnext)
                config["foundation_checkpoint"] = relative(foundation)
                if convnext.is_file():
                    config["convnext_checkpoint_sha256"] = sha256_file(convnext)
                if foundation.is_file():
                    config["foundation_checkpoint_sha256"] = sha256_file(foundation)
            write_json(path, config)

    calibration_path = CONFIG_ROOT / "configs/fusion_scales.json"
    calibration = read_json(calibration_path)
    calibration["run_dir"] = relative(output_root / "fusion_scale_calibration")
    calibration["fold_runs"] = [
        {
            "fold": fold,
            "anchor_run": relative(output_root / "supervised_convnext" / f"fold{fold}"),
            "fusion_run": relative(output_root / "dense_fusion" / f"fold{fold}"),
        }
        for fold in range(5)
    ]
    write_json(calibration_path, calibration)

    dense_checkpoints = [
        output_root / "dense_fusion" / f"fold{fold}" / "checkpoints/best_final_proxy.pt"
        for fold in range(5)
    ]
    dense_ready = all(path.is_file() for path in dense_checkpoints)
    records = (
        [checkpoint_record(fold, path) for fold, path in enumerate(dense_checkpoints)]
        if dense_ready
        else None
    )

    calibration_summary = output_root / "fusion_scale_calibration/summary.json"
    task_scales = None
    if calibration_summary.is_file():
        task_scales = {
            str(task): float(scale)
            for task, scale in read_json(calibration_summary)["locked_task_scales"].items()
        }

    warm_path = CONFIG_ROOT / "unlabeled_distillation/configs/warmstart.json"
    warm = read_json(warm_path)
    warm["run_dir"] = relative(output_root / "unlabeled_distillation/warmstart")
    warm["unlabeled_manifest"] = relative(unlabeled_manifest)
    warm["unlabeled_manifest_sha256"] = unlabeled_sha
    if records is not None:
        warm["checkpoints"] = records
    if task_scales is not None:
        warm["task_scales"] = task_scales
    write_json(warm_path, warm)

    final_path = CONFIG_ROOT / "unlabeled_distillation/configs/final.json"
    final = read_json(final_path)
    final["bank_run_dir"] = relative(output_root / "unlabeled_distillation/teacher_bank")
    final["run_dir"] = relative(output_root / "unlabeled_distillation/final_student")
    final["manifest"] = relative(unlabeled_manifest)
    final["manifest_sha256"] = unlabeled_sha
    if records is not None:
        final["checkpoints"] = records
    if task_scales is not None:
        final["task_scales"] = task_scales

    warm_dir = output_root / "unlabeled_distillation/warmstart"
    warm_checkpoint = warm_dir / "stage0_task_private_lora_correction.pt"
    warm_split = warm_dir / "stage0_unlabeled_split.csv"
    final["warmstart_checkpoint"] = relative(warm_checkpoint)
    final["warmstart_split"] = relative(warm_split)
    if warm_checkpoint.is_file():
        final["warmstart_checkpoint_sha256"] = sha256_file(warm_checkpoint)
    if warm_split.is_file():
        final["warmstart_split_sha256"] = sha256_file(warm_split)
    write_json(final_path, final)

    readiness = {
        "supervised_and_usfm": True,
        "dense_fusion": all(
            (output_root / stage / f"fold{fold}" / "checkpoints/best_final_proxy.pt").is_file()
            for stage in ("supervised_convnext", "usfm_heatmap")
            for fold in range(5)
        ),
        "calibration_and_warmstart": dense_ready,
        "teacher_bank": dense_ready and warm_split.is_file(),
        "final_distillation": (
            dense_ready
            and warm_checkpoint.is_file()
            and (output_root / "unlabeled_distillation/teacher_bank/bank_summary.json").is_file()
        ),
        "export": (
            dense_ready
            and (output_root / "unlabeled_distillation/final_student/final_distillation_task_private_lora_correction.pt").is_file()
        ),
    }
    print(json.dumps({"status": "updated", "readiness": readiness}, indent=2))


if __name__ == "__main__":
    main()
