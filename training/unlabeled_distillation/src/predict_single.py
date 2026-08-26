from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
import model as exp301_model


EVAL_SRC = common.PROJECT_ROOT / "2-code/4-eval-audit-tools/src"
RACE_SRC = common.PROJECT_ROOT / "2-code/11-e2e-roialign-cascade/src"
for directory in (EVAL_SRC, RACE_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from foundus_race_lib import canvas_norm_to_original_pixels  # noqa: E402
from predict_official_val_generic import (  # noqa: E402
    ValManifestLetterboxDataset,
    collate_fn,
    make_submission_zip,
    prediction_item,
    run_qc,
    validate_task_configs_against_manifest,
)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(common.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def file_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": common.sha256_file(path),
    }


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--val-manifest",
        default="3-data/2-working/canonical_val_manifest/val_manifest.csv",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    config_path = common.resolve(args.config)
    settings = json.loads(config_path.read_text())
    producing_run = common.resolve(settings["run_dir"])
    training = json.loads((producing_run / "metrics_summary.json").read_text())
    stability = json.loads(
        (producing_run / "unlabeled_stability_summary.json").read_text()
    )
    if training.get("status") != "pass" or stability.get("status") != "pass":
        raise RuntimeError("Exp301 training and stability gates must both pass.")
    checkpoint = producing_run / "exp301_task_private_lora_correction.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = common.resolve(args.run_dir)
    submission_dir = common.resolve(args.submission_dir)
    require_new(run_dir)
    require_new(submission_dir)
    prediction_dir = run_dir / "predictions"
    qc_dir = run_dir / "qc"
    prediction_dir.mkdir()
    qc_dir.mkdir()

    exp290, exp191 = common.load_dependencies()
    payloads, teacher_records = exp290.load_payloads(settings)
    medoid_index = next(
        index
        for index, item in enumerate(settings["checkpoints"])
        if int(item["fold"]) == int(settings["medoid_fold"])
    )
    payload = payloads[medoid_index]
    task_configs = payload["task_configs"]
    model_settings = payload["settings"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor = exp290.build_model(exp191, payload, payload["model_state"], device)
    del payloads
    network = exp301_model.DecoupledTaskPrivateFoundationCorrection(
        anchor,
        task_configs,
        settings["task_scales"],
        shared_channels=int(model_settings.get("fusion_shared_channels", 128)),
        hidden_channels=int(settings["residual_hidden_channels"]),
        logit_bound=float(settings["residual_logit_bound"]),
        lora_last_blocks=int(settings["lora_last_blocks"]),
        qkv_lora_rank=int(settings["qkv_lora_rank"]),
        qkv_lora_alpha=float(settings["qkv_lora_alpha"]),
        projection_lora_rank=int(settings["projection_lora_rank"]),
        projection_lora_alpha=float(settings["projection_lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
    ).to(device)
    del payload
    final_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = final_payload["trainable_state"]
    named = dict(network.anchor.named_parameters())
    for name, value in state["task_private_lora"].items():
        if name not in named:
            raise RuntimeError(f"Exp301 LoRA key is absent: {name}")
        named[name].data.copy_(value.to(named[name].dtype))
    result = network.residual_heads.load_state_dict(
        state["task_private_correction"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Exp301 correction checkpoint did not load strictly.")
    network.eval()

    val_manifest = common.resolve(args.val_manifest)
    dataset = ValManifestLetterboxDataset(val_manifest, int(settings["input_size"]))
    validate_task_configs_against_manifest(task_configs, dataset.dataframe)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    amp_dtype = (
        torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    )
    predictions: list[dict | None] = [None] * len(dataset)
    task_counts: dict[str, int] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Exp301 single-model official-val prediction"):
            images = batch["image"].to(device, non_blocking=True)
            for task_id in sorted(set(batch["task_id"])):
                indices = [
                    index
                    for index, value in enumerate(batch["task_id"])
                    if value == task_id
                ]
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=bool(settings.get("amp", True) and device.type == "cuda"),
                ):
                    output = network(images[indices], str(task_id))
                internal = exp191.exp184.decode_topk(
                    output["heatmap_logits"].float(),
                    int(settings["decode_topk"]),
                    float(settings["decode_topk_beta"]),
                )
                official = exp191.sort_official_vertical(
                    internal, str(task_id), model_settings
                )
                original = canvas_norm_to_original_pixels(
                    official,
                    [batch["letterbox"][index] for index in indices],
                    int(settings["input_size"]),
                )
                for local_index, batch_index in enumerate(indices):
                    row_index = int(batch["index"][batch_index])
                    predictions[row_index] = prediction_item(
                        original[local_index].cpu().numpy(),
                        batch["image_path"][batch_index],
                        str(task_id),
                        int(batch["width"][batch_index]),
                        int(batch["height"][batch_index]),
                    )
                    task_counts[str(task_id)] = task_counts.get(str(task_id), 0) + 1
    if any(value is None for value in predictions):
        raise RuntimeError("Exp301 produced missing official-validation predictions.")
    prediction_json = prediction_dir / "regression_predictions.json"
    common.write_json(prediction_json, predictions)
    submission_zip = make_submission_zip(prediction_json, submission_dir, None)
    args.skip_qc = False
    args.qc_out_dir = str(qc_dir)
    args.out_dir = str(prediction_dir)
    args.val_manifest = str(val_manifest)
    run_qc(args, prediction_json, submission_zip)
    qc = json.loads((qc_dir / "submission_qc_report.json").read_text())
    compatibility = json.loads(
        (qc_dir / "submission_compatibility_report.json").read_text()
    )
    if qc.get("status") != "pass" or compatibility.get("status") != "pass":
        raise RuntimeError("Exp301 official submission QC failed.")
    submission_json = submission_dir / "regression_predictions.json"
    with zipfile.ZipFile(submission_zip) as archive:
        if archive.namelist() != ["regression_predictions.json"]:
            raise RuntimeError("Submission ZIP layout is invalid.")
        if archive.read("regression_predictions.json") != submission_json.read_bytes():
            raise RuntimeError("Submission ZIP payload differs from its JSON.")

    common.write_json(
        run_dir / "command.json",
        {"cwd": str(common.PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
    )
    common.write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    source_dir = run_dir / "source_snapshot"
    source_dir.mkdir()
    source_records = []
    for source in sorted((common.EXPERIMENT_DIR / "src").glob("*.py")):
        target = source_dir / source.name
        shutil.copy2(source, target)
        source_records.append({**file_record(source), "snapshot": relative(target)})
    common.write_json(run_dir / "source_status.json", source_records)
    method = (
        "single frozen Exp205 medoid graph with task-private USFM rank-4 LoRA "
        "and contextual heatmap correction distilled from five teachers on "
        "182,870 official challenge-unlabeled ultrasound images"
    )
    manifest = {
        "status": "ready_not_uploaded",
        "producing_experiment": "301-full-official-unlabeled-ensemble-distillation",
        "producing_run": relative(producing_run),
        "method": method,
        "model_count": 1,
        "checkpoint": file_record(checkpoint),
        "teacher_models": teacher_records,
        "prediction_json": file_record(prediction_json),
        "submission_json": file_record(submission_json),
        "submission_zip": file_record(submission_zip),
        "validation_manifest": file_record(val_manifest),
        "validation_rows": int(len(dataset)),
        "task_counts": task_counts,
        "training_gate": training["gates"],
        "stability_gate": stability["gates"],
        "qc_status": "pass",
        "official_validation_labels_used": False,
        "upload_result": "not_uploaded",
    }
    common.write_json(run_dir / "submission_manifest.json", manifest)
    common.write_json(submission_dir / "submission_manifest.json", manifest)
    common.output_manifest(run_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
