from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import model as exp148_model
from evaluation import evaluate
from protocol import (
    PROJECT_ROOT,
    DeterministicTaskUniformBatchSampler,
    Exp148Dataset,
    SequentialTaskBatchSampler,
    collate_exp148,
    load_exp148_fold,
    phase_frame,
    resolve_path,
    sha256_file,
    task_uniform_expected_draws,
    worker_init_fn,
)


DEFAULTS: dict[str, Any] = {
    "train_csv_dir": "3-data/2-working/canonical_train_csv",
    "protocol_manifest": "3-data/2-working/grouped_splits/foundus_exp148_sequence_grouped_5fold_seed42_filtered.csv",
    "protocol_manifest_sha256": "0a073a3067f70bc8f4dbcc9e9b39c7950ded76ea747204582fb7a61fa94443a7",
    "active_label_reversion_audit": "3-data/9-checksums/official_cardiac_csv_reversion_20260718.json",
    "active_label_reversion_audit_sha256": "e58fae296d822e99926bd73bb27fa68f5591a9fba6a04c01c340641c61a7f097",
    "active_label_canonical_sha256": {
        "A4C": "5815bfefad4c074a999800912db110f79920d4a78021a6d852a36d5501b22f1a",
        "PSAX": "0da63d49db79c58582d1140bdcae3324dbbc8b2c5c89e6b502c2a2258d9b3e36",
    },
    "model_variant": "control",
    "num_folds": 5,
    "fold": 0,
    "seed": 42,
    "input_size": 518,
    "heatmap_size": 64,
    "heatmap_sigma": 1.8,
    "encoder": "convnext_small.in12k_ft_in1k",
    "encoder_weights": "3-data/2-working/pretrained/timm/convnext_small.in12k_ft_in1k/model.safetensors",
    "head_hidden_channels": None,
    "batch_size": 4,
    "val_batch_size": 8,
    "num_workers": 4,
    "weight_decay": 0.01,
    "train_softargmax_beta": 4.0,
    "decode_topk": 25,
    "decode_topk_beta": 1.0,
    "coord_smooth_l1_beta": 0.02,
    "measurement_smooth_l1_beta": 0.02,
    "measurement_tasks": ["HC", "FA", "AOP"],
    "aop_loss_mode": "legacy_exp61",
    "aop_hsd_loss_weight": 1.0,
    "aop_ray_loss_weight": 0.1,
    "amp": True,
    "amp_dtype": "bfloat16",
    "grad_clip_norm": 1.0,
    "evaluate_epoch_zero": False,
    "enforce_empty_run_dir": True,
    "save_best_checkpoints": True,
    "save_diagnostic_best_checkpoints": True,
    "save_resume_checkpoint": True,
    "save_fixed_final_checkpoint": True,
    "primary_selection_metric": "final_proxy_score",
    "max_steps_per_epoch": None,
    "phases": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train active vertical-order Exp56 or Exp157 on a fixed grouped fold.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-forward", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args()


def recursive_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = recursive_update(output[key], value)
        else:
            output[key] = value
    return output


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    settings = recursive_update(DEFAULTS, payload)
    if args.run_dir:
        settings["run_dir"] = str(args.run_dir)
    if args.fold is not None:
        settings["fold"] = int(args.fold)
    if args.init_checkpoint:
        settings["init_checkpoint"] = str(args.init_checkpoint)
    if args.max_steps_per_epoch is not None:
        settings["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    suffix = ""
    if args.dry_run:
        suffix = "-dry-run"
    elif args.smoke_forward:
        suffix = "-smoke-forward"
        settings["num_workers"] = 0
        settings["max_steps_per_epoch"] = 1
    if "run_dir" not in settings:
        raise ValueError("Config must define run_dir.")
    settings["run_dir"] = str(settings["run_dir"]) + suffix
    if not settings.get("phases"):
        raise ValueError("Config must define at least one phase.")
    if str(settings.get("primary_selection_metric")) != "final_proxy_score":
        raise ValueError(
            "Exp157 primary checkpoint selection is fixed to final_proxy_score "
            "= 0.5 * task-macro MRE + 0.5 * task-macro measurement MAE."
        )
    folds = int(settings.get("num_folds", 5))
    if folds != 5 or int(settings["fold"]) not in set(range(folds)):
        raise ValueError(f"Exp157 requires fold 0..4, got fold={settings['fold']}, folds={folds}.")
    audit_hash = sha256_file(settings["active_label_reversion_audit"])
    if audit_hash != str(settings["active_label_reversion_audit_sha256"]):
        raise RuntimeError(f"Active-label reversion audit checksum mismatch: {audit_hash}")
    for task, expected_hash in settings["active_label_canonical_sha256"].items():
        path = resolve_path(settings["train_csv_dir"]) / f"{task}_train.csv"
        actual_hash = sha256_file(path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(
                f"Active-label canonical checksum mismatch for {task}: {actual_hash}"
            )
    return settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def component_hashes(network: torch.nn.Module) -> dict[str, str]:
    return {
        "model": state_hash(network),
        "encoder": state_hash(network.encoder),
        **{f"head_{task}": state_hash(head) for task, head in sorted(network.heads.items())},
    }


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def snapshot_sources(run_dir: Path, config: Path) -> None:
    destination = run_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = set(Path(__file__).resolve().parent.glob("*.py"))
    sources.update(
        {
            config.resolve(),
            PROJECT_ROOT / "2-code/56-convnext-small-heatmap/src/model.py",
            PROJECT_ROOT / "2-code/11-e2e-roialign-cascade/src/foundus_race_lib.py",
            PROJECT_ROOT / "2-code/19-dinov2-base-heatmap/src/baseline_like_lib.py",
        }
    )
    records = []
    for source in sorted(path for path in sources if path.exists()):
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


def output_manifest(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "output_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def phase_for_epoch(phases: list[dict[str, Any]], epoch: int) -> tuple[dict[str, Any], int, int]:
    cursor = 0
    for index, phase in enumerate(phases):
        length = int(phase["epochs"])
        if epoch <= cursor + length:
            return dict(phase), epoch - cursor, index
        cursor += length
    raise ValueError(f"Epoch {epoch} is outside configured phases.")


def measurement_weight(phase: dict[str, Any], phase_epoch: int) -> float:
    maximum = float(phase.get("measurement_weight", 0.0))
    warmup = int(phase.get("measurement_warmup_epochs", 0))
    ramp = int(phase.get("measurement_ramp_epochs", 1))
    if maximum <= 0.0 or phase_epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return maximum
    progress = min(1.0, max(0.0, float(phase_epoch - warmup) / float(ramp)))
    return maximum * progress


def phase_learning_rates(phase: dict[str, Any], phase_epoch: int) -> tuple[float, float]:
    epochs = max(int(phase["epochs"]), 1)
    progress = float(phase_epoch - 1) / float(epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    encoder_base = float(phase.get("encoder_lr", 0.0))
    head_base = float(phase["head_lr"])
    encoder_min = float(phase.get("encoder_min_lr", min(encoder_base, 1e-6)))
    head_min = float(phase.get("head_min_lr", min(head_base, 1e-6)))
    return (
        encoder_min + (encoder_base - encoder_min) * cosine,
        head_min + (head_base - head_min) * cosine,
    )


def apply_learning_rates(
    optimizer: torch.optim.Optimizer,
    encoder_lr: float,
    head_lr: float,
) -> None:
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        group["lr"] = float(encoder_lr if name == "encoder" else head_lr)


def build_loader(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    epoch: int,
    train: bool,
) -> tuple[DataLoader, DeterministicTaskUniformBatchSampler | None]:
    dataset = Exp148Dataset(frame, settings, augment=train, epoch=epoch)
    workers = int(settings["num_workers"])
    generator = torch.Generator().manual_seed(int(settings["seed"]) + epoch * 65_537)
    if train:
        sampler = DeterministicTaskUniformBatchSampler(
            dataset,
            int(settings["batch_size"]),
            int(settings["seed"]),
            settings.get("max_steps_per_epoch"),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_exp148,
            generator=generator,
        )
        return loader, sampler
    sampler = SequentialTaskBatchSampler(dataset, int(settings["val_batch_size"]))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_exp148,
        generator=generator,
    )
    return loader, None


def target_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "heatmap": batch["heatmap"].to(device, non_blocking=True),
        "points_norm": batch["points_norm"].to(device, non_blocking=True),
        "points_original": batch["points_original"].to(device, non_blocking=True),
    }


def load_initial_checkpoint(network: torch.nn.Module, settings: dict[str, Any]) -> dict[str, Any] | None:
    value = settings.get("init_checkpoint")
    if not value:
        return None
    path = resolve_path(str(value).format(fold=int(settings["fold"])))
    expected = str(settings.get("init_checkpoint_sha256", ""))
    actual = sha256_file(path)
    if expected and expected != actual:
        raise RuntimeError(f"Initial checkpoint SHA256 mismatch: expected={expected}, actual={actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    result = network.load_state_dict(state, strict=True)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": actual,
        "source_epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else None,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def save_checkpoint(
    path: Path,
    network: torch.nn.Module,
    settings: dict[str, Any],
    task_configs: list[dict[str, Any]],
    epoch: int,
    phase: dict[str, Any],
    phase_epoch: int,
    summary: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    include_training_state: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": network.state_dict(),
        "settings": settings,
        "task_configs": task_configs,
        "epoch": int(epoch),
        "phase": str(phase["name"]),
        "phase_epoch": int(phase_epoch),
        "validation_summary": summary,
        "training_state_included": bool(include_training_state),
    }
    if include_training_state:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            }
        )
    torch.save(payload, path)


def save_evaluation(
    run_dir: Path,
    prefix: str,
    result: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]],
) -> dict[str, float]:
    per_image, per_point, per_task, per_measurement, summary = result
    per_image.to_csv(run_dir / f"{prefix}_per_image.csv", index=False)
    per_point.to_csv(run_dir / f"{prefix}_per_point.csv", index=False)
    per_task.to_csv(run_dir / f"{prefix}_per_task.csv", index=False)
    per_measurement.to_csv(run_dir / f"{prefix}_measurements.csv", index=False)
    write_json(run_dir / f"{prefix}_summary.json", summary)
    return summary


def dry_run(
    run_dir: Path,
    settings: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    split_info: dict[str, Any],
) -> None:
    phase = dict(settings["phases"][0])
    frame = phase_frame(train, phase)
    dataset = Exp148Dataset(frame, settings, augment=True, epoch=1)
    sampler = DeterministicTaskUniformBatchSampler(
        dataset,
        int(settings["batch_size"]),
        int(settings["seed"]),
        settings.get("max_steps_per_epoch"),
    )
    list(iter(sampler))
    payload = {
        "status": "complete",
        "mode": "dry_run",
        "split": split_info,
        "phase": phase,
        "phase_rows": int(len(frame)),
        "validation_rows": int(len(validation)),
        "sampler_audit": sampler.last_audit,
        "expected_task_equivalent_passes": task_uniform_expected_draws(
            frame, int(settings["batch_size"])
        ),
        "inverse_frequency_loss_weight": False,
        "augmentation": "original Exp56 light intensity with stateless per-occurrence RNG",
        "total_epochs": int(sum(int(item["epochs"]) for item in settings["phases"])),
    }
    write_json(run_dir / "dry_run_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    run_dir = resolve_path(settings["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and bool(settings["enforce_empty_run_dir"]):
        if not args.resume_checkpoint:
            raise RuntimeError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(settings["seed"]))
    write_json(run_dir / "config.resolved.json", settings)
    write_json(run_dir / "environment.json", environment_payload())
    write_json(run_dir / "command.json", {"argv": sys.argv, "cwd": os.getcwd()})
    shutil.copy2(args.config, run_dir / "config.input.json")
    snapshot_sources(run_dir, args.config)

    train, validation, task_configs, split_info = load_exp148_fold(settings)
    input_dir = run_dir / "input_data"
    input_dir.mkdir(exist_ok=True)
    train.to_csv(input_dir / "train.csv", index=False)
    validation.to_csv(input_dir / "validation.csv", index=False)
    write_json(input_dir / "split_info.json", split_info)
    write_json(run_dir / "task_configs.json", task_configs)
    if args.dry_run:
        dry_run(run_dir, settings, train, validation, split_info)
        write_json(run_dir / "output_manifest.json", output_manifest(run_dir))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = exp148_model.build_model(settings, task_configs).to(device)
    initialization = load_initial_checkpoint(network, settings)
    write_json(
        run_dir / "model_metadata.json",
        {
            "model_name": exp148_model.MODEL_NAME,
            "model_variant": settings["model_variant"],
            "encoder_load_info": network.encoder.load_info,
            "initial_checkpoint": initialization,
        },
    )
    initial_hashes = component_hashes(network)
    write_json(run_dir / "state_hash_before.json", initial_hashes)
    validation_loader, _ = build_loader(validation, settings, epoch=0, train=False)

    if args.smoke_forward:
        phase = dict(settings["phases"][0])
        scope_audit = exp148_model.configure_trainable(network, str(phase.get("train_scope", "all")))
        exp148_model.set_train_mode(network, str(phase.get("train_scope", "all")))
        frame = phase_frame(train, phase)
        available_tasks = set(frame["task_id"].astype(str))
        if str(settings.get("model_variant")) == "psax_pair":
            task = "PSAX"
        else:
            task = "AOP" if "AOP" in available_tasks else str(frame.iloc[0]["task_id"])
        if task not in available_tasks:
            raise RuntimeError(f"Smoke task {task} is unavailable in the selected phase.")
        sample = frame[frame["task_id"].astype(str) == task].head(int(settings["batch_size"]))
        dataset = Exp148Dataset(sample, settings, augment=True, epoch=1)
        batch = collate_exp148([dataset[(index, index)] for index in range(len(dataset))])
        images = batch["image"].to(device)
        target = target_to_device(batch, device)
        smoke_settings = dict(settings)
        smoke_settings["_measurement_weight_current"] = measurement_weight(phase, 1)
        optimizer = torch.optim.AdamW(
            exp148_model.optimizer_groups(network, phase),
            weight_decay=float(settings["weight_decay"]),
        )
        output = exp148_model.forward_train(network, images, task, smoke_settings)
        loss, parts = exp148_model.compute_loss(output, target, task, phase, smoke_settings)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {
            name: float(parameter.grad.detach().float().norm().cpu())
            for name, parameter in network.named_parameters()
            if parameter.grad is not None
        }
        if not gradients or not all(math.isfinite(value) for value in gradients.values()):
            raise RuntimeError("Smoke step did not produce finite gradients.")
        optimizer.step()
        payload = {
            "status": "complete",
            "mode": "smoke_forward",
            "task": task,
            "scope": str(phase.get("train_scope", "all")),
            "scope_audit": scope_audit,
            "loss": float(loss.detach().cpu()),
            "loss_components": {
                name: float(value.detach().cpu()) for name, value in parts.items()
            },
            "gradient_parameter_count": len(gradients),
            "gradient_norm": float(math.sqrt(sum(value * value for value in gradients.values()))),
            "coords_shape": list(output["coords_norm"].shape),
            "state_hash_before": initial_hashes,
            "state_hash_after": component_hashes(network),
            "trainable_parameter_names": exp148_model.trainable_parameter_names(network),
        }
        write_json(run_dir / "smoke_forward_summary.json", payload)
        write_json(run_dir / "output_manifest.json", output_manifest(run_dir))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    total_epochs = int(sum(int(phase["epochs"]) for phase in settings["phases"]))
    start_epoch = 1
    resume_payload: dict[str, Any] | None = None
    if args.resume_checkpoint:
        resume_path = resolve_path(args.resume_checkpoint)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        network.load_state_dict(resume_payload["model_state"], strict=True)
        start_epoch = int(resume_payload["epoch"]) + 1
        random.setstate(resume_payload["python_random_state"])
        np.random.set_state(resume_payload["numpy_random_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        write_json(
            run_dir / "resume_status.json",
            {
                "checkpoint": str(resume_path),
                "checkpoint_sha256": sha256_file(resume_path),
                "start_epoch": start_epoch,
            },
        )

    if bool(settings.get("evaluate_epoch_zero", False)) and start_epoch == 1:
        initial_result = evaluate(network, validation_loader, settings, device)
        save_evaluation(run_dir, "val_epoch_000", initial_result)

    tracked = {
        "task_macro_mre_original_px": "best_mre.pt",
        "task_macro_parameter_mae_proxy": "best_measurement.pt",
        "final_proxy_score": "best_final_proxy.pt",
    }
    best = {name: float("inf") for name in tracked}
    best_records: dict[str, dict[str, Any] | None] = {name: None for name in tracked}
    metrics_path = run_dir / "metrics.csv"
    if start_epoch > 1 and metrics_path.exists():
        previous_metrics = pd.read_csv(metrics_path)
        for metric in tracked:
            if metric not in previous_metrics.columns or previous_metrics.empty:
                continue
            row = previous_metrics.loc[previous_metrics[metric].astype(float).idxmin()]
            best[metric] = float(row[metric])
            best_records[metric] = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in row.to_dict().items()
            }
    current_phase_index: int | None = None
    optimizer: torch.optim.Optimizer | None = None
    autocast_dtype = (
        torch.bfloat16 if str(settings.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(settings.get("amp", True))
        and device.type == "cuda"
        and autocast_dtype == torch.float16,
    )
    final_summary: dict[str, Any] = {}

    for epoch in range(start_epoch, total_epochs + 1):
        phase, phase_epoch, phase_index = phase_for_epoch(settings["phases"], epoch)
        scope = str(phase.get("train_scope", "all"))
        if phase_index != current_phase_index:
            scope_audit = exp148_model.configure_trainable(network, scope)
            optimizer = torch.optim.AdamW(
                exp148_model.optimizer_groups(network, phase),
                weight_decay=float(phase.get("weight_decay", settings["weight_decay"])),
            )
            if resume_payload is not None and str(resume_payload.get("phase")) == str(phase["name"]):
                optimizer.load_state_dict(resume_payload["optimizer_state"])
                resume_payload = None
            current_phase_index = phase_index
            write_json(
                run_dir / f"phase_{phase_index:02d}_trainable.json",
                {
                    "phase": phase,
                    "scope_audit": scope_audit,
                    "trainable_parameter_names": exp148_model.trainable_parameter_names(network),
                },
            )
        assert optimizer is not None
        encoder_lr, head_lr = phase_learning_rates(phase, phase_epoch)
        apply_learning_rates(optimizer, encoder_lr, head_lr)
        exp148_model.set_train_mode(network, scope)
        current_frame = phase_frame(train, phase)
        train_loader, sampler = build_loader(current_frame, settings, epoch=epoch, train=True)
        measurement_current = measurement_weight(phase, phase_epoch)
        epoch_settings = dict(settings)
        epoch_settings["_measurement_weight_current"] = measurement_current
        parts_by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        started = time.time()
        for batch in tqdm(train_loader, desc=f"[Train {epoch:03d}/{total_epochs}]", leave=False):
            task = str(batch["task_id"])
            images = batch["image"].to(device, non_blocking=True)
            target = target_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=bool(settings.get("amp", True)) and device.type == "cuda",
                dtype=autocast_dtype,
            ):
                output = exp148_model.forward_train(network, images, task, epoch_settings)
                loss, parts = exp148_model.compute_loss(
                    output, target, task, phase, epoch_settings
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip = float(settings.get("grad_clip_norm", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in network.parameters() if parameter.requires_grad],
                    clip,
                )
            scaler.step(optimizer)
            scaler.update()
            for name, value in parts.items():
                parts_by_task[task][name].append(float(value.detach().cpu()))
        if sampler is not None:
            write_json(run_dir / f"sampler_audit_epoch_{epoch:03d}.json", sampler.last_audit)

        cache_dir = run_dir / "oof_heatmaps_fixed_final" if epoch == total_epochs else None
        result = evaluate(network, validation_loader, settings, device, cache_logits_dir=cache_dir)
        per_image, per_point, per_task, per_measurement, validation_summary = result
        prefix = f"val_epoch_{epoch:03d}"
        per_task.to_csv(run_dir / f"{prefix}_per_task.csv", index=False)
        per_measurement.to_csv(run_dir / f"{prefix}_measurements.csv", index=False)
        if epoch == total_epochs:
            per_image.to_csv(run_dir / f"{prefix}_per_image.csv", index=False)
            per_point.to_csv(run_dir / f"{prefix}_per_point.csv", index=False)
        summary = {
            **validation_summary,
            "epoch": epoch,
            "phase": str(phase["name"]),
            "phase_epoch": phase_epoch,
            "train_scope": scope,
            "aop_loss_mode": str(settings["aop_loss_mode"]),
            "measurement_weight": measurement_current,
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "seconds": time.time() - started,
        }
        append_csv(run_dir / "metrics.csv", summary)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for task, values_by_name in sorted(parts_by_task.items()):
            append_csv(
                run_dir / "train_loss_by_task_epoch.csv",
                {
                    "epoch": epoch,
                    "phase": str(phase["name"]),
                    "task_id": task,
                    **{
                        name: float(np.mean(values))
                        for name, values in sorted(values_by_name.items())
                        if values
                    },
                },
            )
        for metric, filename in tracked.items():
            value = float(summary[metric])
            if value < best[metric]:
                best[metric] = value
                best_records[metric] = dict(summary)
                save_this_best = filename == "best_final_proxy.pt" or bool(
                    settings.get("save_diagnostic_best_checkpoints", True)
                )
                if bool(settings.get("save_best_checkpoints", True)) and save_this_best:
                    save_checkpoint(
                        run_dir / "checkpoints" / filename,
                        network,
                        settings,
                        task_configs,
                        epoch,
                        phase,
                        phase_epoch,
                        summary,
                        optimizer,
                    )
        if bool(settings.get("save_resume_checkpoint", True)):
            save_checkpoint(
                run_dir / "checkpoints/resume_last.pt",
                network,
                settings,
                task_configs,
                epoch,
                phase,
                phase_epoch,
                summary,
                optimizer,
                include_training_state=True,
            )
        if phase_epoch == int(phase["epochs"]) and bool(
            settings.get("save_fixed_final_checkpoint", True)
        ):
            save_checkpoint(
                run_dir / "checkpoints" / f"fixed_{str(phase['name'])}_final.pt",
                network,
                settings,
                task_configs,
                epoch,
                phase,
                phase_epoch,
                summary,
                optimizer,
            )
        final_summary = summary
        write_json(
            run_dir / "summary.json",
            {
                "status": "running" if epoch < total_epochs else "selecting_primary_checkpoint",
                "fixed_final": final_summary,
                "primary_selection": {
                    "metric": "final_proxy_score",
                    "formula": (
                        "0.5 * task_macro_mre_original_px + "
                        "0.5 * task_macro_parameter_mae_proxy"
                    ),
                    "checkpoint": "checkpoints/best_final_proxy.pt",
                },
                "best_by_metric": best_records,
                "split": split_info,
            },
        )
        print(json.dumps(summary, ensure_ascii=False))

    if not final_summary:
        raise RuntimeError("No training epoch completed.")
    final_hashes = component_hashes(network)
    write_json(run_dir / "state_hash_after_training.json", final_hashes)
    frozen_audit = {
        key: {
            "before": value,
            "after": final_hashes[key],
            "unchanged": value == final_hashes[key],
        }
        for key, value in initial_hashes.items()
    }
    write_json(run_dir / "state_hash_audit.json", frozen_audit)

    selected_path = run_dir / "checkpoints/best_final_proxy.pt"
    if not selected_path.exists():
        raise RuntimeError(f"Primary checkpoint was not saved: {selected_path}")
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    network.load_state_dict(selected_payload["model_state"], strict=True)
    selected_hashes = component_hashes(network)
    write_json(run_dir / "state_hash_selected_best_final_proxy.json", selected_hashes)
    selected_result = evaluate(
        network,
        validation_loader,
        settings,
        device,
        cache_logits_dir=run_dir / "oof_heatmaps_selected_best_final_proxy",
    )
    selected_summary = save_evaluation(
        run_dir,
        "selected_best_final_proxy",
        selected_result,
    )
    selected_checkpoint = {
        "path": str(selected_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(selected_path),
        "selection_metric": "final_proxy_score",
        "selection_formula": (
            "0.5 * task_macro_mre_original_px + "
            "0.5 * task_macro_parameter_mae_proxy"
        ),
        "source_epoch": int(selected_payload["epoch"]),
        "source_phase": str(selected_payload["phase"]),
        "source_phase_epoch": int(selected_payload["phase_epoch"]),
        "saved_validation_summary": selected_payload["validation_summary"],
        "reevaluated_summary": selected_summary,
    }
    write_json(run_dir / "selected_checkpoint.json", selected_checkpoint)
    selected_matches_saved = all(
        math.isclose(
            float(selected_summary[name]),
            float(selected_payload["validation_summary"][name]),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        for name in (
            "task_macro_mre_original_px",
            "task_macro_parameter_mae_proxy",
            "final_proxy_score",
        )
    )
    write_json(
        run_dir / "summary.json",
        {
            "status": "complete",
            "primary_selection": {
                "metric": "final_proxy_score",
                "formula": (
                    "0.5 * task_macro_mre_original_px + "
                    "0.5 * task_macro_parameter_mae_proxy"
                ),
                "checkpoint": "checkpoints/best_final_proxy.pt",
            },
            "selected_primary": selected_checkpoint,
            "fixed_final": final_summary,
            "best_by_metric": best_records,
            "diagnostic_only_checkpoints": [
                "checkpoints/best_mre.pt",
                "checkpoints/best_measurement.pt",
                f"checkpoints/fixed_{str(settings['phases'][-1]['name'])}_final.pt",
            ],
            "split": split_info,
        },
    )
    write_json(
        run_dir / "qc_summary.json",
        {
            "status": "pass" if selected_matches_saved else "fail",
            "completed_epochs": total_epochs,
            "fixed_epoch_result_finite": all(
                math.isfinite(float(final_summary[name]))
                for name in (
                    "task_macro_mre_original_px",
                    "task_macro_parameter_mae_proxy",
                    "final_proxy_score",
                )
            ),
            "selected_checkpoint": "checkpoints/best_final_proxy.pt",
            "selected_checkpoint_exists": selected_path.exists(),
            "selected_checkpoint_reevaluation_matches_saved": selected_matches_saved,
            "selected_result_finite": all(
                math.isfinite(float(selected_summary[name]))
                for name in (
                    "task_macro_mre_original_px",
                    "task_macro_parameter_mae_proxy",
                    "final_proxy_score",
                )
            ),
            "cross_fold_groups": int(split_info["cross_fold_groups"]),
            "femur_excluded_rows": int(split_info["femur_excluded_rows"]),
            "split_screen_training_only_rows": int(split_info["split_screen_training_only_rows"]),
            "patient_group_safe": False,
            "device_group_safe": False,
            "aop_loss_mode": str(settings["aop_loss_mode"]),
        },
    )
    write_json(run_dir / "output_manifest.json", output_manifest(run_dir))


if __name__ == "__main__":
    run()
