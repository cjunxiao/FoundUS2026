from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from data import (
    PROJECT_ROOT,
    Exp164LabeledDataset,
    FixedPlanBatchSampler,
    SequentialTaskBatchSampler,
    StyleDonorDataset,
    build_exhaustive_schedule,
    build_labeled_plans,
    collate_donors,
    collate_labeled,
    content_masks_from_letterbox,
    load_fold,
    load_unlabeled_manifest,
    partition_coverage,
    resolve_path,
    worker_init_fn,
    write_schedule_assignments,
)
from style_transfer import bounded_masked_moment_transfer


EXP161_EVALUATION_PATH = (
    PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/evaluation.py"
)
DEFAULTS: dict[str, Any] = {
    "branch": "B0",
    "style_donor": "labeled",
    "epochs": 8,
    "batch_size": 4,
    "unlabeled_batch_size": 16,
    "val_batch_size": 8,
    "num_workers": 4,
    "donor_num_workers": 4,
    "adapter_bottleneck": 64,
    "adapter_groups": 8,
    "adapter_learning_rate": 1e-4,
    "adapter_min_learning_rate": 1e-6,
    "weight_decay": 1e-3,
    "lambda_style_supervised": 1.0,
    "lambda_anchor_preservation": 0.25,
    "style_alpha_min": 0.25,
    "style_alpha_max": 0.60,
    "style_mean_shift_limit": 0.15,
    "style_std_ratio_min": 0.60,
    "style_std_ratio_max": 1.60,
    "style_method": "bounded_masked_channel_moments",
    "fourier_transfer_enabled": False,
    "ssl_warmup_epochs": 0,
    "unlabeled_tasks": ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX"],
    "supervised_only_tasks": ["fetal_femur"],
    "supervised_only_task_weight": 1.0,
    "unlabeled_task_balance_power": 0.0,
    "unlabeled_task_weight_min": 1.0,
    "unlabeled_task_weight_max": 1.0,
    "unlabeled_manifest": "4-runs/2-data-template-audit/training_dataset_distribution/training_release_unlabeled_dedup_manifest.csv",
    "unlabeled_manifest_sha256": "a1a552dd7d9a853b46f25f07414f1b618d9db5ada77ed03777936003ed4bc844",
    "unlabeled_summary": "4-runs/2-data-template-audit/training_dataset_distribution/training_dataset_distribution_summary.json",
    "bad_image_paths": ["3-data/1-extracted/Data/FA/unlabeled/FAall/FAall/27188.png"],
    "require_foundus_unlabeled_path_marker": True,
    "require_full_unlabeled_coverage": True,
    "amp": True,
    "amp_dtype": "bfloat16",
    "grad_clip_norm": 1.0,
    "seed": 42,
    "max_steps_per_epoch": None,
    "save_every_epoch": True,
    "enforce_empty_run_dir": True,
    "primary_evaluation_space": "official_vertical",
}


def _load_evaluation():
    name = "exp161_evaluation_for_exp164"
    spec = importlib.util.spec_from_file_location(name, EXP161_EVALUATION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(EXP161_EVALUATION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load_evaluation()


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp164 appearance replay Adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-forward", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(modules: dict[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(
                f"{module_name}.{name}|{value.dtype}|{tuple(value.shape)}".encode()
            )
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_settings(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = resolve_path(args.config)
    config = read_json(config_path)
    anchor_path = resolve_path(config["anchor_checkpoint"])
    actual_sha = sha256_file(anchor_path)
    if actual_sha != str(config["anchor_checkpoint_sha256"]):
        raise RuntimeError(
            f"Anchor SHA256 mismatch: expected={config['anchor_checkpoint_sha256']}, actual={actual_sha}"
        )
    try:
        payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(anchor_path, map_location="cpu")
    settings = dict(payload["settings"])
    settings.update(DEFAULTS)
    settings.update(config)
    for key in ("run_dir", "anchor_checkpoint", "unlabeled_manifest", "unlabeled_summary"):
        settings[key] = str(resolve_path(settings[key]))
    settings["bad_image_paths"] = [str(resolve_path(value)) for value in settings["bad_image_paths"]]
    if args.run_dir:
        settings["run_dir"] = str(resolve_path(args.run_dir))
    if args.epochs is not None:
        settings["epochs"] = int(args.epochs)
    if args.max_steps_per_epoch is not None:
        settings["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    if args.dry_run:
        settings["run_dir"] += "-dry-run"
    if args.smoke_forward:
        settings["run_dir"] += "-smoke-forward"
        settings["epochs"] = 1
        settings["max_steps_per_epoch"] = 2
        settings["require_full_unlabeled_coverage"] = False
    if str(settings["branch"]) not in {"B0", "B1"}:
        raise ValueError("Exp164 branch must be B0 or B1.")
    expected_donor = "labeled" if settings["branch"] == "B0" else "unlabeled"
    if str(settings["style_donor"]) != expected_donor:
        raise ValueError(
            f"Branch {settings['branch']} requires style_donor={expected_donor}."
        )
    if bool(settings.get("fourier_transfer_enabled", False)):
        raise ValueError("The first Exp164 screen isolates bounded moment transfer; Fourier is disabled.")
    return settings, payload


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def output_manifest(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "output_manifest.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def snapshot_sources(run_dir: Path, config_path: Path) -> None:
    destination = run_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = set(Path(__file__).resolve().parent.glob("*.py"))
    sources.update(
        {
            config_path,
            PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/model.py",
            PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/evaluation.py",
            PROJECT_ROOT / "2-code/161-stable-internal-exp152/src/protocol.py",
            PROJECT_ROOT / "2-code/159-canonical-internal-exp152/src/model.py",
            PROJECT_ROOT / "2-code/152-exp56-psax-pair-decoder/src/model.py",
            PROJECT_ROOT / "2-code/157-active-vertical-exp152-5fold/src/protocol.py",
            PROJECT_ROOT / "2-code/116-exp56-ssl-bridge-exp61/src/bridge_protocol.py",
            PROJECT_ROOT / "2-code/56-convnext-small-heatmap/src/model.py",
            PROJECT_ROOT / "2-code/11-e2e-roialign-cascade/src/foundus_race_lib.py",
        }
    )
    rows = []
    for source in sorted(path for path in sources if path.exists()):
        relative = source.relative_to(PROJECT_ROOT)
        target = destination / "__".join(relative.parts)
        shutil.copy2(source, target)
        rows.append(
            {
                "source": str(relative),
                "snapshot": str(target.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    write_json(run_dir / "source_status.json", rows)


def load_anchor(
    network: model.Exp164Model,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = network.load_state_dict(payload["model_state"], strict=False)
    invalid_missing = [
        key for key in result.missing_keys if not key.startswith("task_adapters.")
    ]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Anchor load mismatch: missing={invalid_missing}, unexpected={result.unexpected_keys}"
        )
    return {
        "source_epoch": int(payload["epoch"]),
        "source_phase": str(payload["phase"]),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def make_validation_loader(
    validation: pd.DataFrame,
    settings: dict[str, Any],
) -> DataLoader:
    dataset = Exp164LabeledDataset(validation, settings, augment=False, epoch=0)
    return DataLoader(
        dataset,
        batch_sampler=SequentialTaskBatchSampler(dataset, int(settings["val_batch_size"])),
        num_workers=int(settings["num_workers"]),
        pin_memory=True,
        collate_fn=collate_labeled,
        worker_init_fn=worker_init_fn,
    )


def evaluate_and_save(
    network: model.Exp164Model,
    loader: DataLoader,
    settings: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    epoch: int,
    scale: float,
) -> dict[str, float]:
    settings["_evaluation_adapter_scale"] = float(scale)
    try:
        per_image, per_point, per_task, per_measurement, summary = evaluation.evaluate(
            network, loader, settings, device
        )
    finally:
        settings.pop("_evaluation_adapter_scale", None)
    tag = f"{float(scale):.2f}".replace(".", "p")
    per_image.to_csv(run_dir / f"val_per_image_epoch_{epoch:03d}_scale_{tag}.csv", index=False)
    per_point.to_csv(run_dir / f"val_per_point_epoch_{epoch:03d}_scale_{tag}.csv", index=False)
    per_task.to_csv(run_dir / f"val_per_task_epoch_{epoch:03d}_scale_{tag}.csv", index=False)
    per_measurement.to_csv(
        run_dir / f"val_per_measurement_epoch_{epoch:03d}_scale_{tag}.csv", index=False
    )
    write_json(run_dir / f"val_summary_epoch_{epoch:03d}_scale_{tag}.json", summary)
    return summary


def save_adapter_checkpoint(
    path: Path,
    network: model.Exp164Model,
    settings: dict[str, Any],
    task_configs: list[dict[str, Any]],
    epoch: int,
    metrics: dict[str, Any],
    anchor_state_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task_adapter_state": network.task_adapters.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "settings": settings,
            "task_configs": task_configs,
            "anchor_checkpoint_sha256": settings["anchor_checkpoint_sha256"],
            "anchor_encoder_head_state_sha256": anchor_state_sha256,
            "adapter_architecture": {
                "type": "independent_task_residual_adapters",
                "bottleneck": int(settings["adapter_bottleneck"]),
                "groups": int(settings["adapter_groups"]),
                "zero_initialized_up_projection": True,
            },
        },
        path,
    )


def target_from_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "heatmap": batch["heatmap"].to(device, non_blocking=True),
        "points_norm": batch["points_norm"].to(device, non_blocking=True),
        "points_original": batch["points_original"].to(device, non_blocking=True),
    }


def inactive_gradient_audit(network: model.Exp164Model, task_id: str) -> None:
    for other_task, adapter in network.task_adapters.items():
        if other_task == str(task_id):
            continue
        if any(parameter.grad is not None for parameter in adapter.parameters()):
            raise RuntimeError(
                f"Inactive Adapter {other_task} received a gradient during task {task_id}."
            )


@torch.no_grad()
def fallback_logits_audit(
    network: model.Exp164Model,
    train: pd.DataFrame,
    settings: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    network.eval()
    rows = []
    for task, frame in train.groupby("task_id", sort=True):
        sample = Exp164LabeledDataset(frame.head(1), settings, augment=False, epoch=0)[0]
        image = sample["image"].unsqueeze(0).to(device)
        anchor = network(image, str(task), adapter_enabled=False)["heatmap_logits"]
        fallback = network(image, str(task), adapter_enabled=True, adapter_scale=0.0)[
            "heatmap_logits"
        ]
        rows.append(
            {
                "task_id": str(task),
                "max_abs_logit_difference": float((anchor - fallback).abs().max().cpu()),
            }
        )
    return rows


def run(args: argparse.Namespace, settings: dict[str, Any], anchor_payload: dict[str, Any]) -> None:
    run_dir = Path(settings["run_dir"])
    config_path = resolve_path(args.config)
    set_seed(int(settings["seed"]))
    train, validation, task_configs, split_info = load_fold(settings)
    checkpoint_tasks = [
        (str(item["task_id"]), int(item["num_classes"]))
        for item in anchor_payload["task_configs"]
    ]
    current_tasks = [
        (str(item["task_id"]), int(item["num_classes"])) for item in task_configs
    ]
    if checkpoint_tasks != current_tasks:
        raise RuntimeError(
            f"Task config mismatch: checkpoint={checkpoint_tasks}, fold={current_tasks}"
        )
    unlabeled, unlabeled_info = load_unlabeled_manifest(settings)
    schedule, schedule_info = build_exhaustive_schedule(
        unlabeled, settings, labeled_frame=train
    )
    assignment_info = write_schedule_assignments(
        unlabeled,
        schedule,
        run_dir / "input_data/unlabeled_schedule.csv.gz",
        int(settings["seed"]),
    )
    train.to_csv(run_dir / "input_data/train.csv", index=False)
    validation.to_csv(run_dir / "input_data/validation.csv", index=False)
    write_json(run_dir / "input_data/split_info.json", split_info)
    write_json(run_dir / "input_data/unlabeled_info.json", unlabeled_info)
    write_json(
        run_dir / "input_data/schedule_info.json", {**schedule_info, **assignment_info}
    )
    write_json(run_dir / "resolved_config.json", settings)
    write_json(
        run_dir / "command.json",
        {
            "argv": [sys.executable, *sys.argv],
            "working_directory": str(PROJECT_ROOT),
        },
    )
    write_json(run_dir / "environment.json", environment_payload())
    snapshot_sources(run_dir, config_path)
    print(json.dumps({"split": split_info, "unlabeled": unlabeled_info, "schedule": schedule_info}, indent=2))
    if args.dry_run:
        write_json(run_dir / "run_summary.json", {"status": "dry_run_complete"})
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Exp164 training requires CUDA.")

    device = torch.device("cuda")
    network = model.build_model(settings, task_configs).to(device)
    load_info = load_anchor(network, anchor_payload)
    model.train_adapter_only(network)
    anchor_state_before = state_sha256(
        {"encoder": network.encoder, "heads": network.heads}
    )
    write_json(run_dir / "checkpoint_load_info.json", load_info)
    fallback_before = fallback_logits_audit(network, train, settings, device)
    pd.DataFrame(fallback_before).to_csv(run_dir / "fallback_before.csv", index=False)
    if max(row["max_abs_logit_difference"] for row in fallback_before) > 1e-7:
        raise RuntimeError("Initial scale-zero fallback failed.")

    validation_loader = make_validation_loader(validation, settings)
    anchor_summary = evaluate_and_save(
        network, validation_loader, settings, device, run_dir, 0, 0.0
    )
    initial_summary = evaluate_and_save(
        network, validation_loader, settings, device, run_dir, 0, 1.0
    )
    if abs(
        float(anchor_summary["official_task_macro_mre_original_px"])
        - float(initial_summary["official_task_macro_mre_original_px"])
    ) > 1e-9:
        raise RuntimeError("Zero-initialized Adapter does not reproduce the anchor metric.")

    parameters = model.adapter_parameters(network)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings["adapter_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(settings["epochs"]), 1),
        eta_min=float(settings["adapter_min_learning_rate"]),
    )
    amp_enabled = bool(settings["amp"])
    amp_dtype = torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    supervised_only = {str(value) for value in settings["supervised_only_tasks"]}
    seen_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    draws_by_task: dict[str, int] = defaultdict(int)
    schedule_audits: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    best = {"epoch": 0, **initial_summary}
    stop_after_step = bool(args.smoke_forward)

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train_adapter_only(network)
        epoch_specs = schedule[epoch]
        content_batches, labeled_donor_batches, plan_audit = build_labeled_plans(
            train,
            epoch_specs,
            int(settings["batch_size"]),
            int(settings["seed"]),
            epoch,
            supervised_only,
        )
        style_digest = hashlib.sha256()
        content_dataset = Exp164LabeledDataset(train, settings, augment=True, epoch=epoch)
        content_loader = DataLoader(
            content_dataset,
            batch_sampler=FixedPlanBatchSampler(content_batches),
            num_workers=int(settings["num_workers"]),
            pin_memory=True,
            collate_fn=collate_labeled,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(int(settings["seed"]) + epoch * 3011),
        )
        active_specs = [
            spec
            for spec in epoch_specs
            if spec.indices and str(spec.task_id) not in supervised_only
        ]
        if str(settings["style_donor"]) == "unlabeled":
            donor_dataset = StyleDonorDataset(unlabeled, int(settings["input_size"]), "canonical_id")
            donor_batches = [list(spec.indices) for spec in active_specs]
        else:
            donor_dataset = StyleDonorDataset(train, int(settings["input_size"]), "row_id_current")
            donor_batches = labeled_donor_batches
        donor_loader = DataLoader(
            donor_dataset,
            batch_sampler=FixedPlanBatchSampler(donor_batches),
            num_workers=int(settings["donor_num_workers"]),
            pin_memory=True,
            collate_fn=collate_donors,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(int(settings["seed"]) + epoch * 4001),
        )
        donor_iterator = iter(donor_loader)
        start = time.time()
        for step, (spec, labeled) in enumerate(
            tqdm(
                zip(epoch_specs, content_loader),
                total=len(epoch_specs),
                desc=f"Exp164 {settings['branch']} epoch {epoch}/{settings['epochs']}",
            ),
            start=1,
        ):
            task = str(spec.task_id)
            if str(labeled["task_id"]) != task:
                raise RuntimeError("Labeled task schedule mismatch.")
            images = labeled["image"].to(device, non_blocking=True)
            target = target_from_batch(labeled, device)
            active_style = bool(spec.indices) and task not in supervised_only
            donor = next(donor_iterator) if active_style else None
            if donor is not None and str(donor["task_id"]) != task:
                raise RuntimeError("Same-task style donor contract violated.")
            rng = np.random.RandomState(
                int(settings["seed"]) + epoch * 1_000_003 + step * 97
            )
            alpha = float(
                rng.uniform(settings["style_alpha_min"], settings["style_alpha_max"])
            ) if active_style else 0.0
            style_digest.update(f"{epoch}|{step}|{task}|{alpha:.12f}\n".encode("utf-8"))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                clean_output, anchor_output = network.forward_with_anchor(
                    images, task, adapter_scale=1.0
                )
                clean_loss, clean_parts = model.compute_supervised_loss(
                    clean_output, target, task, settings
                )
                anchor_loss = F.mse_loss(
                    torch.sigmoid(clean_output["heatmap_logits"]),
                    torch.sigmoid(anchor_output["heatmap_logits"]).detach(),
                )
                style_loss = clean_loss.new_zeros(())
                transfer_audit: dict[str, Any] = {
                    "donor_count": 0,
                    "all_donors_contributed": True,
                    "mean_abs_shift": 0.0,
                    "mean_std_ratio": 1.0,
                }
                if donor is not None:
                    donor_images = donor["image"].to(device, non_blocking=True)
                    donor_masks = donor["content_mask"].to(device, non_blocking=True)
                    content_masks = content_masks_from_letterbox(
                        labeled["letterbox"], int(settings["input_size"]), device
                    )
                    styled, transfer_audit = bounded_masked_moment_transfer(
                        images,
                        donor_images,
                        content_masks,
                        donor_masks,
                        alpha,
                        float(settings["style_mean_shift_limit"]),
                        float(settings["style_std_ratio_min"]),
                        float(settings["style_std_ratio_max"]),
                    )
                    if not bool(transfer_audit["all_donors_contributed"]):
                        raise RuntimeError("At least one loaded donor did not enter style statistics.")
                    styled_output = network(
                        styled, task_id=task, adapter_enabled=True, adapter_scale=1.0
                    )
                    style_loss, _ = model.compute_supervised_loss(
                        styled_output, target, task, settings
                    )
                total = (
                    clean_loss
                    + float(settings["lambda_style_supervised"]) * style_loss
                    + float(settings["lambda_anchor_preservation"]) * anchor_loss
                )
            total.backward()
            inactive_gradient_audit(network, task)
            if float(settings["grad_clip_norm"]) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(settings["grad_clip_norm"])
                )
            optimizer.step()

            if donor is not None and str(settings["style_donor"]) == "unlabeled":
                identifiers = [str(value) for value in donor["donor_id"]]
                seen_counts.update(identifiers)
                effective_counts.update(identifiers)
                draws_by_task[task] += len(identifiers)
            row = {
                "epoch": epoch,
                "step": step,
                "task_id": task,
                "style_active": active_style,
                "style_alpha": alpha,
                "clean_supervised_loss": float(clean_loss.detach().cpu()),
                "styled_supervised_loss": float(style_loss.detach().cpu()),
                "anchor_preservation_loss": float(anchor_loss.detach().cpu()),
                "total_loss": float(total.detach().cpu()),
                "heatmap_loss": float(clean_parts["heatmap_loss"].detach().cpu()),
                "donor_count": int(transfer_audit["donor_count"]),
                "style_mean_abs_shift": float(transfer_audit["mean_abs_shift"]),
                "style_mean_std_ratio": float(transfer_audit["mean_std_ratio"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history_rows.append(row)
            if stop_after_step:
                break
        if not stop_after_step:
            try:
                next(donor_iterator)
                raise RuntimeError("Style donor loader contains unexpected extra batches.")
            except StopIteration:
                pass
        plan_audit["style_parameter_schedule_sha256"] = style_digest.hexdigest()
        plan_audit["seconds"] = float(time.time() - start)
        schedule_audits.append(plan_audit)
        scheduler.step()
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
        write_json(run_dir / "labeled_schedule_audit.json", schedule_audits)

        summary = evaluate_and_save(
            network, validation_loader, settings, device, run_dir, epoch, 1.0
        )
        record = {"epoch": epoch, **summary}
        if (
            float(record["official_final_proxy_score"]), epoch
        ) < (
            float(best["official_final_proxy_score"]), int(best["epoch"])
        ):
            best = record
            save_adapter_checkpoint(
                run_dir / "checkpoints/best_scale1_final_proxy.pt",
                network,
                settings,
                task_configs,
                epoch,
                summary,
                anchor_state_before,
            )
        save_adapter_checkpoint(
            run_dir / "checkpoints/last.pt",
            network,
            settings,
            task_configs,
            epoch,
            summary,
            anchor_state_before,
        )
        if bool(settings["save_every_epoch"]):
            save_adapter_checkpoint(
                run_dir / f"checkpoints/epoch_{epoch:03d}.pt",
                network,
                settings,
                task_configs,
                epoch,
                summary,
                anchor_state_before,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "official_macro_mre": summary["official_task_macro_mre_original_px"],
                    "official_final_proxy": summary["official_final_proxy_score"],
                    "best_epoch": best["epoch"],
                    "best_official_final_proxy": best["official_final_proxy_score"],
                }
            )
        )
        if stop_after_step:
            break

    coverage = partition_coverage(unlabeled, seen_counts, draws_by_task)
    coverage["unlabeled_effective_style_unique"] = int(
        sum(1 for count in effective_counts.values() if count > 0)
    )
    coverage["unlabeled_effective_style_draws"] = int(sum(effective_counts.values()))
    coverage["unlabeled_effective_style_repeated_unique"] = int(
        sum(1 for count in effective_counts.values() if count > 1)
    )
    coverage["style_donor_source"] = str(settings["style_donor"])
    write_json(run_dir / "unlabeled_coverage.json", coverage)
    if (
        str(settings["style_donor"]) == "unlabeled"
        and bool(settings["require_full_unlabeled_coverage"])
        and not stop_after_step
    ):
        expected = int(coverage["unlabeled_total_unique"])
        if not (
            int(coverage["unlabeled_seen_unique"]) == expected
            and int(coverage["unlabeled_total_draws"]) == expected
            and int(coverage["unlabeled_repeated_unique"]) == 0
            and int(coverage["unlabeled_effective_style_unique"]) == expected
            and int(coverage["unlabeled_effective_style_draws"]) == expected
            and int(coverage["unlabeled_effective_style_repeated_unique"]) == 0
        ):
            raise RuntimeError(f"Effective exhaustive style coverage failed: {coverage}")

    anchor_state_after = state_sha256(
        {"encoder": network.encoder, "heads": network.heads}
    )
    if anchor_state_after != anchor_state_before:
        raise RuntimeError("Frozen Exp161 encoder/head state changed.")
    fallback_after = fallback_logits_audit(network, train, settings, device)
    pd.DataFrame(fallback_after).to_csv(run_dir / "fallback_after.csv", index=False)
    fallback_max = max(row["max_abs_logit_difference"] for row in fallback_after)
    if fallback_max > 1e-7:
        raise RuntimeError("Final scale-zero fallback failed.")
    final_epoch = 1 if stop_after_step else int(settings["epochs"])
    final_summary = read_json(
        run_dir / f"val_summary_epoch_{final_epoch:03d}_scale_1p00.json"
    )
    final_scale_zero = evaluate_and_save(
        network,
        validation_loader,
        settings,
        device,
        run_dir,
        final_epoch,
        0.0,
    )
    anchor_metric_diff = abs(
        float(final_scale_zero["official_task_macro_mre_original_px"])
        - float(anchor_summary["official_task_macro_mre_original_px"])
    )
    if anchor_metric_diff > 1e-9:
        raise RuntimeError("Final scale-zero metric does not reproduce Exp161 anchor.")
    qc = {
        "status": "pass",
        "anchor_encoder_head_state_unchanged": anchor_state_before == anchor_state_after,
        "anchor_state_sha256_before": anchor_state_before,
        "anchor_state_sha256_after": anchor_state_after,
        "scale_zero_fallback_max_abs_logit_difference": fallback_max,
        "scale_zero_anchor_macro_mre_abs_difference": anchor_metric_diff,
        "inactive_adapter_gradient_contract": "checked_every_step",
        "full_unlabeled_coverage_required": bool(
            settings["require_full_unlabeled_coverage"]
            and settings["style_donor"] == "unlabeled"
        ),
        "unlabeled_coverage": coverage,
    }
    write_json(run_dir / "qc_summary.json", qc)
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "smoke_complete" if stop_after_step else "complete",
            "branch": settings["branch"],
            "primary_protocol": "fixed final epoch, adapter scale 1.0, official endpoint space",
            "anchor_epoch0_scale0": anchor_summary,
            "final_epoch_scale1": final_summary,
            "best_diagnostic_epoch_scale1": best,
            "unlabeled_coverage": coverage,
            "qc": qc,
        },
    )


def main() -> None:
    args = parse_args()
    settings, payload = resolve_settings(args)
    run_dir = Path(settings["run_dir"])
    if bool(settings["enforce_empty_run_dir"]) and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_name = "dry_run.log" if args.dry_run else (
        "smoke_forward.log" if args.smoke_forward else "train.log"
    )
    with (run_dir / log_name).open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(
            Tee(sys.stderr, handle)
        ):
            run(args, settings, payload)
    write_json(run_dir / "output_manifest.json", output_manifest(run_dir))


if __name__ == "__main__":
    main()
