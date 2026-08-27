from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import common
import model as final_distillation_model


class BankTargetDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        task: str,
        input_size: int,
        mean_path: Path,
        dispersion_path: Path,
        teacher_runtime: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task = str(task)
        self.base = teacher_runtime.UnlabeledDataset(self.frame, int(input_size))
        self.mean_path = mean_path
        self.dispersion_path = dispersion_path
        self._mean = None
        self._dispersion = None

    def __len__(self) -> int:
        return len(self.frame)

    def _banks(self):
        if self._mean is None:
            self._mean = np.load(self.mean_path, mmap_mode="r")
            self._dispersion = np.load(self.dispersion_path, mmap_mode="r")
        return self._mean, self._dispersion

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[int(index)]
        row = self.frame.iloc[int(index)]
        position = int(row["task_position"])
        mean, dispersion = self._banks()
        item["teacher_probability"] = torch.from_numpy(
            np.asarray(mean[position], dtype=np.float32)
        )
        item["teacher_dispersion"] = torch.from_numpy(
            np.asarray(dispersion[position], dtype=np.float32)
        )
        return item


def collate_bank(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {str(item["task_id"]) for item in items}
    if len(tasks) != 1:
        raise RuntimeError(f"A training batch mixes tasks: {sorted(tasks)}")
    return {
        "image": torch.stack([item["image"] for item in items]),
        "task_id": str(items[0]["task_id"]),
        "canonical_id": [str(item["canonical_id"]) for item in items],
        "teacher_probability": torch.stack(
            [item["teacher_probability"] for item in items]
        ),
        "teacher_dispersion": torch.stack(
            [item["teacher_dispersion"] for item in items]
        ),
    }


def task_loader(
    frame: pd.DataFrame,
    task: str,
    settings: dict[str, Any],
    bank_dir: Path,
    teacher_runtime: Any,
    shuffle: bool,
) -> DataLoader:
    paths = common.bank_paths(bank_dir, task)
    dataset = BankTargetDataset(
        frame,
        task,
        int(settings["input_size"]),
        paths["mean"],
        paths["dispersion"],
        teacher_runtime,
    )
    generator = torch.Generator().manual_seed(
        int(settings["seed"])
        + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
    )
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(
            settings["train_batch_size"] if shuffle else settings["audit_batch_size"]
        ),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=collate_bank,
        drop_last=False,
    )


def load_warmstart(network, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["trainable_state"]
    named = dict(network.anchor.named_parameters())
    missing = []
    for name, value in state["task_private_lora"].items():
        if name not in named:
            missing.append(name)
            continue
        named[name].data.copy_(value.to(named[name].dtype))
    if missing:
        raise RuntimeError(f"Warm-start LoRA keys are absent: {missing[:4]}")
    result = network.residual_heads.load_state_dict(
        state["task_private_correction"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Warm-start correction state did not load strictly.")
    return {
        "format": payload.get("format"),
        "lora_tensors": len(state["task_private_lora"]),
        "correction_tensors": len(state["task_private_correction"]),
    }


def task_parameters(network, task: str) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    correction = list(network.residual_heads[str(task)].parameters())
    lora = []
    for layer in network.lora_layers():
        lora.extend(layer.lora_down[str(task)].parameters())
        lora.extend(layer.lora_up[str(task)].parameters())
    if not correction or not lora:
        raise RuntimeError(f"Task {task} has no private trainable route.")
    return correction, lora


def task_lora_up_regularization(network, task: str) -> torch.Tensor:
    values = [
        layer.lora_up[str(task)].weight.square().mean()
        for layer in network.lora_layers()
    ]
    return torch.stack(values).mean()


def cosine(initial: float, minimum: float, progress: float) -> float:
    return float(minimum) + 0.5 * (float(initial) - float(minimum)) * (
        1.0 + math.cos(math.pi * min(max(float(progress), 0.0), 1.0))
    )


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-12)


def train_task(
    network,
    frame: pd.DataFrame,
    task: str,
    settings: dict[str, Any],
    bank_dir: Path,
    teacher_runtime: Any,
    device: torch.device,
) -> tuple[dict[str, Any], list[str]]:
    paths = common.bank_paths(bank_dir, task)
    dispersion_bank = np.load(paths["dispersion"], mmap_mode="r")
    positions = frame["task_position"].astype(int).to_numpy()
    values = np.asarray(dispersion_bank[positions], dtype=np.float32)
    scale = float(
        np.quantile(values, float(settings["disagreement_scale_quantile"]))
    )
    scale = max(scale, 1e-4)
    correction, lora = task_parameters(network, task)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": correction,
                "lr": float(settings["correction_learning_rate"]),
                "route": "correction",
            },
            {
                "params": lora,
                "lr": float(settings["lora_learning_rate"]),
                "route": "lora",
            },
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    loader = task_loader(frame, task, settings, bank_dir, teacher_runtime, shuffle=True)
    microbatches = len(loader)
    target_updates = int(settings["target_optimizer_updates_per_task"])
    accumulation = max(1, int(math.ceil(microbatches / max(target_updates, 1))))
    planned_updates = int(math.ceil(microbatches / accumulation))
    amp_dtype = (
        torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    )
    floor = float(settings["disagreement_weight_floor"])
    totals = {
        "loss": 0.0,
        "kl": 0.0,
        "coordinate": 0.0,
        "correction_l2": 0.0,
        "lora_up_l2": 0.0,
        "reliability": 0.0,
        "foundation_feature_delta_abs_mean": 0.0,
    }
    seen_ids: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    update_index = 0
    network.train()
    for micro_index, batch in enumerate(
        tqdm(loader, desc=f"FinalDistillation full distillation {task}"), 1
    ):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["teacher_probability"].to(device, non_blocking=True).float()
        target = target / target.sum((-2, -1), keepdim=True).clamp_min(1e-12)
        dispersion = batch["teacher_dispersion"].to(device, non_blocking=True).float()
        reliability = floor + (1.0 - floor) / (
            1.0 + (dispersion / scale).square()
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(settings.get("amp", True)),
        ):
            output = network(images, task)
        logits = output["heatmap_logits"].float()
        log_probability = F.log_softmax(
            logits.flatten(2) / max(float(settings["temperature"]), 1e-6), dim=-1
        )
        target_flat = target.flatten(2).clamp_min(1e-12)
        kl_point = (
            target_flat * (target_flat.log() - log_probability)
        ).sum(-1)
        student_coordinate = common.topk_coordinates_px(
            log_probability.exp().reshape_as(target), int(settings["decode_topk"])
        )
        teacher_coordinate = common.topk_coordinates_px(
            target, int(settings["decode_topk"])
        )
        coordinate_point = F.smooth_l1_loss(
            student_coordinate,
            teacher_coordinate,
            beta=float(settings["coordinate_smooth_l1_beta"]),
            reduction="none",
        ).mean(-1)
        kl = weighted_mean(kl_point, reliability)
        coordinate = weighted_mean(coordinate_point, reliability)
        correction_l2 = output["residual_heatmap_logits"].float().square().mean()
        lora_up_l2 = task_lora_up_regularization(network, task)
        loss = (
            kl
            + float(settings["coordinate_loss_weight"]) * coordinate
            + float(settings["correction_l2_weight"]) * correction_l2
            + float(settings["lora_up_l2_weight"]) * lora_up_l2
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite FinalDistillation loss for task {task}.")
        group_start = ((micro_index - 1) // accumulation) * accumulation + 1
        group_stop = min(group_start + accumulation - 1, microbatches)
        group_size = group_stop - group_start + 1
        (loss / group_size).backward()
        should_step = micro_index % accumulation == 0 or micro_index == microbatches
        if should_step:
            progress = update_index / max(planned_updates - 1, 1)
            correction_lr = cosine(
                settings["correction_learning_rate"],
                settings["correction_minimum_learning_rate"],
                progress,
            )
            lora_lr = cosine(
                settings["lora_learning_rate"],
                settings["lora_minimum_learning_rate"],
                progress,
            )
            for group in optimizer.param_groups:
                group["lr"] = correction_lr if group["route"] == "correction" else lora_lr
            torch.nn.utils.clip_grad_norm_(correction + lora, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_index += 1
        totals["loss"] += float(loss.detach())
        totals["kl"] += float(kl.detach())
        totals["coordinate"] += float(coordinate.detach())
        totals["correction_l2"] += float(correction_l2.detach())
        totals["lora_up_l2"] += float(lora_up_l2.detach())
        totals["reliability"] += float(reliability.mean().detach())
        totals["foundation_feature_delta_abs_mean"] += float(
            output["foundation_feature_delta_abs_mean"].detach()
        )
        seen_ids.extend(batch["canonical_id"])
    return (
        {
            "task_id": task,
            "images": int(len(frame)),
            "microbatches": int(microbatches),
            "gradient_accumulation": int(accumulation),
            "optimizer_updates": int(update_index),
            "disagreement_scale_px": scale,
            **{key: value / max(microbatches, 1) for key, value in totals.items()},
        },
        seen_ids,
    )


@torch.inference_mode()
def evaluate(
    network,
    frame: pd.DataFrame,
    settings: dict[str, Any],
    bank_dir: Path,
    teacher_runtime: Any,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    amp_dtype = (
        torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    )
    rows = []
    scale_zero_max = 0.0
    network.eval()
    for task in settings["tasks"]:
        current = frame[frame["task_id"].astype(str) == str(task)].reset_index(drop=True)
        loader = task_loader(current, str(task), settings, bank_dir, teacher_runtime, shuffle=False)
        values = {"anchor_js": [], "student_js": [], "anchor_coord": [], "student_coord": []}
        for batch in tqdm(loader, desc=f"FinalDistillation audit {task}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["teacher_probability"].to(device, non_blocking=True).float()
            target = target / target.sum((-2, -1), keepdim=True).clamp_min(1e-12)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(settings.get("amp", True)),
            ):
                output = network(images, str(task))
                fallback = network(images, str(task), residual_scale=0.0)
            student = common.spatial_probability(
                output["heatmap_logits"], settings["temperature"]
            )
            anchor = common.spatial_probability(
                output["anchor_heatmap_logits"], settings["temperature"]
            )
            scale_zero_max = max(
                scale_zero_max,
                float(
                    (fallback["heatmap_logits"] - output["anchor_heatmap_logits"])
                    .abs()
                    .max()
                ),
            )
            target_coord = common.topk_coordinates_px(target, settings["decode_topk"])
            anchor_coord = common.topk_coordinates_px(anchor, settings["decode_topk"])
            student_coord = common.topk_coordinates_px(student, settings["decode_topk"])
            values["anchor_js"].extend(common.symmetric_js(anchor, target).mean(1).cpu().tolist())
            values["student_js"].extend(common.symmetric_js(student, target).mean(1).cpu().tolist())
            values["anchor_coord"].extend(
                torch.linalg.vector_norm(anchor_coord - target_coord, dim=-1)
                .mean(1)
                .cpu()
                .tolist()
            )
            values["student_coord"].extend(
                torch.linalg.vector_norm(student_coord - target_coord, dim=-1)
                .mean(1)
                .cpu()
                .tolist()
            )
        row = {
            "task_id": str(task),
            "images": int(len(current)),
            "anchor_js_to_teacher": float(np.mean(values["anchor_js"])),
            "student_js_to_teacher": float(np.mean(values["student_js"])),
            "anchor_coordinate_distance_px": float(np.mean(values["anchor_coord"])),
            "student_coordinate_distance_px": float(np.mean(values["student_coord"])),
        }
        row["js_ratio"] = row["student_js_to_teacher"] / max(
            row["anchor_js_to_teacher"], 1e-12
        )
        row["coordinate_ratio"] = row["student_coordinate_distance_px"] / max(
            row["anchor_coordinate_distance_px"], 1e-12
        )
        rows.append(row)
    aggregate = {
        "task_macro_anchor_js": float(np.mean([row["anchor_js_to_teacher"] for row in rows])),
        "task_macro_student_js": float(np.mean([row["student_js_to_teacher"] for row in rows])),
        "task_macro_anchor_coordinate_distance_px": float(
            np.mean([row["anchor_coordinate_distance_px"] for row in rows])
        ),
        "task_macro_student_coordinate_distance_px": float(
            np.mean([row["student_coordinate_distance_px"] for row in rows])
        ),
    }
    aggregate["task_macro_js_ratio"] = aggregate["task_macro_student_js"] / max(
        aggregate["task_macro_anchor_js"], 1e-12
    )
    aggregate["task_macro_coordinate_ratio"] = aggregate[
        "task_macro_student_coordinate_distance_px"
    ] / max(aggregate["task_macro_anchor_coordinate_distance_px"], 1e-12)
    return aggregate, rows, scale_zero_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = common.resolve(args.config)
    settings = json.loads(config_path.read_text())
    random.seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("FinalDistillation full distillation requires CUDA.")
    common.enforce_space_limit(settings)
    run_dir = common.resolve(settings["run_dir"])
    if bool(settings.get("enforce_empty_output", True)) and run_dir.exists() and any(
        run_dir.iterdir()
    ):
        raise RuntimeError(f"FinalDistillation run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    common.snapshot_sources(run_dir, config_path)
    common.write_json(run_dir / "config.resolved.json", settings)
    common.write_json(run_dir / "environment.json", common.environment_record())

    bank_run_dir = common.resolve(settings["bank_run_dir"])
    bank_summary = json.loads((bank_run_dir / "bank_summary.json").read_text())
    if bank_summary["status"] != "pass":
        raise RuntimeError("FinalDistillation teacher bank did not pass QC.")
    bank_dir = bank_run_dir / "teacher_bank"
    frame = pd.read_csv(bank_run_dir / "full_unlabeled_roles.csv")
    train_frame = frame[frame["role"] == "full_train"].reset_index(drop=True)
    audit_frame = frame[frame["role"] == "final_audit"].reset_index(drop=True)
    if len(audit_frame) != int(settings["final_audit_images_per_task"]) * len(
        settings["tasks"]
    ):
        raise RuntimeError("Final audit split size drifted.")

    teacher_runtime, dense_fusion = common.load_dependencies()
    payloads, checkpoint_records = teacher_runtime.load_payloads(settings)
    medoid_index = next(
        index
        for index, item in enumerate(settings["checkpoints"])
        if int(item["fold"]) == int(settings["medoid_fold"])
    )
    device = torch.device("cuda")
    medoid_payload = payloads[medoid_index]
    anchor = teacher_runtime.build_model(
        dense_fusion, medoid_payload, medoid_payload["model_state"], device
    )
    del payloads
    network = final_distillation_model.DecoupledTaskPrivateFoundationCorrection(
        anchor,
        medoid_payload["task_configs"],
        settings["task_scales"],
        shared_channels=int(medoid_payload["settings"].get("fusion_shared_channels", 128)),
        hidden_channels=int(settings["residual_hidden_channels"]),
        logit_bound=float(settings["residual_logit_bound"]),
        lora_last_blocks=int(settings["lora_last_blocks"]),
        qkv_lora_rank=int(settings["qkv_lora_rank"]),
        qkv_lora_alpha=float(settings["qkv_lora_alpha"]),
        projection_lora_rank=int(settings["projection_lora_rank"]),
        projection_lora_alpha=float(settings["projection_lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
    ).to(device)
    del medoid_payload
    warmstart_path = common.resolve(settings["warmstart_checkpoint"])
    if common.sha256_file(warmstart_path) != str(settings["warmstart_checkpoint_sha256"]):
        raise RuntimeError("WarmStart warm-start checkpoint checksum mismatch.")
    warmstart_info = load_warmstart(network, warmstart_path)
    frozen_hash_before = final_distillation_model.frozen_anchor_state_sha256(network.anchor)
    warm_aggregate, warm_rows, warm_scale_zero = evaluate(
        network, audit_frame, settings, bank_dir, teacher_runtime, device
    )

    initial_task_hashes = {
        task: {
            "lora": network.lora_state_sha256(task),
            "correction": network.correction_state_sha256(task),
        }
        for task in network.task_scales
    }
    histories = []
    all_seen: list[str] = []
    task_checkpoint_dir = run_dir / "task_checkpoints"
    task_checkpoint_dir.mkdir()
    for task in settings["tasks"]:
        current = train_frame[
            train_frame["task_id"].astype(str) == str(task)
        ].reset_index(drop=True)
        history, seen = train_task(
            network, current, str(task), settings, bank_dir, teacher_runtime, device
        )
        histories.append(history)
        all_seen.extend(seen)
        torch.save(
            {
                "format": "final_distillation_task_boundary_recovery_v1",
                "completed_tasks": [value["task_id"] for value in histories],
                "trainable_state": network.trainable_state_dict(),
                "training_history": histories,
                "observed_rows": len(all_seen),
                "settings": settings,
            },
            task_checkpoint_dir / f"after_{str(task).lower()}.pt",
        )
        common.write_json(
            run_dir / "training_progress.json",
            {
                "status": "in_progress",
                "completed_tasks": [value["task_id"] for value in histories],
                "observed_rows": len(all_seen),
                "last_task": str(task),
            },
        )
        common.enforce_space_limit(settings)

    frozen_hash_after = final_distillation_model.frozen_anchor_state_sha256(network.anchor)
    final_aggregate, final_rows, final_scale_zero = evaluate(
        network, audit_frame, settings, bank_dir, teacher_runtime, device
    )
    final_task_hashes = {
        task: {
            "lora": network.lora_state_sha256(task),
            "correction": network.correction_state_sha256(task),
        }
        for task in network.task_scales
    }
    expected_train_ids = set(train_frame["sha256"].astype(str))
    observed_train_ids = set(all_seen)
    coverage = {
        "scheduled_rows": int(len(train_frame)),
        "observed_rows": int(len(all_seen)),
        "expected_unique": int(len(expected_train_ids)),
        "observed_unique": int(len(observed_train_ids)),
        "duplicate_draws": int(len(all_seen) - len(observed_train_ids)),
        "missing_ids": int(len(expected_train_ids - observed_train_ids)),
        "unexpected_ids": int(len(observed_train_ids - expected_train_ids)),
        "all_teacher_bank_images": int(bank_summary["coverage"]["rows"]),
        "warmstart_gradient_images": int(bank_summary["coverage"]["warmstart_train_images"]),
        "full_continuation_gradient_images": int(len(expected_train_ids)),
        "total_unique_gradient_images": int(
            bank_summary["coverage"]["warmstart_train_images"] + len(expected_train_ids)
        ),
        "locked_non_gradient_audit_images": int(
            bank_summary["coverage"]["warmstart_audit_images"]
            + bank_summary["coverage"]["final_audit_images"]
        ),
    }
    task_routes = {
        task: {
            "lora_changed": initial_task_hashes[task]["lora"]
            != final_task_hashes[task]["lora"],
            "correction_changed": initial_task_hashes[task]["correction"]
            != final_task_hashes[task]["correction"],
        }
        for task in network.task_scales
    }
    gates = {
        "js_ratio_pass": final_aggregate["task_macro_js_ratio"]
        <= float(settings["gate"]["maximum_js_ratio"]),
        "coordinate_ratio_pass": final_aggregate["task_macro_coordinate_ratio"]
        <= float(settings["gate"]["maximum_coordinate_ratio"]),
        "per_task_js_pass": max(row["js_ratio"] for row in final_rows)
        <= float(settings["gate"]["maximum_per_task_js_ratio"]),
        "js_improves_warmstart": final_aggregate["task_macro_student_js"]
        < warm_aggregate["task_macro_student_js"],
        "coordinate_improves_warmstart": final_aggregate[
            "task_macro_student_coordinate_distance_px"
        ]
        < warm_aggregate["task_macro_student_coordinate_distance_px"],
        "frozen_hash_pass": frozen_hash_before == frozen_hash_after,
        "scale_zero_exact_pass": warm_scale_zero == 0.0 and final_scale_zero == 0.0,
        "coverage_exact_pass": coverage["duplicate_draws"] == 0
        and coverage["missing_ids"] == 0
        and coverage["unexpected_ids"] == 0,
        "all_trained_routes_changed": all(
            task_routes[task]["lora_changed"]
            and task_routes[task]["correction_changed"]
            for task in settings["tasks"]
        ),
        "fetal_femur_route_unchanged": not task_routes["fetal_femur"]["lora_changed"]
        and not task_routes["fetal_femur"]["correction_changed"],
    }
    passed = all(bool(value) for value in gates.values())
    pd.DataFrame(histories).to_csv(run_dir / "training_by_task.csv", index=False)
    pd.DataFrame(warm_rows).to_csv(run_dir / "warmstart_audit_by_task.csv", index=False)
    pd.DataFrame(final_rows).to_csv(run_dir / "final_audit_by_task.csv", index=False)
    torch.save(
        {
            "format": "final_distillation_full_official_unlabeled_ensemble_distillation_v1",
            "trainable_state": network.trainable_state_dict(),
            "settings": settings,
            "teacher_models": checkpoint_records,
            "warmstart": warmstart_info,
            "audit_summary": final_aggregate,
        },
        run_dir / "final_distillation_task_private_lora_correction.pt",
    )
    summary = {
        "status": "pass" if passed else "fail",
        "decision": "authorize_single_model_generalization_evaluation"
        if passed
        else "stop_before_official_evaluation",
        "scope": "single-model full official-unlabeled ensemble distillation",
        "official_or_labeled_ground_truth_read": False,
        "warmstart": warmstart_info,
        "warmstart_audit": warm_aggregate,
        "final_audit": final_aggregate,
        "gates": gates,
        "coverage": coverage,
        "task_routes": task_routes,
        "frozen_hash_before": frozen_hash_before,
        "frozen_hash_after": frozen_hash_after,
        "warm_scale_zero_max": warm_scale_zero,
        "final_scale_zero_max": final_scale_zero,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in network.trainable_parameters())
        ),
        "filesystem_used_bytes_after": common.filesystem_used_bytes(),
    }
    common.write_json(run_dir / "metrics_summary.json", summary)
    common.write_json(
        run_dir / "qc_summary.json",
        {"status": "pass" if passed else "fail", "checks": gates},
    )
    common.write_json(
        run_dir / "training_progress.json",
        {
            "status": "complete",
            "completed_tasks": [value["task_id"] for value in histories],
            "observed_rows": len(all_seen),
        },
    )
    common.output_manifest(run_dir, hash_large_files=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
