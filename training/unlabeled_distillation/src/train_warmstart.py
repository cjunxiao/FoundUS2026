from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WARMSTART_BASE_SRC = Path(__file__).resolve().parent
TASK_LORA_MODEL = (
    PROJECT_ROOT / "training/unlabeled_distillation/dependencies/lora/model.py"
)
CONTEXTUAL_CORRECTION_MODEL = (
    PROJECT_ROOT / "training/unlabeled_distillation/dependencies/contextual_head/model.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def differentiable_topk_coordinates(
    logits: torch.Tensor, topk: int, beta: float
) -> torch.Tensor:
    _, _, height, width = logits.shape
    values, indices = torch.topk(
        logits.flatten(2), min(max(int(topk), 1), height * width), dim=-1
    )
    weights = torch.softmax(values * float(beta), dim=-1)
    x = (indices % width).to(logits.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(logits.dtype)
    return torch.stack(((weights * x).sum(-1), (weights * y).sum(-1)), dim=-1)


def cosine_value(initial: float, minimum: float, progress: float) -> float:
    return float(minimum) + 0.5 * (float(initial) - float(minimum)) * (
        1.0 + math.cos(math.pi * float(progress))
    )


def make_train_stage0(base):
    def train_stage0(network, train_frame, teacher, teacher_runtime, settings, device):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": network.correction_parameters(),
                    "lr": float(settings["correction_learning_rate"]),
                    "route": "correction",
                },
                {
                    "params": network.lora_parameters(),
                    "lr": float(settings["lora_learning_rate"]),
                    "route": "lora",
                },
            ],
            weight_decay=float(settings["weight_decay"]),
        )
        epochs = int(settings["epochs"])
        amp_dtype = (
            torch.bfloat16
            if settings["amp_dtype"] == "bfloat16"
            else torch.float16
        )
        history = []
        network.train()
        for epoch in range(1, epochs + 1):
            progress = (epoch - 1) / max(epochs - 1, 1)
            correction_lr = cosine_value(
                settings["correction_learning_rate"],
                settings["correction_minimum_learning_rate"],
                progress,
            )
            lora_lr = cosine_value(
                settings["lora_learning_rate"],
                settings["lora_minimum_learning_rate"],
                progress,
            )
            for group in optimizer.param_groups:
                group["lr"] = (
                    correction_lr if group["route"] == "correction" else lora_lr
                )
            loader = base.loader_for(
                train_frame, teacher, teacher_runtime, settings, epoch, shuffle=True
            )
            totals = {
                "loss": 0.0,
                "kl": 0.0,
                "coordinate": 0.0,
                "correction_l2": 0.0,
                "lora_up_l2": 0.0,
                "foundation_feature_delta_abs_mean": 0.0,
            }
            steps = 0
            for batch in tqdm(
                loader, desc=f"FinalStudent train epoch {epoch:02d}", leave=False
            ):
                task = str(batch["task_id"])
                images = batch["image"].to(device, non_blocking=True)
                target = batch["teacher_probability"].to(
                    device, non_blocking=True
                ).float()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=bool(settings.get("amp", True)),
                ):
                    output = network(images, task)
                logits = output["heatmap_logits"].float()
                flat_logits = logits.flatten(2) / max(
                    float(settings["temperature"]), 1e-6
                )
                log_probability = F.log_softmax(flat_logits, dim=-1)
                target_flat = target.flatten(2).clamp_min(1e-12)
                kl = (
                    target_flat * (target_flat.log() - log_probability)
                ).sum(-1).mean()
                student_coordinate = differentiable_topk_coordinates(
                    logits,
                    int(settings["decode_topk"]),
                    float(settings["decode_topk_beta"]),
                )
                target_coordinate = teacher_runtime.decode_probability(
                    target, int(settings["decode_topk"])
                ) * float(target.shape[-1] - 1)
                coordinate = F.smooth_l1_loss(
                    student_coordinate,
                    target_coordinate,
                    beta=float(settings["coordinate_smooth_l1_beta"]),
                )
                correction_l2 = output["residual_heatmap_logits"].float().square().mean()
                lora_up_l2 = network.lora_up_regularization()
                loss = (
                    kl
                    + float(settings["coordinate_loss_weight"]) * coordinate
                    + float(settings["correction_l2_weight"]) * correction_l2
                    + float(settings["lora_up_l2_weight"]) * lora_up_l2
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite FinalStudent Stage-0 loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.trainable_parameters(), 1.0)
                optimizer.step()
                totals["loss"] += float(loss.detach())
                totals["kl"] += float(kl.detach())
                totals["coordinate"] += float(coordinate.detach())
                totals["correction_l2"] += float(correction_l2.detach())
                totals["lora_up_l2"] += float(lora_up_l2.detach())
                totals["foundation_feature_delta_abs_mean"] += float(
                    output["foundation_feature_delta_abs_mean"].detach()
                )
                steps += 1
            history.append(
                {
                    "epoch": epoch,
                    "correction_learning_rate": correction_lr,
                    "lora_learning_rate": lora_lr,
                    "steps": steps,
                    **{
                        key: value / max(steps, 1)
                        for key, value in totals.items()
                    },
                }
            )
        return history

    return train_stage0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(WARMSTART_BASE_SRC))
    base = _load(
        "warmstart_base_runner_for_final_student",
        WARMSTART_BASE_SRC / "warmstart_base.py",
    )
    model = _load("final_student_model", Path(__file__).resolve().parent / "model.py")
    configured_settings = json.loads(base.resolve_path(args.config).read_text())

    class ConfiguredStudent(model.DecoupledTaskPrivateFoundationCorrection):
        last_instance = None

        def __init__(self, *student_args, **student_kwargs):
            student_kwargs.update(
                {
                    "lora_last_blocks": int(configured_settings["lora_last_blocks"]),
                    "qkv_lora_rank": int(configured_settings["qkv_lora_rank"]),
                    "qkv_lora_alpha": float(configured_settings["qkv_lora_alpha"]),
                    "projection_lora_rank": int(
                        configured_settings["projection_lora_rank"]
                    ),
                    "projection_lora_alpha": float(
                        configured_settings["projection_lora_alpha"]
                    ),
                    "lora_dropout": float(configured_settings["lora_dropout"]),
                }
            )
            super().__init__(*student_args, **student_kwargs)
            self.initial_lora_hashes = {
                task: self.lora_state_sha256(task) for task in self.task_scales
            }
            self.initial_correction_hashes = {
                task: self.correction_state_sha256(task)
                for task in self.task_scales
            }
            ConfiguredStudent.last_instance = self

    base.FunctionalCompressionStudent = ConfiguredStudent
    base.train_stage0 = make_train_stage0(base)
    base.module_state_sha256 = model.frozen_anchor_state_sha256
    base.EXPERIMENT_DIR = EXPERIMENT_DIR
    base.run(args.config)

    instance = ConfiguredStudent.last_instance
    if instance is None:
        raise RuntimeError("FinalStudent student instance was not retained.")
    run_dir = base.resolve_path(configured_settings["run_dir"])
    old_checkpoint = run_dir / "stage0_residual_heads.pt"
    if old_checkpoint.exists():
        old_checkpoint.unlink()
    torch.save(
        {
            "format": "final_student_decoupled_task_private_foundation_correction_stage0_v1",
            "trainable_state": instance.trainable_state_dict(),
            "settings": configured_settings,
            "stage0_only": True,
        },
        run_dir / "stage0_task_private_lora_correction.pt",
    )

    final_lora_hashes = {
        task: instance.lora_state_sha256(task) for task in instance.task_scales
    }
    final_correction_hashes = {
        task: instance.correction_state_sha256(task)
        for task in instance.task_scales
    }
    lora_changed = {
        task: instance.initial_lora_hashes[task] != final_lora_hashes[task]
        for task in instance.task_scales
    }
    correction_changed = {
        task: instance.initial_correction_hashes[task]
        != final_correction_hashes[task]
        for task in instance.task_scales
    }
    trained_tasks = [str(value) for value in configured_settings["tasks"]]
    route_qc = {
        "all_trained_task_lora_changed": all(
            lora_changed[task] for task in trained_tasks
        ),
        "all_trained_task_correction_changed": all(
            correction_changed[task] for task in trained_tasks
        ),
        "fetal_femur_lora_unchanged": not lora_changed["fetal_femur"],
        "fetal_femur_correction_unchanged": not correction_changed["fetal_femur"],
    }

    summary_path = run_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "experiment": "FinalStudent decoupled task-private USFM correction",
            "isolated_change_vs_task_lora": (
                "locked old task scale applies only to anchor; new task-private "
                "contextual correction is added independently"
            ),
            "lora_locations": [list(value) for value in instance._lora_locations],
            "lora_trainable_parameters": int(
                sum(parameter.numel() for parameter in instance.lora_parameters())
            ),
            "correction_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in instance.correction_parameters()
                )
            ),
            "lora_changed_by_task": lora_changed,
            "correction_changed_by_task": correction_changed,
            "task_route_qc": route_qc,
        }
    )
    qc_path = run_dir / "qc_summary.json"
    qc = json.loads(qc_path.read_text())
    qc["task_private_routes"] = route_qc
    if not all(route_qc.values()):
        qc["status"] = "fail"
        summary["status"] = "fail"
        summary["decision"] = "stop_before_full_bank"
    base.write_json(summary_path, summary)
    base.write_json(qc_path, qc)
    base.write_output_manifest(run_dir)


if __name__ == "__main__":
    main()
