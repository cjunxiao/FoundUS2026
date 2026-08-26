from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
import model as exp301_model
import train_full_distillation as train


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def photometric_view(image: torch.Tensor, gamma: float, gain: float) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    black = rgb.mean(1, keepdim=True) < 0.005
    changed = (rgb.clamp_min(1e-6).pow(float(gamma)) * float(gain)).clamp(0.0, 1.0)
    changed = torch.where(black, rgb, changed)
    return (changed - mean) / std


def load_network(settings, exp290, exp191, device):
    payloads, _ = exp290.load_payloads(settings)
    medoid_index = next(
        index
        for index, item in enumerate(settings["checkpoints"])
        if int(item["fold"]) == int(settings["medoid_fold"])
    )
    payload = payloads[medoid_index]
    anchor = exp290.build_model(exp191, payload, payload["model_state"], device)
    network = exp301_model.DecoupledTaskPrivateFoundationCorrection(
        anchor,
        payload["task_configs"],
        settings["task_scales"],
        shared_channels=int(payload["settings"].get("fusion_shared_channels", 128)),
        hidden_channels=int(settings["residual_hidden_channels"]),
        logit_bound=float(settings["residual_logit_bound"]),
        lora_last_blocks=int(settings["lora_last_blocks"]),
        qkv_lora_rank=int(settings["qkv_lora_rank"]),
        qkv_lora_alpha=float(settings["qkv_lora_alpha"]),
        projection_lora_rank=int(settings["projection_lora_rank"]),
        projection_lora_alpha=float(settings["projection_lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
    ).to(device)
    return network


def load_final(network, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["trainable_state"]
    named = dict(network.anchor.named_parameters())
    for name, value in state["task_private_lora"].items():
        named[name].data.copy_(value.to(named[name].dtype))
    result = network.residual_heads.load_state_dict(
        state["task_private_correction"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Final Exp301 state did not load strictly.")


@torch.inference_mode()
def evaluate_state(network, frame, settings, exp290, device, state_name):
    views = [(0.90, 1.05), (1.10, 0.95)]
    amp_dtype = (
        torch.bfloat16 if settings["amp_dtype"] == "bfloat16" else torch.float16
    )
    rows = []
    network.eval()
    for task in settings["tasks"]:
        current = frame[frame["task_id"].astype(str) == str(task)].reset_index(drop=True)
        dataset = exp290.UnlabeledDataset(current, int(settings["input_size"]))
        loader = DataLoader(
            dataset,
            batch_size=int(settings["audit_batch_size"]),
            shuffle=False,
            num_workers=int(settings["num_workers"]),
            persistent_workers=int(settings["num_workers"]) > 0,
            pin_memory=True,
            collate_fn=exp290.collate_unlabeled,
        )
        anchor_drift = []
        student_drift = []
        for batch in tqdm(loader, desc=f"Exp301 stability {state_name} {task}", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(settings.get("amp", True)),
            ):
                reference = network(image, str(task))
            anchor_reference = common.topk_coordinates_px(
                common.spatial_probability(
                    reference["anchor_heatmap_logits"], settings["temperature"]
                ),
                settings["decode_topk"],
            )
            student_reference = common.topk_coordinates_px(
                common.spatial_probability(
                    reference["heatmap_logits"], settings["temperature"]
                ),
                settings["decode_topk"],
            )
            for gamma, gain in views:
                changed = photometric_view(image, gamma, gain)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=bool(settings.get("amp", True)),
                ):
                    output = network(changed, str(task))
                anchor_changed = common.topk_coordinates_px(
                    common.spatial_probability(
                        output["anchor_heatmap_logits"], settings["temperature"]
                    ),
                    settings["decode_topk"],
                )
                student_changed = common.topk_coordinates_px(
                    common.spatial_probability(
                        output["heatmap_logits"], settings["temperature"]
                    ),
                    settings["decode_topk"],
                )
                anchor_drift.extend(
                    torch.linalg.vector_norm(anchor_changed - anchor_reference, dim=-1)
                    .flatten()
                    .cpu()
                    .tolist()
                )
                student_drift.extend(
                    torch.linalg.vector_norm(student_changed - student_reference, dim=-1)
                    .flatten()
                    .cpu()
                    .tolist()
                )
        rows.append(
            {
                "state": state_name,
                "task_id": str(task),
                "images": int(len(current)),
                "anchor_mean_drift_px": float(np.mean(anchor_drift)),
                "student_mean_drift_px": float(np.mean(student_drift)),
                "anchor_p90_drift_px": float(np.quantile(anchor_drift, 0.90)),
                "student_p90_drift_px": float(np.quantile(student_drift, 0.90)),
            }
        )
    return rows


def aggregate(rows):
    return {
        "task_macro_anchor_mean_drift_px": float(
            np.mean([row["anchor_mean_drift_px"] for row in rows])
        ),
        "task_macro_student_mean_drift_px": float(
            np.mean([row["student_mean_drift_px"] for row in rows])
        ),
        "task_macro_anchor_p90_drift_px": float(
            np.mean([row["anchor_p90_drift_px"] for row in rows])
        ),
        "task_macro_student_p90_drift_px": float(
            np.mean([row["student_p90_drift_px"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(common.resolve(args.config).read_text())
    run_dir = common.resolve(settings["run_dir"])
    checkpoint = run_dir / "exp301_task_private_lora_correction.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    bank_run = common.resolve(settings["bank_run_dir"])
    frame = pd.read_csv(bank_run / "full_unlabeled_roles.csv")
    audit = frame[frame["role"] == "final_audit"].reset_index(drop=True)
    exp290, exp191 = common.load_dependencies()
    device = torch.device("cuda")
    network = load_network(settings, exp290, exp191, device)
    train.load_warmstart(network, common.resolve(settings["warmstart_checkpoint"]))
    warm_rows = evaluate_state(network, audit, settings, exp290, device, "warmstart")
    load_final(network, checkpoint)
    final_rows = evaluate_state(network, audit, settings, exp290, device, "final")
    warm = aggregate(warm_rows)
    final = aggregate(final_rows)
    nondegraded_tasks = sum(
        final_row["student_mean_drift_px"]
        <= final_row["anchor_mean_drift_px"] + 0.10
        for final_row in final_rows
    )
    gates = {
        "mean_not_worse_than_anchor": final["task_macro_student_mean_drift_px"]
        <= final["task_macro_anchor_mean_drift_px"] + 0.10,
        "p90_not_worse_than_anchor": final["task_macro_student_p90_drift_px"]
        <= final["task_macro_anchor_p90_drift_px"] + 0.25,
        "mean_not_worse_than_warmstart": final["task_macro_student_mean_drift_px"]
        <= warm["task_macro_student_mean_drift_px"] + 0.10,
        "at_least_five_tasks_nondegraded": nondegraded_tasks >= 5,
    }
    summary = {
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "label-free fixed photometric perturbation stability audit",
        "official_or_labeled_ground_truth_read": False,
        "views": [{"gamma": 0.90, "gain": 1.05}, {"gamma": 1.10, "gain": 0.95}],
        "warmstart": warm,
        "final": final,
        "nondegraded_tasks": int(nondegraded_tasks),
        "gates": gates,
    }
    pd.DataFrame(warm_rows + final_rows).to_csv(
        run_dir / "unlabeled_stability_by_task.csv", index=False
    )
    common.write_json(run_dir / "unlabeled_stability_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise RuntimeError("Exp301 unlabeled stability gate failed.")


if __name__ == "__main__":
    main()

