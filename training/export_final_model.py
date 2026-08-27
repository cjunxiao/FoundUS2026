from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_hash(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected and actual != str(expected):
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the fused anchor and distilled trainable state into a deployable checkpoint."
    )
    parser.add_argument(
        "--config",
        default="training/unlabeled_distillation/configs/final.json",
        type=Path,
    )
    parser.add_argument("--output-dir", default="outputs/deployment", type=Path)
    args = parser.parse_args()

    config_path = resolve(args.config)
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Export directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    medoid_fold = int(settings["medoid_fold"])
    anchor_record = next(
        item for item in settings["checkpoints"] if int(item["fold"]) == medoid_fold
    )
    anchor_path = resolve(anchor_record["path"])
    anchor_sha = validate_hash(anchor_path, anchor_record.get("sha256"))
    correction_path = resolve(settings["run_dir"]) / "final_distillation_task_private_lora_correction.pt"
    correction_sha = validate_hash(correction_path, None)

    common = load_module(
        "final_export_common", PROJECT_ROOT / "training/unlabeled_distillation/src/common.py"
    )
    student_model = load_module(
        "final_export_student_model",
        PROJECT_ROOT / "training/unlabeled_distillation/src/model.py",
    )
    teacher_runtime, dense_fusion = common.load_dependencies()
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    anchor = teacher_runtime.build_model(
        dense_fusion, anchor_payload, anchor_payload["model_state"], torch.device("cpu")
    )
    network = student_model.DecoupledTaskPrivateFoundationCorrection(
        anchor,
        anchor_payload["task_configs"],
        settings["task_scales"],
        shared_channels=int(anchor_payload["settings"].get("fusion_shared_channels", 128)),
        hidden_channels=int(settings["residual_hidden_channels"]),
        logit_bound=float(settings["residual_logit_bound"]),
        lora_last_blocks=int(settings["lora_last_blocks"]),
        qkv_lora_rank=int(settings["qkv_lora_rank"]),
        qkv_lora_alpha=float(settings["qkv_lora_alpha"]),
        projection_lora_rank=int(settings["projection_lora_rank"]),
        projection_lora_alpha=float(settings["projection_lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
    )

    correction = torch.load(correction_path, map_location="cpu", weights_only=False)
    trainable = correction["trainable_state"]
    named_parameters = dict(network.anchor.named_parameters())
    for name, value in trainable["task_private_lora"].items():
        if name not in named_parameters:
            raise RuntimeError(f"Missing LoRA parameter in anchor: {name}")
        named_parameters[name].data.copy_(value.to(named_parameters[name].dtype))
    result = network.residual_heads.load_state_dict(
        trainable["task_private_correction"], strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Contextual correction state did not load strictly")
    network.eval()

    deployment = load_module(
        "final_export_deployment_model", PROJECT_ROOT / "final_inference/model.py"
    )
    deployment_network = deployment.build_network(
        anchor_payload["task_configs"], settings["task_scales"]
    )
    native_state = network.state_dict()
    deployment_state = deployment_network.state_dict()
    if list(native_state) != list(deployment_state):
        raise RuntimeError("Training and deployment state keys differ")
    mismatched = [
        name
        for name in native_state
        if tuple(native_state[name].shape) != tuple(deployment_state[name].shape)
    ]
    if mismatched:
        raise RuntimeError(f"Training and deployment tensor shapes differ: {mismatched[:5]}")
    strict = deployment_network.load_state_dict(native_state, strict=True)
    if strict.missing_keys or strict.unexpected_keys:
        raise RuntimeError("Deployment network strict-load preflight failed")
    del deployment_network, deployment_state
    gc.collect()

    payload = {
        "format_version": 1,
        "model_name": "FoundUS 2026 final single-model student",
        "task_configs": anchor_payload["task_configs"],
        "task_scales": settings["task_scales"],
        "model_state": native_state,
        "decode_topk": int(settings["decode_topk"]),
        "decode_topk_beta": float(settings["decode_topk_beta"]),
        "input_size": int(settings["input_size"]),
        "heatmap_size": int(settings["heatmap_size"]),
        "medoid_fold": medoid_fold,
        "official_output_sort_tasks": ["A4C", "PSAX"],
        "training_unlabeled_unique_images": int(settings["expected_unique_images"]),
        "source_anchor_sha256": anchor_sha,
        "source_correction_sha256": correction_sha,
    }
    checkpoint_path = output_dir / "best_model.pth"
    temporary = output_dir / ".best_model.pth.tmp"
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    inference_dir = PROJECT_ROOT / "final_inference"
    for name in (
        "model.py",
        "predict.py",
        "Dockerfile",
        "requirements.txt",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copy2(inference_dir / name, output_dir / name)
    dockerfile = output_dir / "Dockerfile"
    dockerfile.write_text(
        re.sub(
            r'org\.opencontainers\.image\.version="[^"]+"',
            'org.opencontainers.image.version="local-reproduction"',
            dockerfile.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema": "fub2026_local_reproduction_v1",
        "status": "locally_reproduced_unverified",
        "method": payload["model_name"],
        "model_count_at_inference": 1,
        "checkpoint_count_at_inference": 1,
        "image_views_at_inference": 1,
        "checkpoint": {
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "embedded_in_docker_build_context": True,
        },
        "checkpoint_sha256": checkpoint_sha,
        "source_anchor": {"path": relative(anchor_path), "sha256": anchor_sha},
        "source_correction": {
            "path": relative(correction_path),
            "sha256": correction_sha,
        },
        "final_training_config": relative(config_path),
        "runtime_network_required": False,
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "export_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "checkpoint": manifest["checkpoint"],
                "strict_state_keys": True,
                "strict_state_shapes": True,
                "strict_deployment_load": True,
                "docker_context": relative(output_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
