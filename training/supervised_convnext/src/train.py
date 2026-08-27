from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import evaluation  # noqa: F401 - bind SupervisedConvNeXt evaluator before loading the shared trainer
import model
from protocol import PROJECT_ROOT, resolve_path, sha256_file


FIVEFOLD_TRAINER_TRAINER = (
    Path(__file__).resolve().parents[1] / "dependencies/fivefold_trainer/train.py"
)


def _load_trainer():
    name = "fivefold_trainer_trainer_engine_for_supervised_convnext"
    spec = importlib.util.spec_from_file_location(name, FIVEFOLD_TRAINER_TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError(FIVEFOLD_TRAINER_TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_trainer()
_base_resolve_settings = base.resolve_settings


def validate_identity_contract(settings: dict[str, Any]) -> None:
    if str(settings.get("endpoint_identity_mode")) != "canonical_channel_identity":
        raise ValueError("SupervisedConvNeXt requires endpoint_identity_mode=canonical_channel_identity.")
    if sorted(str(value) for value in settings.get("internal_identity_tasks", [])) != [
        "A4C",
        "PSAX",
    ]:
        raise ValueError("SupervisedConvNeXt requires fixed internal identities for A4C and PSAX.")
    if str(settings.get("validation_primary_identity")) != "fixed_internal":
        raise ValueError("SupervisedConvNeXt checkpoint selection must use fixed_internal predictions.")
    if str(settings.get("validation_target_identity")) != "fixed_internal":
        raise ValueError("SupervisedConvNeXt primary validation targets must be canonicalized internally.")
    if str(settings.get("submission_aggregation")) != (
        "internal_coordinate_median_then_single_official_vertical_conversion"
    ):
        raise ValueError("SupervisedConvNeXt requires internal-first ensemble aggregation.")


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = _base_resolve_settings(args)
    validate_identity_contract(settings)
    return settings


base.resolve_settings = resolve_settings


def snapshot_sources(run_dir: Path, config: Path) -> None:
    del run_dir, config

base.snapshot_sources = snapshot_sources


def _invocation_paths() -> tuple[Path, Path] | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-forward", action="store_true")
    args, _ = parser.parse_known_args()
    if args.config is None:
        return None
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = str(args.run_dir or payload["run_dir"])
    if args.dry_run:
        run_dir += "-dry-run"
    elif args.smoke_forward:
        run_dir += "-smoke-forward"
    return args.config, resolve_path(run_dir)


def main() -> None:
    invocation = _invocation_paths()
    base.run()
    if invocation is None:
        return
    config_path, run_dir = invocation
    config_settings = json.loads(config_path.read_text(encoding="utf-8"))
    base.write_json(
        run_dir / "endpoint_identity_policy.json",
        model.endpoint_identity_policy(config_settings),
    )
    base.write_json(run_dir / "output_manifest.json", base.output_manifest(run_dir))


if __name__ == "__main__":
    main()
