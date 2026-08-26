from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import evaluation  # noqa: F401 - bind Exp161 evaluator before loading the shared trainer
import model
from protocol import PROJECT_ROOT, resolve_path, sha256_file


EXP157_TRAINER = (
    PROJECT_ROOT / "2-code/157-active-vertical-exp152-5fold/src/train_exp157.py"
)


def _load_trainer():
    name = "exp157_trainer_engine_for_exp161"
    spec = importlib.util.spec_from_file_location(name, EXP157_TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError(EXP157_TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_trainer()
_base_resolve_settings = base.resolve_settings


def validate_identity_contract(settings: dict[str, Any]) -> None:
    if str(settings.get("endpoint_identity_mode")) != "exp152_fixed_internal":
        raise ValueError("Exp161 requires endpoint_identity_mode=exp152_fixed_internal.")
    if sorted(str(value) for value in settings.get("internal_identity_tasks", [])) != [
        "A4C",
        "PSAX",
    ]:
        raise ValueError("Exp161 requires fixed internal identities for A4C and PSAX.")
    if str(settings.get("validation_primary_identity")) != "fixed_internal":
        raise ValueError("Exp161 checkpoint selection must use fixed_internal predictions.")
    if str(settings.get("validation_target_identity")) != "fixed_internal":
        raise ValueError("Exp161 primary validation targets must be canonicalized internally.")
    if str(settings.get("submission_aggregation")) != (
        "internal_coordinate_median_then_single_official_vertical_conversion"
    ):
        raise ValueError("Exp161 requires internal-first ensemble aggregation.")


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = _base_resolve_settings(args)
    validate_identity_contract(settings)
    expected = settings.get("internal_identity_reference_sha256", {})
    for task in settings["internal_identity_tasks"]:
        path = settings["internal_identity_reference_csv"][task]
        actual = sha256_file(path)
        if actual != str(expected[task]):
            raise RuntimeError(
                f"Historical internal-identity reference hash mismatch for {task}: {actual}"
            )
    return settings


base.resolve_settings = resolve_settings


def snapshot_sources(run_dir: Path, config: Path) -> None:
    destination = run_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = set(Path(__file__).resolve().parent.glob("*.py"))
    sources.update(
        {
            config.resolve(),
            PROJECT_ROOT / "2-code/157-active-vertical-exp152-5fold/src/train_exp157.py",
            PROJECT_ROOT / "2-code/157-active-vertical-exp152-5fold/src/protocol.py",
            PROJECT_ROOT / "2-code/152-exp56-psax-pair-decoder/src/model.py",
            PROJECT_ROOT / "2-code/159-canonical-internal-exp152/src/model.py",
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
    base.write_json(run_dir / "source_status.json", records)


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
