from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import evaluation as local_evaluation
import model as local_model
import protocol as local_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINER_SOURCE = PROJECT_ROOT / "training/supervised_convnext/dependencies/fivefold_trainer/train.py"


def _load_trainer():
    sys.modules["model"] = local_model
    sys.modules["evaluation"] = local_evaluation
    sys.modules["protocol"] = local_protocol
    name = "fivefold_trainer_trainer_for_usfm_heatmap"
    spec = importlib.util.spec_from_file_location(name, TRAINER_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()
trainer.grouped_fold_model = local_model
trainer.DEFAULTS.update(
    {
        "model_variant": "usfm_task_private_vitpose",
        "input_size": 256,
        "heatmap_size": 64,
        "usfm_checkpoint": "weights/USFM_latest.pth",
        "usfm_checkpoint_sha256": "d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17",
        "decoder_channels": 256,
        "decoder_norm_groups": 32,
        "internal_identity_tasks": ["A4C", "PSAX"],
        "official_output_sort_tasks": ["A4C", "PSAX"],
        "aop_loss_mode": "correct_shared_vertex",
    }
)


def snapshot_sources(run_dir: Path, config: Path) -> None:
    destination = run_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = set(Path(__file__).resolve().parent.glob("*.py"))
    sources.update(
        {
            config.resolve(),
            TRAINER_SOURCE,
            PROJECT_ROOT / "training/supervised_convnext/dependencies/fivefold_trainer/protocol.py",
            PROJECT_ROOT / "training/supervised_convnext/src/evaluation.py",
            PROJECT_ROOT / "training/supervised_convnext/dependencies/canonical_identity/model.py",
            PROJECT_ROOT / "training/supervised_convnext/dependencies/psax_pair_head/model.py",
            PROJECT_ROOT / "training/supervised_convnext/dependencies/baseline_heatmap/model.py",
            PROJECT_ROOT / "training/supervised_convnext/dependencies/shared/foundus_race_lib.py",
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
                "sha256": trainer.sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    trainer.write_json(run_dir / "source_status.json", records)


trainer.snapshot_sources = snapshot_sources


if __name__ == "__main__":
    trainer.run()
