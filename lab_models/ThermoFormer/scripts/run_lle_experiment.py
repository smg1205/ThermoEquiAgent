"""Run one isolated LLE experiment from a task_mode=lle configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.protocols.lle_runner import run_lle_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    cache = PROJECT_ROOT / "cache" / encoder_cache_filename(config.encoder)
    artifact_root = PROJECT_ROOT / 'experiments/run_records' / ("single_smoke" if args.smoke else "experiments") / "lle"
    checkpoint_root = (PROJECT_ROOT / 'experiments/run_records/single_smoke/checkpoints/lle'
                       if args.smoke else PROJECT_ROOT / 'models/vle')
    results_root = artifact_root / 'experiments/reference_results' if args.smoke else PROJECT_ROOT / 'experiments/vle/generalization/evaluations/lle'
    manifest = run_lle_experiment(
        args.config, args.split, args.seed, artifact_root, checkpoint_root, results_root, cache,
        device_name=args.device, run_kind="smoke" if args.smoke else "formal",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
