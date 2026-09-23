"""Run fixed ThermoFormer registered protocols and aggregate complete seed sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.protocols.runner import requested_run_fingerprint, run_paper_experiment
from src.thermoformer.protocols.registry import PAPER_SEEDS, PROTOCOL_CONFIGS
from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.reporting.aggregation import aggregate_protocol_results


def smoke_overrides() -> tuple[str, ...]:
    """Keep registered-suite smoke runs on the supervised Stage-1 path."""

    return (
        "training.epochs_supervised=2",
        "training.epochs_physics=0",
        "training.minimum_physics_epochs=0",
        "training.solver_iterations_eval=8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", action="append", choices=sorted(PROTOCOL_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocols = args.protocol or list(PROTOCOL_CONFIGS)
    if args.smoke:
        run_root = PROJECT_ROOT / 'experiments/run_records/suite_smoke/registered'
        checkpoint_root = PROJECT_ROOT / 'experiments/run_records/suite_smoke/checkpoints'
        results_root = PROJECT_ROOT / 'experiments/run_records/suite_smoke/results'
        overrides = smoke_overrides()
    else:
        run_root = PROJECT_ROOT / 'experiments/run_records/registered'
        checkpoint_root = PROJECT_ROOT / 'models/vle'
        results_root = PROJECT_ROOT / 'experiments/reference_results'
        overrides = ()
    for protocol in protocols:
        config = PROJECT_ROOT / PROTOCOL_CONFIGS[protocol]
        experiment = load_experiment_config(config, overrides)
        feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
        for seed in args.seeds:
            manifest_path = results_root / protocol / f"seed_{seed}" / "manifest.json"
            if manifest_path.exists() and not args.overwrite:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_status = "smoke" if args.smoke else "completed"
                if manifest.get("status") == expected_status:
                    expected_request = requested_run_fingerprint(
                        config,
                        PROJECT_ROOT / 'datasets/splits/vle' / protocol / f"seed_{seed}.json",
                        seed,
                        feature_cache,
                        args.device,
                        overrides,
                        "smoke" if args.smoke else "formal",
                    )
                    if manifest.get("request_sha256") != expected_request:
                        raise RuntimeError(
                            f"Existing result for {protocol}/seed_{seed} has stale provenance; "
                            "rerun explicitly with --overwrite"
                        )
                    print(json.dumps({"protocol": protocol, "seed": seed, "status": "skipped_existing"}))
                    continue
            manifest = run_paper_experiment(
                config_path=config,
                split_path=PROJECT_ROOT / 'datasets/splits/vle' / protocol / f"seed_{seed}.json",
                seed=seed,
                run_root=run_root,
                checkpoint_root=checkpoint_root,
                results_root=results_root,
                feature_cache=feature_cache,
                device_name=args.device,
                overrides=overrides,
                allow_overwrite=args.overwrite,
                run_kind="smoke" if args.smoke else "formal",
            )
            print(
                json.dumps(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "status": manifest["status"],
                        "seconds": manifest["training_seconds"],
                    },
                    ensure_ascii=False,
                )
            )
        if not args.smoke and set(args.seeds) == set(PAPER_SEEDS):
            aggregate_protocol_results(results_root / protocol, expected_seeds=args.seeds)


if __name__ == "__main__":
    main()
