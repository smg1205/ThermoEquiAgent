"""Run retained five-seed C1 molecular-view ablations on the joint test."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.multiview_protocols import (
    MULTIVIEW_SEEDS,
    MULTIVIEW_VARIANTS,
    PREDICTIVE_PROTOCOLS,
    PREDICTIVE_VARIANTS,
)
from src.thermoformer.protocols.runner import result_protocol_name, run_paper_experiment
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.reporting.aggregation import aggregate_protocol_results


def release_accelerator_memory() -> None:
    """Release per-run CUDA caches before the next experiment."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--variant", action="append", choices=sorted(MULTIVIEW_VARIANTS))
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT)
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    variants = tuple(args.variant or PREDICTIVE_VARIANTS)
    protocols = PREDICTIVE_PROTOCOLS
    seeds = MULTIVIEW_SEEDS
    artifact_root = args.artifact_root.resolve()
    namespace = "predictive"
    run_root = artifact_root / 'experiments/vle/ablation/training_records' / namespace
    checkpoint_root = artifact_root / 'models/vle/multiview' / namespace
    results_root = artifact_root / 'experiments/vle/ablation/multiview' / namespace / 'experiments/run_records'
    overrides: tuple[str, ...] = ()
    for variant_id in variants:
        variant = MULTIVIEW_VARIANTS[variant_id]
        config_path = PROJECT_ROOT / variant.config
        experiment = load_experiment_config(config_path, overrides)
        feature_cache = artifact_root / "cache" / encoder_cache_filename(experiment.encoder)
        for split_protocol in protocols:
            protocol = result_protocol_name(experiment.name, split_protocol)
            for seed in seeds:
                try:
                    manifest = run_paper_experiment(
                        config_path=config_path,
                        split_path=PROJECT_ROOT / 'datasets/splits/vle' / split_protocol / f"seed_{seed}.json",
                        seed=seed,
                        run_root=run_root,
                        checkpoint_root=checkpoint_root,
                        results_root=results_root,
                        feature_cache=feature_cache,
                        device_name=args.device,
                        overrides=overrides,
                        allow_overwrite=args.overwrite,
                        run_kind="formal",
                    )
                finally:
                    release_accelerator_memory()
                print(json.dumps({
                    "variant": variant_id,
                    "protocol": split_protocol,
                    "seed": seed,
                    "status": manifest["status"],
                    "training_seconds": manifest["training_seconds"],
                }))
            aggregate_protocol_results(
                results_root / protocol,
                expected_seeds=MULTIVIEW_SEEDS,
            )


if __name__ == "__main__":
    main()
