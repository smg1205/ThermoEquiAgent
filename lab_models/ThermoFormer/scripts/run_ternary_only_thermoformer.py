"""Run strict ternary-train to ternary-test ThermoFormer experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import result_protocol_name, run_paper_experiment
from src.thermoformer.reporting.aggregation import aggregate_protocol_results


SPLIT_PROTOCOL = "overall_ternary"
STAGE0_CONFIG = (
    PROJECT_ROOT
    / "configs/vle/comparison/studies/ternary_only/thermoformer_stage0.yaml"
)
DIRECT_GE_CONFIG = (
    PROJECT_ROOT
    / "configs/vle/comparison/studies/ternary_only/thermoformer_direct_ge.yaml"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def _seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    seeds = tuple(arguments.seeds or ((0,) if arguments.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if arguments.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    return seeds


def _roots(smoke: bool) -> dict[str, Path]:
    if smoke:
        base = PROJECT_ROOT / "experiments/run_records/smoke/ternary_only_comparison"
        return {
            "stage0_run": base / "stage0/runs",
            "stage0_checkpoint": base / "stage0/checkpoints",
            "stage0_result": base / "stage0/results",
            "direct_run": base / "direct_ge/runs",
            "direct_checkpoint": base / "direct_ge/checkpoints",
            "direct_result": base / "direct_ge/results",
        }
    base = PROJECT_ROOT
    return {
        "stage0_run": base / "experiments/vle/generalization/training_records/comparisons/ternary_only/stage0",
        "stage0_checkpoint": base / "models/vle/experiments/comparisons/ternary_only/stage0",
        "stage0_result": base / "experiments/vle/generalization/evaluations/comparisons/ternary_only/stage0",
        "direct_run": base / "experiments/vle/generalization/training_records/comparisons/ternary_only/direct_ge",
        "direct_checkpoint": base / "models/vle/experiments/comparisons/ternary_only/direct_ge",
        "direct_result": base / "experiments/vle/generalization/evaluations/comparisons/ternary_only/direct_ge",
    }


def _completed(manifest_path: Path, checkpoint_path: Path, *, smoke: bool) -> bool:
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"smoke"} if smoke else {"completed"}
    return payload.get("status") in expected


def _smoke_overrides(*, direct_ge: bool) -> tuple[str, ...]:
    common = (
        "protocol.evaluation_partition=validation",
        "training.epochs_supervised=1",
        "training.minimum_supervised_epochs=1",
        "training.batch_size=32",
        "training.solver_iterations_eval=8",
    )
    if not direct_ge:
        return common
    return common + (
        "direct_ge_supervision.pretrain_epochs=1",
        "direct_ge_supervision.fugacity_epochs=1",
    )


def _run_stage(
    *,
    config_path: Path,
    split_path: Path,
    seed: int,
    roots: dict[str, Path],
    prefix: str,
    device: str,
    smoke: bool,
    overwrite: bool,
    stage0_checkpoint: Path | None = None,
    stage0_manifest: Path | None = None,
) -> tuple[Path, Path]:
    experiment = load_experiment_config(config_path)
    protocol = result_protocol_name(experiment.name, SPLIT_PROTOCOL)
    result_dir = roots[f"{prefix}_result"] / protocol / f"seed_{seed}"
    checkpoint = roots[f"{prefix}_checkpoint"] / protocol / f"seed_{seed}/best_model.pt"
    manifest = result_dir / "manifest.json"
    if _completed(manifest, checkpoint, smoke=smoke) and not overwrite:
        print(f"[ternary-only] {prefix} seed {seed}: verified existing result; skipping", flush=True)
        return manifest, checkpoint
    if manifest.exists() and not overwrite:
        raise FileExistsError(
            f"Incomplete or incompatible result exists: {manifest}; pass --overwrite"
        )
    overrides = _smoke_overrides(direct_ge=prefix == "direct") if smoke else ()
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    print(f"[ternary-only] {prefix} seed {seed}: starting", flush=True)
    run_paper_experiment(
        config_path=config_path,
        split_path=split_path,
        seed=seed,
        run_root=roots[f"{prefix}_run"],
        checkpoint_root=roots[f"{prefix}_checkpoint"],
        results_root=roots[f"{prefix}_result"],
        feature_cache=feature_cache,
        device_name=device,
        overrides=overrides,
        allow_overwrite=overwrite,
        run_kind="smoke" if smoke else "formal",
        evaluation_partition="validation" if smoke else "test",
        stage1_checkpoint=stage0_checkpoint,
        generated_stage1_manifest=stage0_manifest if not smoke else None,
        aggregate_expected=False,
        analysis_status="diagnostic" if smoke else "confirmatory",
    )
    print(f"[ternary-only] {prefix} seed {seed}: completed", flush=True)
    return manifest, checkpoint


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    seeds = _seeds(arguments)
    roots = _roots(arguments.smoke)
    outputs: list[dict[str, object]] = []
    for seed in seeds:
        split = PROJECT_ROOT / f"datasets/splits/vle/{SPLIT_PROTOCOL}/seed_{seed}.json"
        if not split.is_file():
            raise FileNotFoundError(
                f"Missing registered ternary split: {split}. Run scripts/generate_splits.py first."
            )
        stage0_manifest, stage0_checkpoint = _run_stage(
            config_path=STAGE0_CONFIG,
            split_path=split,
            seed=seed,
            roots=roots,
            prefix="stage0",
            device=arguments.device,
            smoke=arguments.smoke,
            overwrite=arguments.overwrite,
        )
        direct_manifest, _ = _run_stage(
            config_path=DIRECT_GE_CONFIG,
            split_path=split,
            seed=seed,
            roots=roots,
            prefix="direct",
            device=arguments.device,
            smoke=arguments.smoke,
            overwrite=arguments.overwrite,
            stage0_checkpoint=stage0_checkpoint,
            stage0_manifest=stage0_manifest,
        )
        outputs.append(
            {
                "seed": seed,
                "stage0_manifest": str(stage0_manifest),
                "direct_ge_manifest": str(direct_manifest),
            }
        )
    if not arguments.smoke and seeds == tuple(range(5)):
        for config_path, prefix in ((STAGE0_CONFIG, "stage0"), (DIRECT_GE_CONFIG, "direct")):
            experiment = load_experiment_config(config_path)
            protocol = result_protocol_name(experiment.name, SPLIT_PROTOCOL)
            aggregate_protocol_results(
                roots[f"{prefix}_result"] / protocol,
                expected_seeds=tuple(range(5)),
            )
    print(json.dumps({"status": "completed", "protocol": SPLIT_PROTOCOL, "runs": outputs}, indent=2))


if __name__ == "__main__":
    main()
