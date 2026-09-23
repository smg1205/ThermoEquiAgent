"""Run the C1 cumulative three-stage ablation on the joint VLE protocol.

Each seed trains one deterministic trajectory: supervised Stage 0, direct-GE
Stage 1, joint-VLE Stage 2, and fugacity-constrained Stage 3. The underlying
runner snapshots every stage's validation-selected checkpoint and evaluates
those frozen snapshots only after checkpoint selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Literal, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_c1_physics_finetune import recover_completed_seed_manifest
from src.thermoformer.configuration import ExperimentConfig, load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import (
    requested_run_fingerprint,
    result_protocol_name,
    run_paper_experiment,
)
from src.thermoformer.reporting.aggregation import aggregate_protocol_results


SPLIT_PROTOCOL = "overall_binary_ternary"
EXPERIMENT_NAMESPACE = Path(
    "configs/vle/ablation/studies/three_stage_training/overall_binary_ternary"
)
SMOKE_NAMESPACE = Path("experiments/run_records/smoke/three_stage_training/overall_binary_ternary")
STAGES = ("stage0", "stage1", "stage2", "stage3")
STAGE0_NAME = "c1_three_view_vanilla_stage0"
THREE_STAGE_NAME = "c1_three_view_vanilla_three_stage"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--report-only", action="store_true")
    return value


def resolve_seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    """Return a registered deterministic seed sequence."""

    seeds = tuple(arguments.seeds or ((0,) if arguments.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if arguments.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    return seeds


def stage0_config_path(project_root: Path) -> Path:
    return project_root / EXPERIMENT_NAMESPACE / "stage0.yaml"


def three_stage_config_path(project_root: Path) -> Path:
    return project_root / EXPERIMENT_NAMESPACE / "three_stage.yaml"


def split_path(project_root: Path, seed: int) -> Path:
    return project_root / 'datasets/splits/vle' / SPLIT_PROTOCOL / f"seed_{seed}.json"


def protocol_name(name: str) -> str:
    return result_protocol_name(name, SPLIT_PROTOCOL)


def artifact_roots(project_root: Path, *, smoke: bool) -> tuple[Path, Path, Path]:
    """Keep smoke, checkpoint, and confirmatory artifact roots disjoint."""

    if smoke:
        root = project_root / SMOKE_NAMESPACE
        return root / 'experiments/run_records', root / 'models/vle', root / 'experiments/reference_results'
    return (
        project_root / 'experiments/run_records' / EXPERIMENT_NAMESPACE,
        project_root / 'models/vle' / EXPERIMENT_NAMESPACE,
        project_root / 'experiments/reference_results' / EXPERIMENT_NAMESPACE,
    )


def smoke_overrides(stage: Literal["stage0", "three_stage"]) -> tuple[str, ...]:
    common = (
        "protocol.evaluation_partition=validation",
        "training.epochs_supervised=1",
        "training.minimum_supervised_epochs=1",
        "training.batch_size=32",
        "training.solver_iterations_eval=8",
    )
    if stage == "stage0":
        return common
    return (*common, "direct_ge_supervision.pretrain_epochs=1", "direct_ge_supervision.fugacity_epochs=1")


def _validate_config(experiment: ExperimentConfig, *, stage: str) -> None:
    if experiment.protocol.registered_splits != (SPLIT_PROTOCOL,):
        raise ValueError("The cumulative ablation must use only overall_binary_ternary")
    if experiment.protocol.seeds != tuple(range(5)):
        raise ValueError("The cumulative ablation requires registered seeds 0--4")
    if experiment.encoder.chemical_attention_bias or experiment.encoder.context_pair_interaction:
        raise ValueError("The C1 ablation must remain a vanilla Transformer")
    if not (
        experiment.encoder.use_rdkit_descriptors
        and experiment.encoder.use_unimol
        and experiment.encoder.use_functional_groups
    ):
        raise ValueError("The C1 ablation requires all three molecular views")
    if stage == "stage0":
        if experiment.name != STAGE0_NAME or experiment.direct_ge_supervision is not None:
            raise ValueError("Stage 0 must be a supervised C1 reference")
        if experiment.protocol.evaluation_partition != "validation":
            raise ValueError("Stage 0 selection must use validation")
        return
    direct = experiment.direct_ge_supervision
    if experiment.name != THREE_STAGE_NAME or direct is None:
        raise ValueError("The three-stage configuration is incomplete")
    if (
        experiment.protocol.evaluation_partition != "test"
        or direct.pretrain_epochs != 20
        or experiment.training.epochs_supervised != 80
        or direct.fugacity_epochs != 10
        or direct.fugacity_weight != 0.01
    ):
        raise ValueError("The three-stage configuration does not match the approved budget")


def _manifest_path(results_root: Path, name: str, seed: int) -> Path:
    return results_root / protocol_name(name) / f"seed_{seed}" / "manifest.json"


def _checkpoint_path(checkpoint_root: Path, name: str, seed: int) -> Path:
    return checkpoint_root / protocol_name(name) / f"seed_{seed}" / "best_model.pt"


def _recorded_commit(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("git_commit")
    return value if isinstance(value, str) and value else None


def _recover(
    *,
    manifest_path: Path,
    config_path: Path,
    split: Path,
    seed: int,
    feature_cache: Path,
    device: str,
    overrides: tuple[str, ...],
    run_kind: str,
    evaluation_partition: str,
    stage0_checkpoint: Path | None,
    analysis_status: str,
) -> dict[str, object] | None:
    request = requested_run_fingerprint(
        config_path,
        split,
        seed,
        feature_cache,
        device,
        overrides,
        run_kind,
        evaluation_partition,
        stage0_checkpoint,
        False,
        _recorded_commit(manifest_path),
        analysis_status,
    )
    return recover_completed_seed_manifest(
        manifest_path,
        expected_status="smoke" if run_kind == "smoke" else "completed",
        expected_protocol=protocol_name(load_experiment_config(config_path).name),
        expected_seed=seed,
        expected_evaluation_partition=evaluation_partition,
        expected_request_sha256=request,
        expected_analysis_status=analysis_status,
    )


def _require_stage_evidence(manifest: Mapping[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("Three-stage run has no artifact provenance")
    required = {"stage_comparison"}
    for stage in STAGES:
        required.update((stage + "_checkpoint", stage + "_predictions"))
    missing = sorted(required.difference(artifacts))
    if missing:
        raise RuntimeError("Three-stage run lacks stage-local evidence: " + ", ".join(missing))


def _run_stage(
    *,
    config_path: Path,
    name: str,
    split: Path,
    seed: int,
    roots: tuple[Path, Path, Path],
    feature_cache: Path,
    device: str,
    overrides: tuple[str, ...],
    smoke: bool,
    overwrite: bool,
    evaluation_partition: str,
    stage0_checkpoint: Path | None = None,
    stage0_manifest: Path | None = None,
) -> dict[str, object]:
    manifest_path = _manifest_path(roots[2], name, seed)
    run_kind = "smoke" if smoke else "formal"
    analysis_status = "diagnostic" if smoke else "confirmatory"
    recovered = None
    if not overwrite and manifest_path.is_file():
        recovered = _recover(
            manifest_path=manifest_path,
            config_path=config_path,
            split=split,
            seed=seed,
            feature_cache=feature_cache,
            device=device,
            overrides=overrides,
            run_kind=run_kind,
            evaluation_partition=evaluation_partition,
            stage0_checkpoint=stage0_checkpoint,
            analysis_status=analysis_status,
        )
        if recovered is None:
            raise RuntimeError(
                f"Interrupted or stale seed {seed} artifacts found at {manifest_path}; use --overwrite to replace them"
            )
    return recovered or run_paper_experiment(
        config_path=config_path,
        split_path=split,
        seed=seed,
        run_root=roots[0],
        checkpoint_root=roots[1],
        results_root=roots[2],
        feature_cache=feature_cache,
        device_name=device,
        overrides=overrides,
        allow_overwrite=overwrite,
        run_kind=run_kind,
        evaluation_partition=evaluation_partition,
        stage1_checkpoint=stage0_checkpoint,
        generated_stage1_manifest=stage0_manifest,
        aggregate_expected=False,
        analysis_status=analysis_status,
    )


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _write_report(project_root: Path) -> None:
    from scripts.build_three_stage_training_ablation_report import write_report

    write_report(project_root)


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    if arguments.report_only:
        if arguments.smoke or arguments.seeds is not None or arguments.overwrite:
            raise ValueError("--report-only cannot be combined with training options")
        _write_report(PROJECT_ROOT)
        return
    seeds = resolve_seeds(arguments)
    stage0_path = stage0_config_path(PROJECT_ROOT)
    three_path = three_stage_config_path(PROJECT_ROOT)
    stage0_config = load_experiment_config(stage0_path)
    three_config = load_experiment_config(three_path)
    _validate_config(stage0_config, stage="stage0")
    _validate_config(three_config, stage="three_stage")
    roots = artifact_roots(PROJECT_ROOT, smoke=arguments.smoke)
    stage0_roots = tuple(path / "stage0" for path in roots)
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(three_config.encoder)
    evaluation_partition = "validation" if arguments.smoke else "test"
    for seed in seeds:
        split = split_path(PROJECT_ROOT, seed)
        if not split.is_file():
            raise FileNotFoundError(f"Missing registered split: {split}")
        stage0_manifest = _run_stage(
            config_path=stage0_path,
            name=stage0_config.name,
            split=split,
            seed=seed,
            roots=stage0_roots,
            feature_cache=feature_cache,
            device=arguments.device,
            overrides=smoke_overrides("stage0") if arguments.smoke else (),
            smoke=arguments.smoke,
            overwrite=arguments.overwrite,
            evaluation_partition="validation",
        )
        stage0_checkpoint = _checkpoint_path(stage0_roots[1], stage0_config.name, seed)
        if not stage0_checkpoint.is_file():
            raise RuntimeError("Stage 0 checkpoint was not materialized")
        try:
            manifest = _run_stage(
                config_path=three_path,
                name=three_config.name,
                split=split,
                seed=seed,
                roots=roots,
                feature_cache=feature_cache,
                device=arguments.device,
                overrides=smoke_overrides("three_stage") if arguments.smoke else (),
                smoke=arguments.smoke,
                overwrite=arguments.overwrite,
                evaluation_partition=evaluation_partition,
                stage0_checkpoint=stage0_checkpoint,
                stage0_manifest=_manifest_path(stage0_roots[2], stage0_config.name, seed),
            )
            _require_stage_evidence(manifest)
        finally:
            _release_accelerator_memory()
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": manifest["status"],
                    "selected_stage": manifest.get("selected_stage"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not arguments.smoke and seeds == tuple(range(5)):
        result_dir = roots[2] / protocol_name(three_config.name)
        aggregate_protocol_results(result_dir, expected_seeds=tuple(range(5)))
        _write_report(PROJECT_ROOT)


if __name__ == "__main__":
    main()
