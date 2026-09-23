"""Run C1 fugacity-only fine-tuning on one registered paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.reporting.artifacts import (
    artifact_sha256,
    atomic_write_json,
    portable_artifact_path,
    resolve_artifact_path,
)
from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.protocols.runner import (
    requested_run_fingerprint,
    result_protocol_name,
    run_paper_experiment,
)
from src.thermoformer.protocols.registry import PROTOCOL_CONFIGS
from src.thermoformer.training.fugacity_finetuning import write_multiseed_physics_finetune_report
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.reporting.aggregation import aggregate_protocol_results

EXPERIMENT_FOLDER = "c1_three_view_vanilla_fugacity"
DEFAULT_PROTOCOL = "overall_binary_ternary"


def output_roots(
    project_root: Path,
    *,
    smoke: bool,
    experiment_folder: str = EXPERIMENT_FOLDER,
) -> tuple[Path, Path, Path]:
    root = project_root / "experiments/run_records/c1_physics_finetune_smoke" if smoke else project_root
    experiment_path = Path("experiments/physics_finetuning") / experiment_folder
    return (
        root / 'experiments/run_records' / experiment_path,
        root / 'models/vle' / experiment_path,
        root / 'experiments/reference_results' / experiment_path,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--protocol",
        choices=sorted(PROTOCOL_CONFIGS),
        default=DEFAULT_PROTOCOL,
    )
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def stage1_checkpoint_path(project_root: Path, protocol: str, seed: int) -> Path:
    """Return the frozen supervised C1 checkpoint for a paper protocol."""

    if protocol == DEFAULT_PROTOCOL:
        return (
            project_root
            / "models/vle/multiview/chemical_attention/formal"
            / f"c1_three_view_vanilla.on.{protocol}/seed_{seed}/best_model.pt"
        )
    return project_root / f"models/vle/{protocol}/seed_{seed}/best_model.pt"


def require_materialized_checkpoint(path: Path) -> None:
    """Fail before training when a Stage-1 checkpoint is absent or an LFS pointer."""

    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"Stage 1 checkpoint is an unhydrated Git-LFS pointer: {path}. "
            "Run `git lfs pull` or regenerate and commit the supervised C1 checkpoint."
        )


def physics_report_path(
    project_root: Path,
    protocol_dir: Path,
    split_protocol: str,
    *,
    smoke: bool,
) -> Path:
    """Keep diagnostics and protocol reports isolated from the overall leaf report."""

    if smoke:
        return protocol_dir / "smoke_results.md"
    if split_protocol == DEFAULT_PROTOCOL:
        return (
            project_root
            / 'experiments/vle/ablation/study_records/fugacity_finetuning/results.md'
        )
    return protocol_dir / "results.md"


def recover_completed_seed_manifest(
    path: Path,
    *,
    expected_status: str,
    expected_protocol: str,
    expected_seed: int,
    expected_evaluation_partition: str,
    expected_request_sha256: str,
    expected_analysis_status: str,
) -> dict[str, object] | None:
    """Validate a completed seed bundle before repairing its final report."""

    if not path.is_file():
        return None
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("status") != expected_status:
        return None
    invariants = {
        "protocol": expected_protocol,
        "seed": expected_seed,
        "evaluation_partition": expected_evaluation_partition,
        "request_sha256": expected_request_sha256,
        "analysis_status": expected_analysis_status,
    }
    for key, expected in invariants.items():
        if candidate.get(key) != expected:
            raise RuntimeError(
                f"Existing completed seed has stale {key}; use --overwrite to rerun"
            )
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("Existing completed seed has no auditable artifacts")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Malformed artifact record: {name}")
        artifact_path = resolve_artifact_path(str(record.get("path", "")))
        if not artifact_path.is_file() or artifact_sha256(artifact_path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"Existing completed seed artifact failed SHA validation: {name}")
    return candidate


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    split_protocol = args.protocol
    seeds = tuple(args.seeds or ((0,) if args.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if args.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    experiment_folder = EXPERIMENT_FOLDER
    config_path = (
        PROJECT_ROOT
        / "configs/vle/ablation/studies/fugacity_finetuning/config.yaml"
    )
    experiment = load_experiment_config(config_path)
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    run_root, checkpoint_root, results_root = output_roots(
        PROJECT_ROOT,
        smoke=args.smoke,
        experiment_folder=experiment_folder,
    )
    overrides = (
        (
            "training.epochs_physics=1",
            "training.minimum_physics_epochs=1",
            "training.solver_iterations_eval=4",
        )
        if args.smoke
        else ()
    )
    protocol = result_protocol_name(experiment.name, split_protocol)
    protocol_dir = results_root / protocol
    expected_evaluation_partition = "validation" if args.smoke else "test"
    analysis_status = "diagnostic" if args.smoke else "confirmatory"
    manifests: list[tuple[int, Path, dict[str, object]]] = []
    comparison_paths: list[Path] = []
    for seed in seeds:
        split_path = PROJECT_ROOT / f"datasets/splits/vle/{split_protocol}/seed_{seed}.json"
        stage1_checkpoint = stage1_checkpoint_path(PROJECT_ROOT, split_protocol, seed)
        require_materialized_checkpoint(stage1_checkpoint)
        seed_manifest_path = protocol_dir / f"seed_{seed}/manifest.json"
        recorded_git_commit = None
        if seed_manifest_path.is_file() and not args.overwrite:
            existing_payload = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
            value = existing_payload.get("git_commit")
            if isinstance(value, str) and value:
                recorded_git_commit = value
        expected_request_sha256 = requested_run_fingerprint(
            config_path,
            split_path,
            seed,
            feature_cache,
            args.device,
            overrides,
            "smoke" if args.smoke else "formal",
            expected_evaluation_partition,
            stage1_checkpoint,
            not args.smoke,
            recorded_git_commit,
            analysis_status,
        )
        existing_manifest = (
            recover_completed_seed_manifest(
                seed_manifest_path,
                expected_status="smoke" if args.smoke else "completed",
                expected_protocol=protocol,
                expected_seed=seed,
                expected_evaluation_partition=expected_evaluation_partition,
                expected_request_sha256=expected_request_sha256,
                expected_analysis_status=analysis_status,
            )
            if not args.overwrite
            else None
        )
        manifest = existing_manifest or run_paper_experiment(
            config_path=config_path,
            split_path=split_path,
            seed=seed,
            run_root=run_root,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            feature_cache=feature_cache,
            device_name=args.device,
            overrides=overrides,
            allow_overwrite=args.overwrite,
            run_kind="smoke" if args.smoke else "formal",
            evaluation_partition=expected_evaluation_partition,
            stage1_checkpoint=stage1_checkpoint,
            aggregate_expected=not args.smoke,
            analysis_status=analysis_status,
        )
        manifests.append((seed, seed_manifest_path, manifest))
        comparison_paths.append(protocol_dir / f"seed_{seed}/stage_comparison.json")

    complete_campaign = (not args.smoke) and seeds == tuple(range(5))
    if not args.smoke and not complete_campaign:
        print(json.dumps({"report": "deferred_until_seeds_0_to_4_complete"}))
        return
    if complete_campaign:
        aggregate_protocol_results(protocol_dir, expected_seeds=tuple(range(5)))
    report_path = physics_report_path(
        PROJECT_ROOT,
        protocol_dir,
        split_protocol,
        smoke=args.smoke,
    )
    _, summary = write_multiseed_physics_finetune_report(
        comparison_paths,
        report_path,
        expected_evaluation_partition=expected_evaluation_partition,
        physics_epochs=1 if args.smoke else experiment.training.epochs_physics,
        protocol_name=split_protocol,
    )
    summary_path = protocol_dir / (
        "smoke_stage_comparison_summary.json"
        if args.smoke
        else "stage_comparison_summary.json"
    )
    atomic_write_json(summary_path, summary)
    report_manifest = {
        "status": "smoke" if args.smoke else "completed",
        "protocol": protocol,
        "seeds": list(seeds),
        "selection_partition": "validation",
        "evaluation_partition": "validation" if args.smoke else "test",
        "selected_stage_counts": summary["selected_stage_counts"],
        "analysis_status": analysis_status,
        "run_manifests": [
            {
                "seed": seed,
                "path": portable_artifact_path(path),
                "sha256": artifact_sha256(path),
            }
            for seed, path, _ in manifests
        ],
        "stage_comparisons": [
            {
                "seed": seed,
                "path": portable_artifact_path(path),
                "sha256": artifact_sha256(path),
            }
            for seed, path in zip(seeds, comparison_paths)
        ],
        "stage_comparison_summary": {
            "path": portable_artifact_path(summary_path),
            "sha256": artifact_sha256(summary_path),
        },
        "report": {
            "path": portable_artifact_path(report_path),
            "sha256": artifact_sha256(report_path),
        },
    }
    if complete_campaign:
        aggregate_manifest = protocol_dir / "aggregate_manifest.json"
        report_manifest["aggregate_manifest"] = {
            "path": portable_artifact_path(aggregate_manifest),
            "sha256": artifact_sha256(aggregate_manifest),
        }
    atomic_write_json(protocol_dir / "report_manifest.json", report_manifest)
    print(json.dumps(report_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
