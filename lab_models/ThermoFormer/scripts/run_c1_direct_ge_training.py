"""Run the registered five-seed C1 direct-GE three-stage campaign."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_c1_physics_finetune import (
    recover_completed_seed_manifest,
    require_materialized_checkpoint,
)
from src.thermoformer.configuration import load_experiment_config
from src.thermoformer.features import encoder_cache_filename
from src.thermoformer.protocols.runner import (
    requested_run_fingerprint,
    result_protocol_name,
    run_paper_experiment,
)
from src.thermoformer.reporting.aggregation import aggregate_protocol_results
from src.thermoformer.reporting.artifacts import (
    artifact_sha256,
    atomic_write_json,
    portable_artifact_path,
)
from src.thermoformer.training.direct_ge_reporting import (
    load_joint_baseline_metrics,
    summarize_direct_ge_campaign,
    write_campaign_summary,
    write_direct_ge_campaign_report,
    write_seed_metrics_csv,
)


SPLIT_PROTOCOL = "overall_binary_ternary"
SUPPORTED_PROTOCOLS = ("overall_binary_ternary", "overall_binary")
COMPATIBLE_TRAINING_GIT_COMMITS = (
    "d9b4d36e3c05a32610e1cb84000eea006ec05b10",
    "3f282c612060e536ab40d1f16181f35f71ca0bb6",
)
MIXED_COMMIT_JUSTIFICATION = (
    "Seed 0 predates the live progress reporter used by seeds 1--4; the audited "
    "diff changes only campaign/reporting code and non-numerical progress output."
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--protocol",
        choices=SUPPORTED_PROTOCOLS,
        default=SPLIT_PROTOCOL,
    )
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    value.add_argument("--seeds", type=int, nargs="+", default=None)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def resolve_seeds(arguments: argparse.Namespace) -> tuple[int, ...]:
    seeds = tuple(arguments.seeds or ((0,) if arguments.smoke else range(5)))
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(range(5)):
        raise ValueError("Seeds must be a unique subset of 0--4")
    if arguments.smoke and seeds != (0,):
        raise ValueError("Smoke mode is restricted to seed 0")
    return seeds


def split_path(
    project_root: Path,
    seed: int,
    protocol: str = SPLIT_PROTOCOL,
) -> Path:
    return project_root / f"datasets/splits/vle/{protocol}/seed_{seed}.json"


def stage0_checkpoint_path(
    project_root: Path,
    seed: int,
    protocol: str = SPLIT_PROTOCOL,
) -> Path:
    if protocol == "overall_binary":
        return project_root / f"models/vle/overall_binary/seed_{seed}/best_model.pt"
    return (
        project_root
        / "models/vle/multiview/chemical_attention/formal"
        / f"c1_three_view_vanilla.on.{protocol}/seed_{seed}/best_model.pt"
    )


def experiment_config_path(
    project_root: Path,
    protocol: str = SPLIT_PROTOCOL,
) -> Path:
    if protocol == "overall_binary":
        return (
            project_root
            / "configs/vle/comparison/studies/binary_only/thermoformer_direct_ge.yaml"
        )
    return project_root / "configs/vle/ablation/studies/direct_ge_supervision/config.yaml"


def output_roots(
    project_root: Path,
    *,
    smoke: bool,
    split_protocol: str = SPLIT_PROTOCOL,
) -> tuple[Path, Path, Path]:
    if split_protocol == "overall_binary":
        if smoke:
            root = project_root / "experiments/run_records/smoke/binary_only_direct_ge"
            return root / "paper", root / 'models/vle', root / 'experiments/reference_results'
        return (
            project_root / "experiments/vle/generalization/training_records/comparisons/binary_only/direct_ge",
            project_root / "models/vle/experiments/comparisons/binary_only/direct_ge",
            project_root / "experiments/vle/generalization/evaluations/comparisons/binary_only/direct_ge",
        )
    if smoke:
        root = project_root / "experiments/run_records/smoke/direct_ge_supervision"
        return root / "paper", root / 'models/vle', root / 'experiments/reference_results'
    return (
        project_root / "experiments/vle/generalization/training_records/ablations/direct_ge_supervision",
        project_root / "models/vle/experiments/ablations/direct_ge_supervision",
        project_root / "experiments/vle/generalization/evaluations/ablations/direct_ge_supervision",
    )


def archive_interrupted_seed_outputs(
    *,
    run_root: Path,
    checkpoint_root: Path,
    results_root: Path,
    protocol: str,
    seed: int,
    archive_label: str | None = None,
) -> tuple[Path, ...]:
    """Move an interrupted seed aside so a clean restart cannot mix artifacts."""
    manifest_path = results_root / protocol / f"seed_{seed}/manifest.json"
    if not manifest_path.is_file():
        return ()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status != "running":
        raise RuntimeError(
            f"Seed {seed} manifest status is {status!r}; will not archive a "
            "non-running result automatically"
        )
    label = archive_label or f"interrupted_{datetime.now():%Y%m%d_%H%M%S}"
    sources = tuple(
        root / protocol / f"seed_{seed}"
        for root in (run_root, checkpoint_root, results_root)
    )
    moves = tuple(
        (source, source.with_name(f"seed_{seed}.{label}"))
        for source in sources
        if source.exists()
    )
    collisions = [destination for _, destination in moves if destination.exists()]
    if collisions:
        raise FileExistsError(
            "Interrupted-run archive already exists: "
            + ", ".join(str(path) for path in collisions)
        )
    for source, destination in moves:
        source.rename(destination)
    return tuple(destination for _, destination in moves)


def smoke_overrides() -> tuple[str, ...]:
    return (
        "protocol.evaluation_partition=validation",
        "direct_ge_supervision.pretrain_epochs=1",
        "training.epochs_supervised=1",
        "training.minimum_supervised_epochs=1",
        "direct_ge_supervision.fugacity_epochs=1",
        "training.batch_size=32",
        "training.solver_iterations_eval=8",
    )


def _test_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(value) for value in payload["partitions"]["test"])


def _baseline_specs(project_root: Path) -> tuple[dict[str, object], ...]:
    root = project_root / "experiments/vle/comparison/reference_evaluations/machine_learning/formal"
    return (
        {
            "label": "SPT-NRTL adapted",
            "group": "same registered train/split",
            "path": root / "spt_nrtl_adapted.on.joint_train_joint_test",
        },
        {
            "label": "HANNA adapted",
            "group": "external frozen representation",
            "path": root / "hanna_joint_adapted.on.joint_train_joint_test",
            "id_fallback": True,
        },
        {
            "label": "TeNNet-SAC adapted",
            "group": "external frozen representation",
            "path": root / "tennet_sac_joint_adapted.on.joint_train_joint_test",
            "id_fallback": True,
        },
        {
            "label": "HANNA official",
            "group": "external pretrained reference",
            "path": root / "hanna.on.official_pretrained_to_joint_test",
        },
        {
            "label": "TeNNet-SAC official",
            "group": "external pretrained reference",
            "path": root / "tennet_sac.on.external_fixed_to_joint_test",
        },
        {
            "label": "SPT-NRTL external",
            "group": "external database reference",
            "path": root / "spt_nrtl.on.external_fixed_to_joint_test",
        },
        {
            "label": "SolvGNN",
            "group": "298 K partial-coverage reference",
            "path": root / "solvgnn.on.joint_train_joint_test",
            "allow_partial": True,
        },
    )


def _write_formal_outputs(
    protocol_dir: Path,
    manifests: list[tuple[int, Path, dict[str, object]]],
    comparison_paths: list[Path],
) -> dict[str, object]:
    expected_seeds = tuple(range(5))
    aggregate_protocol_results(
        protocol_dir,
        expected_seeds=expected_seeds,
        compatible_training_git_commits=COMPATIBLE_TRAINING_GIT_COMMITS,
        mixed_commit_justification=MIXED_COMMIT_JUSTIFICATION,
    )
    summary = summarize_direct_ge_campaign(
        comparison_paths,
        manifest_paths=[path for _, path, _ in manifests],
        expected_seeds=expected_seeds,
    )
    provenance = {
        seed: {
            "dataset_sha256": manifest["dataset_sha256"],
            "split_sha256": manifest["split_sha256"],
            "test_sample_ids": _test_ids(split_path(PROJECT_ROOT, seed)),
        }
        for seed, _, manifest in manifests
    }
    baseline_rows = []
    for spec in _baseline_specs(PROJECT_ROOT):
        baseline_rows.append(
            {
                "label": spec["label"],
                "group": spec["group"],
                "summary": load_joint_baseline_metrics(
                    spec["path"],
                    expected_provenance=provenance,
                    expected_seeds=expected_seeds,
                    allow_registered_id_fallback=bool(spec.get("id_fallback", False)),
                    allow_partial=bool(spec.get("allow_partial", False)),
                ),
            }
        )
    summary_path = protocol_dir / "direct_ge_campaign_summary.json"
    seed_csv_path = protocol_dir / "direct_ge_seed_metrics.csv"
    report_path = (
        PROJECT_ROOT / "experiments/vle/ablation/study_records/direct_ge_supervision/results.md"
    )
    write_campaign_summary(summary, summary_path)
    write_seed_metrics_csv(summary, seed_csv_path)
    write_direct_ge_campaign_report(summary, baseline_rows, report_path)
    report_manifest = {
        "status": "completed",
        "protocol": protocol_dir.name,
        "seeds": list(expected_seeds),
        "selection_partition": "validation",
        "evaluation_partition": "test",
        "selected_stage_counts": summary["selected_stage_counts"],
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
            for seed, path in zip(expected_seeds, comparison_paths)
        ],
        "outputs": {
            name: {
                "path": portable_artifact_path(path),
                "sha256": artifact_sha256(path),
            }
            for name, path in {
                "summary": summary_path,
                "seed_metrics": seed_csv_path,
                "report": report_path,
                "aggregate_manifest": protocol_dir / "aggregate_manifest.json",
            }.items()
        },
        "baseline_inputs": [
            {
                "label": row["label"],
                "group": row["group"],
                "provenance_status": row["summary"]["provenance_status"],
            }
            for row in baseline_rows
        ],
    }
    atomic_write_json(protocol_dir / "report_manifest.json", report_manifest)
    return report_manifest


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    seeds = resolve_seeds(args)
    split_protocol = args.protocol
    config_path = experiment_config_path(PROJECT_ROOT, split_protocol)
    experiment = load_experiment_config(config_path)
    feature_cache = PROJECT_ROOT / "cache" / encoder_cache_filename(experiment.encoder)
    overrides = smoke_overrides() if args.smoke else ()
    run_root, checkpoint_root, results_root = output_roots(
        PROJECT_ROOT,
        smoke=args.smoke,
        split_protocol=split_protocol,
    )
    protocol = result_protocol_name(experiment.name, split_protocol)
    protocol_dir = results_root / protocol
    evaluation_partition = "validation" if args.smoke else "test"
    analysis_status = "diagnostic" if args.smoke else "confirmatory"
    manifests: list[tuple[int, Path, dict[str, object]]] = []
    comparison_paths: list[Path] = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(
            f"[campaign] seed {seed} ({seed_index}/{len(seeds)}): checking artifacts",
            flush=True,
        )
        current_split = split_path(PROJECT_ROOT, seed, split_protocol)
        baseline = stage0_checkpoint_path(PROJECT_ROOT, seed, split_protocol)
        require_materialized_checkpoint(baseline)
        manifest_path = protocol_dir / f"seed_{seed}/manifest.json"
        recorded_commit = None
        if manifest_path.is_file() and not args.overwrite:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(recorded.get("git_commit"), str):
                recorded_commit = recorded["git_commit"]
        request_sha256 = requested_run_fingerprint(
            config_path,
            current_split,
            seed,
            feature_cache,
            args.device,
            overrides,
            "smoke" if args.smoke else "formal",
            evaluation_partition,
            baseline,
            False,
            recorded_commit,
            analysis_status,
        )
        recovered = (
            recover_completed_seed_manifest(
                manifest_path,
                expected_status="smoke" if args.smoke else "completed",
                expected_protocol=protocol,
                expected_seed=seed,
                expected_evaluation_partition=evaluation_partition,
                expected_request_sha256=request_sha256,
                expected_analysis_status=analysis_status,
            )
            if not args.overwrite
            else None
        )
        if recovered is not None:
            print(
                f"[campaign] seed {seed}: completed artifacts verified; skipping",
                flush=True,
            )
        elif manifest_path.is_file() and not args.overwrite:
            archived = archive_interrupted_seed_outputs(
                run_root=run_root,
                checkpoint_root=checkpoint_root,
                results_root=results_root,
                protocol=protocol,
                seed=seed,
            )
            print(
                f"[campaign] seed {seed}: archived interrupted outputs "
                f"({len(archived)} directories); restarting cleanly",
                flush=True,
            )
        else:
            print(f"[campaign] seed {seed}: starting training", flush=True)
        manifest = recovered or run_paper_experiment(
            config_path=config_path,
            split_path=current_split,
            seed=seed,
            run_root=run_root,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            feature_cache=feature_cache,
            device_name=args.device,
            overrides=overrides,
            allow_overwrite=args.overwrite,
            run_kind="smoke" if args.smoke else "formal",
            evaluation_partition=evaluation_partition,
            stage1_checkpoint=baseline,
            aggregate_expected=False,
            analysis_status=analysis_status,
        )
        if recovered is None:
            print(
                f"[campaign] seed {seed}: training and evaluation completed",
                flush=True,
            )
        manifests.append((seed, manifest_path, manifest))
        comparison_paths.append(protocol_dir / f"seed_{seed}/stage_comparison.json")

    complete = (not args.smoke) and seeds == tuple(range(5))
    if complete:
        if split_protocol == SPLIT_PROTOCOL:
            result = _write_formal_outputs(protocol_dir, manifests, comparison_paths)
        else:
            aggregate_protocol_results(protocol_dir, expected_seeds=tuple(range(5)))
            aggregate_manifest = protocol_dir / "aggregate_manifest.json"
            result = {
                "status": "completed",
                "protocol": protocol,
                "split_protocol": split_protocol,
                "seeds": list(seeds),
                "selection_partition": "validation",
                "evaluation_partition": "test",
                "aggregate_manifest": portable_artifact_path(aggregate_manifest),
                "aggregate_manifest_sha256": artifact_sha256(aggregate_manifest),
            }
            atomic_write_json(protocol_dir / "report_manifest.json", result)
    else:
        result = {
            "status": "smoke" if args.smoke else "partial",
            "seeds": list(seeds),
            "report": "deferred_until_seeds_0_to_4_complete",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
