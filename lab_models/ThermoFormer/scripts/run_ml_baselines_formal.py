"""Run the registered five-seed formal ML VLE baseline campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermoformer.baselines.machine_learning import (
    BASELINE_CAPABILITIES,
    FormalConfig,
    OVERALL_BENCHMARKS,
    UNIFIED_METRIC_KEYS,
    aggregate_formal_baseline,
    benchmark_for_baseline,
    run_formal_seed,
    write_formal_report,
)
from src.thermoformer.baselines.machine_learning.hanna import OfficialHANNAAssets
from src.thermoformer.baselines.machine_learning.spt_nrtl import SPT_NRTL_DATABASE_REVISION
from src.thermoformer.baselines.machine_learning.tennet_sac import (
    TENNETSAC_PACKAGE_VERSION,
    TENNETSAC_WHEEL_SHA256,
    installed_tennetsac_asset_audit,
    verified_tennetsac_wheel,
)
from src.thermoformer.reporting.artifacts import artifact_sha256
from scripts.run_external_activity_baselines_smoke import SMOKE_IMPLEMENTATION_FILES
from scripts.run_spt_nrtl_adapted_smoke import IMPLEMENTATION_FILES as ADAPTED_SPT_SMOKE_FILES


def _require_committed_input(path: Path) -> None:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _require_current_external_smoke(baseline: str, source_commit: str) -> None:
    path = (
        PROJECT_ROOT / 'experiments/vle/comparison/diagnostic_records/external_activity_baselines_smoke'
        / baseline / "smoke_result.json"
    )
    if not path.is_file():
        raise RuntimeError(f"Run the validation-only {baseline} smoke test before formal evaluation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "smoke_passed",
        "run_kind": "smoke",
        "analysis_status": "diagnostic",
        "evaluation_partition": "validation",
        "test_rows_used": False,
        "baseline": baseline,
        "registered_cells": 4,
        "valid_predictions": 4,
        "git_commit": source_commit,
        "requirements_sha256": artifact_sha256(PROJECT_ROOT / "requirements.txt"),
        "environment_sha256": artifact_sha256(PROJECT_ROOT / "environment.yml"),
        "implementation_sha256": {
            item: artifact_sha256(PROJECT_ROOT / item) for item in SMOKE_IMPLEMENTATION_FILES
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Stale or invalid {baseline} smoke evidence: {key}")
    assets = payload.get("model_audit", {}).get("model_assets", {})
    if baseline == "tennet_sac":
        if (
            assets.get("verified_wheel") != verified_tennetsac_wheel(
                PROJECT_ROOT / 'experiments/run_records/cache/tennetsac'
            )
            or assets.get("installed_assets") != installed_tennetsac_asset_audit()
            or assets.get("official_api_reference", {}).get("log_gamma")
            != [0.6033937335014343, 0.3612835109233856]
        ):
            raise RuntimeError("Stale TeNNet-SAC smoke asset evidence")
    elif (
        assets.get("database_revision") != SPT_NRTL_DATABASE_REVISION
        or assets.get("database_download_attempts") != 4
        or assets.get("database_license")
        != "not_declared_at_fixed_revision_publication_use_requires_review"
    ):
        raise RuntimeError("Stale SPT-NRTL smoke database evidence")


def _require_current_adapted_spt_smoke(source_commit: str) -> None:
    path = (
        PROJECT_ROOT / 'experiments/vle/comparison/diagnostic_records/external_activity_baselines_smoke'
        / "spt_nrtl_adapted" / "smoke_result.json"
    )
    if not path.is_file():
        raise RuntimeError("Run the validation-only spt_nrtl_adapted smoke test first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "smoke_passed",
        "run_kind": "smoke",
        "analysis_status": "diagnostic",
        "selection_partition": "validation",
        "evaluation_partition": "validation",
        "test_rows_used": False,
        "baseline": "spt_nrtl_adapted",
        "registered_cells": 4,
        "valid_predictions": 4,
        "git_commit": source_commit,
        "requirements_sha256": artifact_sha256(PROJECT_ROOT / "requirements.txt"),
        "environment_sha256": artifact_sha256(PROJECT_ROOT / "environment.yml"),
        "implementation_sha256": {
            item: artifact_sha256(PROJECT_ROOT / item) for item in ADAPTED_SPT_SMOKE_FILES
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Stale or invalid spt_nrtl_adapted smoke evidence: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("configs/vle/comparison/studies/machine_learning/formal_settings.json"),
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=tuple(BASELINE_CAPABILITIES),
        default=tuple(BASELINE_CAPABILITIES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=(0, 1, 2, 3, 4),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("experiments/vle/comparison/v1/machine_learning"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/vle/comparison/v1/machine_learning"),
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Formal seeds must be unique")
    settings_path = args.settings if args.settings.is_absolute() else PROJECT_ROOT / args.settings
    _require_committed_input(settings_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    formal = settings.get("formal", {})
    if settings.get("schema_version") != 1 or formal.get("enabled") is not True:
        raise ValueError("Formal settings must use schema 1 with formal.enabled=true")
    if settings.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("Formal settings must freeze seeds 0--4")
    if formal.get("hyperparameter_selection_partition") != "validation" or formal.get("evaluation_partition") != "test":
        raise ValueError("Formal settings must select on validation and evaluate on test")
    if tuple(formal.get("metrics", ())) != UNIFIED_METRIC_KEYS:
        raise ValueError("Formal metric registry does not match the executable metric contract")
    expected_benchmarks = [
        {
            "key": value.key,
            "split": value.split_protocol,
            "train_components": list(value.train_component_counts),
            "test_components": list(value.test_component_counts),
        }
        for value in OVERALL_BENCHMARKS
    ]
    if settings.get("benchmarks") != expected_benchmarks:
        raise ValueError("Formal benchmark registry does not match the executable protocol matrix")
    expected_external = {
        "tennet_sac": {
            "package": f"tennetsac=={TENNETSAC_PACKAGE_VERSION}",
            "wheel_sha256": TENNETSAC_WHEEL_SHA256,
            "model_version": "tuned_mean_of_10_experimental_finetuned_heads",
            "test_retraining_or_tuning": False,
        },
        "spt_nrtl": {
            "database_revision": SPT_NRTL_DATABASE_REVISION,
            "missing_pair_policy": "unavailable",
            "ternary_policy": "three_binary_pairs_standard_multicomponent_nrtl",
            "test_fitting_or_imputation": False,
        },
    }
    if settings.get("fixed_external_baselines") != expected_external:
        raise ValueError("Formal fixed-external baseline settings do not match the executable contract")
    adapted_spt = settings.get("adapted_spt_nrtl", {})
    expected_adapted_fixed = {
        "label_source": "binary_rows_from_registered_training_partition_only",
        "validation_role": "pair_label_checkpoint_selection_only",
        "test_fitting_or_tuning": False,
        "author_weights_or_cosmo_pretraining_used": False,
    }
    if {key: adapted_spt.get(key) for key in expected_adapted_fixed} != expected_adapted_fixed:
        raise ValueError("Adapted SPT-NRTL data-use contract does not match the executable protocol")
    config = FormalConfig(
        epochs=int(formal["epochs"]),
        learning_rate=float(formal["learning_rate"]),
        patience=int(formal["patience"]),
        mc_dropout_samples=int(formal["mc_dropout_samples"]),
        fixed_temperature_tolerance_k=float(formal["fixed_temperature_tolerance_k"]),
        minimum_composition=float(formal["minimum_composition"]),
        gdi_weight=float(formal["gdi_weight"]),
        device=str(formal["device"]),
        spt_adapted_max_sequence_length=int(adapted_spt["max_sequence_length"]),
        spt_adapted_embedding_dimension=int(adapted_spt["embedding_dimension"]),
        spt_adapted_attention_heads=int(adapted_spt["attention_heads"]),
        spt_adapted_transformer_layers=int(adapted_spt["transformer_layers"]),
        spt_adapted_feedforward_dimension=int(adapted_spt["feedforward_dimension"]),
        spt_adapted_dropout=float(adapted_spt["dropout"]),
        spt_adapted_batch_size=int(adapted_spt["batch_size"]),
        spt_adapted_epochs=int(adapted_spt["epochs"]),
        spt_adapted_learning_rate=float(adapted_spt["learning_rate"]),
        spt_adapted_patience=int(adapted_spt["patience"]),
        spt_adapted_minimum_pair_rows=int(adapted_spt["minimum_pair_rows"]),
        spt_adapted_label_fit_evaluations=int(adapted_spt["label_fit_evaluations"]),
        spt_adapted_label_regularization=float(adapted_spt["label_regularization"]),
    )
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal_settings.json requests CUDA, but torch.cuda.is_available() is false")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs/vle/comparison/studies/machine_learning", "datasets/vle", "splits", "models/baselines/hanna_official", "requirements.txt", "environment.yml"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Formal baseline runs require committed code, settings, dataset, and splits")
    for workbook in sorted((PROJECT_ROOT / "datasets" / "vle").glob("*.xlsx")):
        _require_committed_input(workbook)
    if "hanna" in args.baselines:
        for path in OfficialHANNAAssets.default(PROJECT_ROOT).required_files():
            _require_committed_input(path)
        _require_committed_input(PROJECT_ROOT / "requirements.txt")
    if "tennet_sac" in args.baselines:
        _require_committed_input(PROJECT_ROOT / "requirements.txt")
    settings_sha256 = artifact_sha256(settings_path)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    for baseline in args.baselines:
        if baseline in {"tennet_sac", "spt_nrtl"}:
            _require_current_external_smoke(baseline, source_commit)
        elif baseline == "spt_nrtl_adapted":
            _require_current_adapted_spt_smoke(source_commit)
    result_root = args.result_root if args.result_root.is_absolute() else PROJECT_ROOT / args.result_root
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else PROJECT_ROOT / args.checkpoint_root
    for baseline in args.baselines:
        benchmark = benchmark_for_baseline(baseline)
        for seed in args.seeds:
            split = PROJECT_ROOT / 'datasets/splits/vle' / benchmark.split_protocol / f"seed_{seed}.json"
            _require_committed_input(split)
            run_formal_seed(
                PROJECT_ROOT,
                baseline,
                seed,
                result_root / f"{baseline}.on.{benchmark.key}" / f"seed_{seed}",
                checkpoint_root / f"{baseline}.on.{benchmark.key}" / f"seed_{seed}",
                config,
                settings_sha256,
                source_commit,
            )
        if set(args.seeds) == set(range(5)):
            aggregate_formal_baseline(PROJECT_ROOT, baseline, result_root)
    if set(args.seeds) == set(range(5)) and set(args.baselines) == set(BASELINE_CAPABILITIES):
        write_formal_report(
            result_root,
            PROJECT_ROOT / 'experiments/vle/comparison/study_records/machine_learning/results.md',
        )
    print(json.dumps({"status": "completed_with_blocked_models", "baselines": args.baselines, "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()



