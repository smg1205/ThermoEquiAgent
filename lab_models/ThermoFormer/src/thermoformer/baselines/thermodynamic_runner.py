"""Reproducible state-generalization campaigns for classical VLE models."""

from __future__ import annotations

import csv
import io
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from ..configuration import experiment_sha256, load_experiment_config
from ..data import discover_vle_workbooks, load_vle_dataset, retain_pure_anchored_systems
from ..data.splitting import dataset_digest, load_split_assignment, system_id
from ..evaluation import prediction_metric_rows
from ..reporting.artifacts import artifact_sha256, atomic_write_json, atomic_write_text
from .thermodynamic_evaluation import (
    predict_fitted_activity_model,
    unavailable_prediction_records,
)
from .thermodynamic_fitting import (
    fit_shared_activity_model,
    fit_vapor_pressure_correlations,
)


STATE_PROTOCOLS = (
    "state_composition_interpolation",
    "state_composition_edge_extrapolation",
    "state_temperature_low_extrapolation",
    "state_temperature_high_extrapolation",
    "state_pressure_low_extrapolation",
    "state_pressure_high_extrapolation",
)
ACTIVITY_MODELS = ("nrtl", "wilson", "uniquac")


@dataclass(frozen=True)
class BaselineCampaignSettings:
    models: tuple[str, ...]
    protocol_seeds: dict[str, tuple[int, ...]]
    pure_vapor_pressure: str
    activity_parameter_temperature_form: str
    nrtl_alpha: float
    maximum_optimizer_iterations: int
    formal_output_root: str
    smoke_output_root: str
    thermoformer_output_template: str
    thermoformer_seeds: tuple[int, ...]
    training_partition: str
    evaluation_partition: str

    @classmethod
    def load(cls, path: Path) -> "BaselineCampaignSettings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "models",
            "protocol_seeds",
            "pure_vapor_pressure",
            "activity_parameter_temperature_form",
            "nrtl_alpha",
            "maximum_optimizer_iterations",
            "formal_output_root",
            "smoke_output_root",
            "thermoformer_output_template",
            "thermoformer_seeds",
            "training_partition",
            "evaluation_partition",
        }
        if set(payload) != required:
            raise ValueError("Baseline settings must contain exactly the documented fields")
        settings = cls(
            models=tuple(payload["models"]),
            protocol_seeds={
                str(protocol): tuple(int(seed) for seed in seeds)
                for protocol, seeds in payload["protocol_seeds"].items()
            },
            pure_vapor_pressure=str(payload["pure_vapor_pressure"]),
            activity_parameter_temperature_form=str(
                payload["activity_parameter_temperature_form"]
            ),
            nrtl_alpha=float(payload["nrtl_alpha"]),
            maximum_optimizer_iterations=int(payload["maximum_optimizer_iterations"]),
            formal_output_root=str(payload["formal_output_root"]),
            smoke_output_root=str(payload["smoke_output_root"]),
            thermoformer_output_template=str(payload["thermoformer_output_template"]),
            thermoformer_seeds=tuple(int(seed) for seed in payload["thermoformer_seeds"]),
            training_partition=str(payload["training_partition"]),
            evaluation_partition=str(payload["evaluation_partition"]),
        )
        if settings.models != ACTIVITY_MODELS:
            raise ValueError("Baseline settings must retain NRTL, Wilson, and UNIQUAC")
        if tuple(settings.protocol_seeds) != STATE_PROTOCOLS:
            raise ValueError("Baseline settings must retain all registered state protocols")
        if settings.protocol_seeds[STATE_PROTOCOLS[0]] != (0, 1, 2, 3, 4):
            raise ValueError("Composition interpolation must use its five distinct splits")
        if any(
            seeds != (0,)
            for protocol, seeds in settings.protocol_seeds.items()
            if protocol != STATE_PROTOCOLS[0]
        ):
            raise ValueError("Deterministic fixed state splits must be evaluated once")
        if not 0.0 < settings.nrtl_alpha <= 1.0:
            raise ValueError("nrtl_alpha must be in (0, 1]")
        if settings.maximum_optimizer_iterations <= 0:
            raise ValueError("maximum_optimizer_iterations must be positive")
        if settings.training_partition != "train" or settings.evaluation_partition != "test":
            raise ValueError("Baseline fitting must use train and evaluation must use test")
        if settings.pure_vapor_pressure != "ln(Psat/kPa) = a + b/T":
            raise ValueError("Unsupported pure-vapor-pressure correlation")
        if settings.activity_parameter_temperature_form != "B_ij/T":
            raise ValueError("Unsupported activity-parameter temperature form")
        if settings.thermoformer_seeds != (0, 1, 2, 3, 4):
            raise ValueError("ThermoFormer comparison must use formal seeds 0--4")
        if "{protocol}" not in settings.thermoformer_output_template:
            raise ValueError("thermoformer_output_template must contain {protocol}")
        return settings


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _config_sources(path: Path, ancestors: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    resolved = path.resolve()
    if resolved in ancestors:
        raise ValueError("Cyclic baseline experiment configuration")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    parent = payload.get("extends")
    if parent is None:
        return (resolved,)
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return (*_config_sources(parent_path, (*ancestors, resolved)), resolved)


def _require_committed_file(project_root: Path, path: Path, label: str) -> None:
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    tracked = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ],
        cwd=project_root,
        capture_output=True,
    )
    unchanged = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
        cwd=project_root,
        capture_output=True,
    )
    if tracked.returncode != 0 or unchanged.returncode != 0:
        raise RuntimeError(f"Formal {label} must be committed and unchanged: {path}")


def _require_clean_scientific_code(project_root: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "status",
            "--porcelain",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored_prefixes = ("results/", "runs/", "figures/", "reports/")
    dirty = []
    for line in completed.stdout.splitlines():
        value = line[3:].strip().replace("\\", "/")
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        if not value.startswith(ignored_prefixes):
            dirty.append(value)
    if dirty:
        raise RuntimeError(
            "Formal baseline run requires committed scientific code and inputs: "
            + ", ".join(dirty[:10])
        )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify_path_record(
    project_root: Path, record: dict[str, object], label: str
) -> Path:
    path = project_root / str(record["path"])
    expected = str(record["sha256"])
    if not path.is_file() or artifact_sha256(path) != expected:
        raise ValueError(f"Current {label} does not match its recorded SHA: {path}")
    return path


def _temperature_bounds(train_samples: Sequence[Any]) -> tuple[float, float]:
    temperatures = [float(row.temperature_k) for row in train_samples]
    if not temperatures:
        return 150.0, 1500.0
    lower = max(150.0, min(temperatures) - 200.0)
    upper = min(1500.0, max(temperatures) + 300.0)
    return (lower, upper) if lower < upper else (150.0, 1500.0)


def run_state_protocol_baseline(
    *,
    project_root: Path,
    protocol: str,
    seed: int,
    models: Sequence[str] = ACTIVITY_MODELS,
    output_root: Path | None = None,
    overwrite: bool = False,
    maximum_systems: int | None = None,
    run_kind: str = "formal",
) -> list[Path]:
    """Fit on one registered training partition and evaluate its held-out test states."""

    project_root = project_root.resolve()
    settings_path = (
        project_root
        / 'configs/vle/comparison/studies/thermodynamic_models/settings.json'
    )
    settings = BaselineCampaignSettings.load(settings_path)
    if protocol not in settings.protocol_seeds:
        raise ValueError(f"Unsupported state protocol: {protocol}")
    if seed not in settings.protocol_seeds[protocol]:
        raise ValueError(f"Seed {seed} is not registered for deterministic protocol {protocol}")
    if any(model not in settings.models for model in models):
        raise ValueError("models must be selected from nrtl, wilson, and uniquac")
    if run_kind not in {"formal", "smoke"}:
        raise ValueError("run_kind must be formal or smoke")
    output_root = (
        output_root.resolve()
        if output_root is not None
        else project_root
        / (settings.formal_output_root if run_kind == "formal" else settings.smoke_output_root)
    )
    config_path = project_root / 'configs/vle/comparison/studies/thermodynamic_models/config.json'
    split_path = project_root / 'datasets/splits/vle' / protocol / f"seed_{seed}.json"
    experiment = load_experiment_config(config_path)
    data_root = project_root / experiment.data.root
    config_sources = _config_sources(config_path)
    workbooks = discover_vle_workbooks(data_root, experiment.data.source_filter)
    if run_kind == "formal":
        _require_clean_scientific_code(project_root)
        for source in (*config_sources, settings_path, split_path, *workbooks):
            _require_committed_file(project_root, source, "scientific input")
    loaded = load_vle_dataset(
        data_root,
        source_filter=experiment.data.source_filter,
        failed_weight=experiment.data.failed_weight,
        max_pressure_kpa=experiment.data.max_pressure_kpa,
    )
    samples = retain_pure_anchored_systems(
        loaded.samples,
        minimum_temperatures=experiment.data.minimum_pure_anchor_temperatures,
    )
    split = load_split_assignment(split_path, samples)
    if split.protocol != protocol or split.seed != seed:
        raise ValueError("Loaded split identity does not match the requested protocol and seed")

    vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(split.train)
    test_by_system: dict[str, list[Any]] = defaultdict(list)
    for row in split.test:
        test_by_system[system_id(row)].append(row)
    selected_systems = sorted(test_by_system)
    if maximum_systems is not None:
        selected_systems = selected_systems[:maximum_systems]
    selected_test = [row for key in selected_systems for row in test_by_system[key]]
    interaction_training_rows = (
        list(split.train)
        if maximum_systems is None
        else [row for row in split.train if system_id(row) in set(selected_systems)]
    )
    bounds = _temperature_bounds(split.train)
    seed_root = output_root / protocol / f"seed_{seed}"
    vapor_path = seed_root / "vapor_pressure_fits.json"
    if vapor_path.exists() and not overwrite:
        raise FileExistsError(f"Baseline output already exists: {vapor_path}")
    atomic_write_json(
        vapor_path,
        {
            "protocol": protocol,
            "seed": seed,
            "training_partition": settings.training_partition,
            "correlation": settings.pure_vapor_pressure,
            "audit": vapor_audit,
            "parameters": {
                key: {
                    "intercept": value.intercept,
                    "inverse_temperature_k": value.inverse_temperature,
                }
                for key, value in sorted(vapor_pressure.items())
            },
        },
    )

    manifests: list[Path] = []
    for model in models:
        started = time.time()
        records: list[dict[str, object]] = []
        successful_systems = 0
        attempted_by_cardinality: dict[int, int] = defaultdict(int)
        fitted_by_cardinality: dict[int, int] = defaultdict(int)
        failure_counts: dict[str, int] = defaultdict(int)
        shared = fit_shared_activity_model(
            model,
            interaction_training_rows,
            vapor_pressure,
            maximum_iterations=settings.maximum_optimizer_iterations,
            nrtl_alpha=settings.nrtl_alpha,
        )
        for index, key in enumerate(selected_systems, start=1):
            test_rows = test_by_system[key]
            cardinality = test_rows[0].component_count
            attempted_by_cardinality[cardinality] += 1
            if not shared.success:
                reason = "global_activity_parameter_fit_failed"
                records.extend(unavailable_prediction_records(test_rows, model, reason))
                failure_counts[reason] += 1
                continue
            components = tuple(sorted(test_rows[0].smiles))
            fitted = shared.for_components(components)
            if fitted is None:
                reason = "shared_pair_or_uniquac_size_parameters_unavailable"
                records.extend(unavailable_prediction_records(test_rows, model, reason))
                failure_counts[reason] += 1
                continue
            try:
                records.extend(
                    predict_fitted_activity_model(
                        test_rows,
                        fitted,
                        vapor_pressure,
                        temperature_bounds_k=bounds,
                    )
                )
                successful_systems += 1
                fitted_by_cardinality[cardinality] += 1
            except (ArithmeticError, KeyError, RuntimeError, ValueError) as error:
                reason = str(error).splitlines()[0][:160]
                records.extend(unavailable_prediction_records(test_rows, model, reason))
                failure_counts[reason] += 1
            if index % 25 == 0 or index == len(selected_systems):
                print(f"[{protocol} seed={seed} {model}] {index}/{len(selected_systems)} systems")

        model_root = seed_root / model
        predictions_path = model_root / "predictions.csv"
        metrics_path = model_root / "metrics.json"
        parameters_path = model_root / "parameters.json"
        manifest_path = model_root / "manifest.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Baseline output already exists: {manifest_path}")
        metric_rows = prediction_metric_rows(records)
        _atomic_csv(predictions_path, records)
        atomic_write_json(metrics_path, metric_rows)
        atomic_write_json(
            parameters_path,
            {
                "model": model,
                "parameterization": (
                    f"tau_ij=B_ij/T, alpha_ij={settings.nrtl_alpha}"
                    if model == "nrtl"
                    else ("ln(Lambda_ij)=B_ij/T" if model == "wilson" else "ln(tau_ij)=B_ij/T")
                ),
                "training_partition": settings.training_partition,
                "fit_objective": "quality-weighted direct log-pressure and vapor-composition residuals",
                "shared_ordered_pair_parameters_k": [
                    {"component_i": pair[0], "component_j": pair[1], "B_ij_k": value}
                    for pair, value in sorted(shared.pair_parameters_k.items())
                ],
                "training_rows_with_model_coverage": shared.training_samples,
                "fitted_ordered_pairs": shared.fitted_pairs,
                "initial_training_loss": shared.initial_training_loss,
                "final_training_loss": shared.final_training_loss,
                "optimizer_message": shared.optimizer_message,
                "requested_systems": len(selected_systems),
                "fitted_systems": successful_systems,
                "system_parameter_coverage": successful_systems / len(selected_systems),
                "parameter_coverage_by_component_count": {
                    str(cardinality): {
                        "attempted_systems": attempted,
                        "fitted_systems": fitted_by_cardinality.get(cardinality, 0),
                        "coverage": fitted_by_cardinality.get(cardinality, 0) / attempted,
                    }
                    for cardinality, attempted in sorted(attempted_by_cardinality.items())
                },
                "failure_counts": dict(sorted(failure_counts.items())),
            },
        )
        manifest = {
            "schema_version": 1,
            "status": "completed" if run_kind == "formal" else "diagnostic",
            "run_kind": run_kind,
            "experiment": "thermodynamic_model_state_generalization",
            "model": model,
            "protocol": protocol,
            "seed": seed,
            "selection_partition": settings.training_partition,
            "evaluation_partition": settings.evaluation_partition,
            "test_labels_used_for_fitting": False,
            "test_labels_used_for_case_selection": False,
            "dataset_sha256": dataset_digest(samples),
            "split_sha256": artifact_sha256(split_path),
            "split": {
                "path": split_path.relative_to(project_root).as_posix(),
                "sha256": artifact_sha256(split_path),
            },
            "resolved_config_sha256": experiment_sha256(experiment),
            "config_sources": [
                {
                    "path": source.relative_to(project_root).as_posix(),
                    "sha256": artifact_sha256(source),
                }
                for source in config_sources
            ],
            "settings_sha256": artifact_sha256(settings_path),
            "dataset_workbooks": [
                {
                    "path": workbook.relative_to(project_root).as_posix(),
                    "sha256": artifact_sha256(workbook),
                }
                for workbook in workbooks
            ],
            "vapor_pressure_fits_sha256": artifact_sha256(vapor_path),
            "git_commit": _git_commit(project_root),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": _version("scipy"),
                "thermo": _version("thermo"),
            },
            "counts": {
                "training_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(selected_test),
                "requested_systems": len(selected_systems),
                "fitted_systems": successful_systems,
                "system_parameter_coverage": successful_systems / len(selected_systems),
                "parameter_coverage_by_component_count": {
                    str(cardinality): {
                        "attempted_systems": attempted,
                        "fitted_systems": fitted_by_cardinality.get(cardinality, 0),
                        "coverage": fitted_by_cardinality.get(cardinality, 0) / attempted,
                    }
                    for cardinality, attempted in sorted(attempted_by_cardinality.items())
                },
            },
            "temperature_solver_bounds_k": list(bounds),
            "elapsed_seconds": time.time() - started,
            "artifacts": {
                "vapor_pressure_fits": {"path": str(vapor_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(vapor_path)},
                "predictions": {"path": str(predictions_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(predictions_path)},
                "metrics": {"path": str(metrics_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(metrics_path)},
                "parameters": {"path": str(parameters_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(parameters_path)},
            },
        }
        atomic_write_json(manifest_path, manifest)
        manifests.append(manifest_path)
    return manifests


def _direction_row(metrics: Sequence[dict[str, object]], direction: str) -> dict[str, object]:
    matches = [row for row in metrics if row.get("scope") == "direction" and row.get("direction") == direction]
    if len(matches) != 1:
        raise ValueError(f"Expected one direction row for {direction}")
    return matches[0]


def aggregate_state_baselines(
    *,
    project_root: Path,
    output_root: Path | None = None,
    report_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Aggregate seed outputs and generate the human-readable priority-1 report."""

    project_root = project_root.resolve()
    _require_clean_scientific_code(project_root)
    aggregation_git_commit = _git_commit(project_root)
    settings_path = project_root / "configs/vle/comparison/studies/thermodynamic_models/settings.json"
    settings = BaselineCampaignSettings.load(settings_path)
    current_settings_sha256 = artifact_sha256(settings_path)
    config_path = project_root / "configs/vle/comparison/studies/thermodynamic_models/config.json"
    current_config_sha256 = experiment_sha256(load_experiment_config(config_path))
    protocols = tuple(settings.protocol_seeds)
    models = settings.models
    output_root = output_root or project_root / settings.formal_output_root
    report_path = report_path or project_root / 'experiments/vle/comparison/study_records/thermodynamic_models/results.md'
    metric_names = (
        "valid_coverage",
        "solver_failure_rate",
        "nonphysical_rate",
        "pressure_mae_kpa",
        "pressure_rmse_kpa",
        "pressure_r2",
        "temperature_mae_k",
        "temperature_rmse_k",
        "temperature_r2",
        "y_mae",
        "y_rmse",
        "y_r2",
        "pressure_system_macro_mae_kpa",
        "pressure_system_macro_rmse_kpa",
        "temperature_system_macro_mae_k",
        "temperature_system_macro_rmse_k",
        "y_system_macro_mae",
        "y_system_macro_rmse",
    )
    summary_rows: list[dict[str, object]] = []
    input_manifests: list[dict[str, object]] = []
    common_provenance: dict[str, object] | None = None
    for protocol in protocols:
        for model in models:
            group_values: dict[tuple[str, int | None], dict[str, list[float]]] = {
                (direction, cardinality): defaultdict(list)
                for direction in ("isothermal", "isobaric")
                for cardinality in (None, 2, 3)
            }
            parameter_coverages: dict[int | None, list[float]] = defaultdict(list)
            seeds = settings.protocol_seeds[protocol]
            for seed in seeds:
                model_root = output_root / protocol / f"seed_{seed}" / model
                manifest_path = model_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("status") != "completed"
                    or manifest.get("run_kind") != "formal"
                    or manifest.get("protocol") != protocol
                    or manifest.get("model") != model
                    or int(manifest.get("seed", -1)) != seed
                    or manifest.get("selection_partition") != settings.training_partition
                    or manifest.get("evaluation_partition") != settings.evaluation_partition
                ):
                    raise ValueError(f"Incomplete baseline manifest: {manifest_path}")
                provenance = {
                    key: manifest[key]
                    for key in (
                        "dataset_sha256",
                        "resolved_config_sha256",
                        "config_sources",
                        "settings_sha256",
                        "dataset_workbooks",
                        "git_commit",
                        "runtime",
                    )
                }
                if manifest["git_commit"] != aggregation_git_commit:
                    raise ValueError(
                        f"Baseline run was produced by a different commit: {manifest_path}"
                    )
                if manifest["settings_sha256"] != current_settings_sha256:
                    raise ValueError(f"Settings changed after baseline run: {manifest_path}")
                if manifest["resolved_config_sha256"] != current_config_sha256:
                    raise ValueError(f"Resolved config changed after baseline run: {manifest_path}")
                for record in manifest["config_sources"]:
                    _verify_path_record(project_root, record, "config source")
                for record in manifest["dataset_workbooks"]:
                    _verify_path_record(project_root, record, "dataset workbook")
                _verify_path_record(project_root, manifest["split"], "split assignment")
                if common_provenance is None:
                    common_provenance = provenance
                elif provenance != common_provenance:
                    raise ValueError(f"Cross-run provenance mismatch: {manifest_path}")
                for artifact in manifest["artifacts"].values():
                    artifact_path = project_root / artifact["path"]
                    if artifact_sha256(artifact_path) != artifact["sha256"]:
                        raise ValueError(f"Artifact SHA mismatch: {artifact_path}")
                input_manifests.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "seed": seed,
                        "path": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
                        "sha256": artifact_sha256(manifest_path),
                    }
                )
                parameter_coverages[None].append(
                    float(manifest["counts"]["system_parameter_coverage"])
                )
                for cardinality, coverage in manifest["counts"][
                    "parameter_coverage_by_component_count"
                ].items():
                    parameter_coverages[int(cardinality)].append(float(coverage["coverage"]))
                metrics_path = model_root / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for (direction, cardinality), values in group_values.items():
                    scope = "direction" if cardinality is None else "direction_cardinality"
                    matches = [
                        row
                        for row in metrics
                        if row.get("scope") == scope
                        and row.get("direction") == direction
                        and row.get("component_count") == cardinality
                    ]
                    if len(matches) != 1:
                        continue
                    row = matches[0]
                    for metric in metric_names:
                        value = row.get(metric)
                        if value is not None and math.isfinite(float(value)):
                            values[metric].append(float(value))
            for (direction, cardinality), values in group_values.items():
                if not values:
                    continue
                output: dict[str, object] = {
                    "protocol": protocol,
                    "model": model,
                    "direction": direction,
                    "component_count": cardinality,
                    "seeds": len(seeds),
                    "system_parameter_coverage_mean": mean(parameter_coverages[cardinality]),
                    "system_parameter_coverage_std": (
                        stdev(parameter_coverages[cardinality])
                        if len(parameter_coverages[cardinality]) > 1
                        else 0.0
                    ),
                }
                for metric in metric_names:
                    available = values.get(metric, [])
                    output[f"{metric}_mean"] = mean(available) if available else None
                    output[f"{metric}_std"] = stdev(available) if len(available) > 1 else (0.0 if available else None)
                summary_rows.append(output)

        thermoformer_root = project_root / settings.thermoformer_output_template.format(
            protocol=protocol
        )
        thermoformer_path = thermoformer_root / "metrics_summary.csv"
        thermoformer_manifest_path = thermoformer_root / "aggregate_manifest.json"
        _require_committed_file(
            project_root, thermoformer_manifest_path, "ThermoFormer aggregate manifest"
        )
        _require_committed_file(
            project_root, thermoformer_path, "ThermoFormer metric summary"
        )
        thermoformer_manifest = json.loads(
            thermoformer_manifest_path.read_text(encoding="utf-8")
        )
        expected_protocol = f"c1_three_view_vanilla_fugacity_finetune.on.{protocol}"
        if (
            thermoformer_manifest.get("status") != "completed"
            or thermoformer_manifest.get("aggregate_kind") != "formal"
            or thermoformer_manifest.get("protocol") != expected_protocol
            or tuple(thermoformer_manifest.get("seeds", ())) != settings.thermoformer_seeds
        ):
            raise ValueError(
                f"Invalid formal ThermoFormer aggregate: {thermoformer_manifest_path}"
            )
        if thermoformer_manifest.get("input_provenance", {}).get(
            "dataset_sha256"
        ) != common_provenance["dataset_sha256"]:
            raise ValueError(
                f"ThermoFormer dataset differs from baseline data: {thermoformer_manifest_path}"
            )
        input_manifest_sha256 = thermoformer_manifest.get("input_manifest_sha256", {})
        for seed in settings.thermoformer_seeds:
            current_split_sha256 = artifact_sha256(
                project_root / 'datasets/splits/vle' / protocol / f"seed_{seed}.json"
            )
            seed_manifest_path = thermoformer_root / f"seed_{seed}" / "manifest.json"
            _require_committed_file(
                project_root, seed_manifest_path, "ThermoFormer seed manifest"
            )
            if input_manifest_sha256.get(str(seed)) != artifact_sha256(seed_manifest_path):
                raise ValueError(
                    f"ThermoFormer seed manifest is not bound by its aggregate: {seed_manifest_path}"
                )
            seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
            if (
                seed_manifest.get("status") != "completed"
                or seed_manifest.get("run_kind") != "formal"
                or seed_manifest.get("analysis_status") != "confirmatory"
                or seed_manifest.get("evaluation_partition") != "test"
                or seed_manifest.get("split_protocol") != protocol
                or int(seed_manifest.get("seed", -1)) != seed
                or seed_manifest.get("dataset_sha256")
                != common_provenance["dataset_sha256"]
                or seed_manifest.get("split_sha256") != current_split_sha256
            ):
                raise ValueError(
                    f"ThermoFormer seed provenance does not match this protocol: {seed_manifest_path}"
                )
        recorded_summary = thermoformer_manifest.get("outputs", {}).get("metrics_summary", {})
        if (
            recorded_summary.get("path") != thermoformer_path.relative_to(project_root).as_posix()
            or recorded_summary.get("sha256") != artifact_sha256(thermoformer_path)
        ):
            raise ValueError(
                f"ThermoFormer summary does not match its aggregate manifest: {thermoformer_path}"
            )
        with thermoformer_path.open("r", encoding="utf-8", newline="") as handle:
            thermoformer_rows = list(csv.DictReader(handle))
        for kind, path in (
            ("aggregate_manifest", thermoformer_manifest_path),
            ("metrics_summary", thermoformer_path),
        ):
            input_manifests.append(
                {
                    "protocol": protocol,
                    "model": "thermoformer_c1_fugacity",
                    "kind": kind,
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": artifact_sha256(path),
                }
            )
        for direction in ("isothermal", "isobaric"):
            for cardinality in (None, 2, 3):
                scope = "direction" if cardinality is None else "direction_cardinality"
                matches = [
                    row
                    for row in thermoformer_rows
                    if row.get("scope") == scope
                    and row.get("direction") == direction
                    and (
                        (cardinality is None and not row.get("component_count"))
                        or row.get("component_count") == str(cardinality)
                    )
                ]
                if len(matches) != 1:
                    continue
                source = matches[0]
                output = {
                    "protocol": protocol,
                    "model": "thermoformer_c1_fugacity",
                    "direction": direction,
                    "component_count": cardinality,
                    "seeds": len(settings.thermoformer_seeds),
                    "system_parameter_coverage_mean": 1.0,
                    "system_parameter_coverage_std": 0.0,
                }
                for metric in metric_names:
                    for suffix in ("mean", "std"):
                        raw = source.get(f"{metric}_{suffix}")
                        output[f"{metric}_{suffix}"] = (
                            float(raw) if raw not in (None, "") else None
                        )
                summary_rows.append(output)
    summary_path = output_root / "metrics_summary.csv"
    _atomic_csv(summary_path, summary_rows)

    def formatted(row: dict[str, object], name: str, digits: int) -> str:
        center, spread = row.get(f"{name}_mean"), row.get(f"{name}_std")
        if center is None:
            return "N/A"
        if int(row["seeds"]) == 1:
            return f"{float(center):.{digits}f}"
        return f"{float(center):.{digits}f} ± {float(spread):.{digits}f}"

    protocol_labels = {
        "state_composition_interpolation": "Composition interpolation",
        "state_composition_edge_extrapolation": "Composition-edge extrapolation",
        "state_temperature_low_extrapolation": "Low-temperature extrapolation",
        "state_temperature_high_extrapolation": "High-temperature extrapolation",
        "state_pressure_low_extrapolation": "Low-pressure extrapolation",
        "state_pressure_high_extrapolation": "High-pressure extrapolation",
    }
    model_labels = {
        "nrtl": "NRTL",
        "wilson": "Wilson",
        "uniquac": "UNIQUAC",
        "thermoformer_c1_fugacity": "ThermoFormer C1 + fugacity",
    }
    report_models = (*models, "thermoformer_c1_fugacity")
    lines = [
        "# NRTL, Wilson, and UNIQUAC state-generalization results",
        "",
        "Status: completed.",
        "",
        "The classical models use shared ordered molecular-pair parameters fitted directly to pressure and vapor-composition residuals in the registered training partition. Validation and test labels are not used for fitting. Composition interpolation uses its five distinct data splits. The other state protocols have one deterministic partition and are fitted once; ThermoFormer values retain five neural-network seeds.",
    ]
    for cardinality, scope_label in ((None, "All systems"), (2, "Binary systems"), (3, "Ternary systems")):
        for direction in ("isothermal", "isobaric"):
            state_label = "P-x-y" if direction == "isothermal" else "T-x-y"
            lines.extend(["", f"## {scope_label}: {state_label}", ""])
            if direction == "isothermal":
                lines.extend([
                    "| Protocol | Model | P MAE (kPa) | P RMSE (kPa) | P R² | y MAE | y RMSE | y R² | Valid coverage | Parameter coverage |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ])
                state_names = ("pressure_mae_kpa", "pressure_rmse_kpa", "pressure_r2")
            else:
                lines.extend([
                    "| Protocol | Model | T MAE (K) | T RMSE (K) | T R² | y MAE | y RMSE | y R² | Valid coverage | Parameter coverage |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ])
                state_names = ("temperature_mae_k", "temperature_rmse_k", "temperature_r2")
            for protocol in protocols:
                for model in report_models:
                    matches = [
                        value
                        for value in summary_rows
                        if value["protocol"] == protocol
                        and value["model"] == model
                        and value["direction"] == direction
                        and value["component_count"] == cardinality
                    ]
                    if not matches:
                        continue
                    row = matches[0]
                    values = [
                        formatted(row, state_names[0], 3),
                        formatted(row, state_names[1], 3),
                        formatted(row, state_names[2], 4),
                        formatted(row, "y_mae", 4),
                        formatted(row, "y_rmse", 4),
                        formatted(row, "y_r2", 4),
                        formatted(row, "valid_coverage", 3),
                        f"{float(row['system_parameter_coverage_mean']):.3f}",
                    ]
                    lines.append(
                        f"| {protocol_labels[protocol]} | {model_labels[model]} | "
                        + " | ".join(values)
                        + " |"
                    )
    lines.extend(["", "## System-macro errors", ""])
    lines.extend([
        "| Scope | Protocol | Model | Direction | State macro MAE | State macro RMSE | y macro MAE | y macro RMSE |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for cardinality, scope_label in ((None, "All"), (2, "Binary"), (3, "Ternary")):
        for protocol in protocols:
            for model in report_models:
                for direction in ("isothermal", "isobaric"):
                    matches = [
                        value
                        for value in summary_rows
                        if value["protocol"] == protocol
                        and value["model"] == model
                        and value["direction"] == direction
                        and value["component_count"] == cardinality
                    ]
                    if not matches:
                        continue
                    row = matches[0]
                    prefix = "pressure" if direction == "isothermal" else "temperature"
                    unit = "kpa" if prefix == "pressure" else "k"
                    lines.append(
                        f"| {scope_label} | {protocol_labels[protocol]} | {model_labels[model]} | {direction} | "
                        f"{formatted(row, f'{prefix}_system_macro_mae_{unit}', 3)} | "
                        f"{formatted(row, f'{prefix}_system_macro_rmse_{unit}', 3)} | "
                        f"{formatted(row, 'y_system_macro_mae', 4)} | {formatted(row, 'y_system_macro_rmse', 4)} |"
                    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are within-system state-generalization baselines. They do not establish predictive performance for an unseen chemical system. Each ordered molecular-pair parameter is shared wherever that pair occurs, including binary and ternary training rows. UNIQUAC coverage is lower when standard UNIFAC-derived molecular size parameters are unavailable; missing systems remain failures rather than ideal substitutions.",
            "",
            "## Reproducibility",
            "",
            f"Machine-readable summary: `{summary_path.relative_to(project_root).as_posix()}`.",
            "",
        ]
    )
    atomic_write_text(report_path, "\n".join(lines))
    manifest_path = output_root / "report_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "experiment": "thermodynamic_model_state_generalization",
            "protocols": list(protocols),
            "models": list(models),
            "protocol_seeds": {key: list(value) for key, value in settings.protocol_seeds.items()},
            "aggregation_git_commit": aggregation_git_commit,
            "provenance": common_provenance,
            "inputs": input_manifests,
            "outputs": {
                "summary": {"path": str(summary_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(summary_path)},
                "report": {"path": str(report_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(report_path)},
            },
        },
    )
    return summary_path, report_path, manifest_path
