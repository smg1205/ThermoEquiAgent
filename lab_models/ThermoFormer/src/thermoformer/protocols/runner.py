"""Single-split, single-seed runner with complete paper artifact provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rdkit
import torch
import yaml

from ..configuration import ExperimentConfig, experiment_sha256, load_experiment_config
from ..reporting.artifacts import artifact_sha256, portable_artifact_path
from ..data.auditing import ternary_subsystem_rows
from ..data import discover_vle_workbooks, load_vle_dataset, retain_pure_anchored_systems
from ..evaluation import predict_vle, prediction_metric_rows, write_prediction_csv
from ..evaluation.thermodynamic_consistency import evaluate_thermodynamic_consistency
from ..models import ThermoFormer
from ..thermodynamics.vapor_pressure import empty_pure_property_catalog, load_pure_property_catalog
from ..features import build_molecular_encoder, feature_subset_sha256, prepare_partition_features
from ..data.splitting import dataset_digest, load_split_assignment, validate_protocol_name
from ..training import fit_model, seed_everything
from ..training.fugacity_finetuning import (
    evaluate_physics_residuals,
    fit_physics_stage,
    load_stage1_checkpoint,
)
from ..training.checkpointing import cpu_state_dict
from ..training.direct_ge_pipeline import (
    evaluate_direct_label_metrics,
    fit_direct_ge_stages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def result_protocol_name(experiment_name: str, split_protocol: str) -> str:
    """Namespace a variant on a reused split without overwriting its reference."""
    experiment = validate_protocol_name(experiment_name)
    split = validate_protocol_name(split_protocol)
    return split if experiment == split else validate_protocol_name(f"{experiment}.on.{split}")


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return artifact_sha256(path)


def _normalized_experiment_digest(experiment: ExperimentConfig) -> str:
    return experiment_sha256(experiment)


def _feature_subset_digest(feature_map: dict[str, np.ndarray]) -> str:
    return feature_subset_sha256(feature_map)



_STAGE0_RECIPE_SCHEMA = "thermoformer-stage0-recipe-v1"
_STAGE0_RUNTIME_MODEL_FIELDS = frozenset(
    {
        "feature_dim",
        "rdkit_feature_dim",
        "unimol_feature_dim",
        "functional_group_feature_dim",
    }
)


def _stage0_recipe_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _stage0_semantic_model_recipe(
    model: Mapping[str, object], encoder: Mapping[str, object]
) -> dict[str, object]:
    normalized = {
        str(name): value
        for name, value in model.items()
        if name not in _STAGE0_RUNTIME_MODEL_FIELDS
    }
    normalized["fusion_mode"] = encoder.get("fusion_mode")
    normalized["chemical_attention_bias"] = encoder.get(
        "chemical_attention_bias"
    )
    normalized["context_pair_interaction"] = encoder.get(
        "context_pair_interaction"
    )
    return normalized


def _formal_c1_stage0_recipe(
    experiment: ExperimentConfig,
    split_path: Path,
    split_protocol: str,
    seed: int,
    dataset_sha256: str,
    feature_cache_sha256: str,
    pure_property_catalog_sha256: str | None,
) -> dict[str, object] | None:
    """Build the versioned recipe required for auditable C1 Stage 0 reuse."""

    if (
        experiment.name != "c1_three_view_vanilla"
        or experiment.direct_ge_supervision is not None
        or experiment.physics_finetuning is not None
        or experiment.protocol is None
        or experiment.protocol.evaluation_partition != "validation"
    ):
        return None
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(split_payload, Mapping):
        return None
    partitions = split_payload.get("partitions")
    if not isinstance(partitions, Mapping):
        return None
    partition_recipe: dict[str, dict[str, object]] = {}
    for partition in ("train", "validation", "test"):
        values = partitions.get(partition)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            return None
        partition_recipe[partition] = {
            "count": len(values),
            "sha256": _stage0_recipe_digest(values),
        }
    payload = experiment.to_dict()
    data = payload.get("data")
    encoder = payload.get("encoder")
    model = payload.get("model")
    training = payload.get("training")
    protocol = payload.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (data, encoder, model, training, protocol)
    ):
        return None
    return {
        "schema": _STAGE0_RECIPE_SCHEMA,
        "variant": experiment.name,
        "protocol": {
            "split_protocol": split_protocol,
            "seed": seed,
            "registered_splits": list(protocol.get("registered_splits", ())),
            "split_sha256": _file_digest(split_path),
            "dataset_sha256": dataset_sha256,
            "partitions": partition_recipe,
            "metadata_sha256": _stage0_recipe_digest(
                split_payload.get("metadata")
            ),
        },
        "data": {
            "config": dict(data),
            "dataset_sha256": dataset_sha256,
            "pure_property_catalog_sha256": pure_property_catalog_sha256,
        },
        "features": {
            "encoder": dict(encoder),
            "cache_sha256": feature_cache_sha256,
        },
        "model": _stage0_semantic_model_recipe(model, encoder),
        "training": dict(training),
        "loss": {
            "pressure_weight": training.get("pressure_weight"),
            "pure_weight": training.get("pure_weight"),
            "direct_ge_supervision": None,
            "physics_finetuning": None,
        },
        "selection": {
            "partition": "validation",
            "evaluation_partition": "validation",
            "test_metrics_used_for_selection": False,
        },
    }


def requested_run_fingerprint(
    config_path: Path,
    split_path: Path,
    seed: int,
    feature_cache: Path,
    device_name: str | None,
    overrides: Sequence[str] = (),
    run_kind: str = "formal",
    evaluation_partition: str = "test",
    stage1_checkpoint: Path | None = None,
    aggregate_expected: bool = True,
    git_commit_override: str | None = None,
    analysis_status: str = "confirmatory",
) -> str:
    """Hash every cheap-to-check input needed to resume an existing run."""
    experiment = load_experiment_config(config_path, overrides)
    requested_device = device_name or experiment.runtime.device
    runtime = _runtime_context(requested_device)
    return _json_digest(
        {
            "git_commit": git_commit_override or _git_commit(),
            "resolved_config_sha256": _normalized_experiment_digest(experiment),
            "split_sha256": _file_digest(split_path),
            "feature_cache_sha256": (
                _file_digest(feature_cache) if feature_cache.is_file() else None
            ),
            "seed": int(seed),
            "runtime": runtime,
            "run_kind": run_kind,
            "evaluation_partition": evaluation_partition,
            "stage1_checkpoint_sha256": (
                _file_digest(stage1_checkpoint)
                if stage1_checkpoint is not None and stage1_checkpoint.is_file()
                else None
            ),
            "aggregate_expected": aggregate_expected,
            "analysis_status": analysis_status,
        }
    )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_state() -> tuple[bool, bool, list[str]]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = []
    for line in completed.stdout.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    artifact_prefixes = ("results/", "figures/", "runs/", "checkpoints/", "cache/")
    code_paths = [path for path in paths if not path.startswith(artifact_prefixes)]
    return bool(paths), bool(code_paths), code_paths


def _config_source_paths(path: Path, ancestors: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    resolved = path.resolve()
    if resolved in ancestors:
        raise ValueError("Cyclic experiment configuration inheritance")
    text = resolved.read_text(encoding="utf-8")
    payload = (
        yaml.safe_load(text)
        if resolved.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text)
    )
    parent = payload.get("extends") if isinstance(payload, dict) else None
    if parent is None:
        return (resolved,)
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("Configuration 'extends' must be a non-empty path string")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return (*_config_source_paths(parent_path, (*ancestors, resolved)), resolved)


def _require_tracked_file(path: Path, label: str) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Formal {label} must be inside the project: {resolved}") from error
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Formal {label} is not tracked by Git: {resolved}")


def _require_committed_file(path: Path, label: str) -> None:
    """Require an audited input to be tracked and byte-equivalent to HEAD."""

    _require_tracked_file(path, label)
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Formal {label} differs from the committed HEAD version: {path}")


def _require_generated_stage1_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    expected_git_commit: str,
    expected_seed: int,
) -> None:
    """Validate a formal Stage 1 checkpoint produced earlier in this campaign."""

    checkpoint = checkpoint_path.resolve()
    manifest = manifest_path.resolve()
    project_root = PROJECT_ROOT.resolve()
    for path, label in ((checkpoint, "checkpoint"), (manifest, "manifest")):
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise RuntimeError(
                f"Generated Stage 1 {label} must be inside the project: {path}"
            ) from error
        if not path.is_file():
            raise RuntimeError(f"Generated Stage 1 {label} does not exist: {path}")

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Generated Stage 1 manifest is unreadable: {manifest}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Generated Stage 1 manifest must contain a JSON object")

    required = {
        "status": "completed",
        "run_kind": "formal",
        "analysis_status": "confirmatory",
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "seed": expected_seed,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Generated Stage 1 manifest has invalid {key}: "
                f"expected {expected!r}, found {payload.get(key)!r}"
            )

    artifact = payload.get("artifacts", {}).get("checkpoint")
    if not isinstance(artifact, dict):
        raise RuntimeError("Generated Stage 1 manifest lacks checkpoint provenance")
    recorded_path = artifact.get("path")
    recorded_hash = artifact.get("sha256")
    if not isinstance(recorded_path, str) or not isinstance(recorded_hash, str):
        raise RuntimeError("Generated Stage 1 checkpoint provenance is incomplete")
    declared_checkpoint = (project_root / recorded_path).resolve()
    if declared_checkpoint != checkpoint:
        raise RuntimeError(
            "Generated Stage 1 manifest references a different checkpoint: "
            f"{declared_checkpoint}"
        )
    actual_hash = artifact_sha256(checkpoint)
    if recorded_hash != actual_hash:
        raise RuntimeError(
            "Generated Stage 1 checkpoint hash differs from its manifest: "
            f"expected {recorded_hash}, found {actual_hash}"
        )


def _validate_formal_inputs(
    config_path: Path,
    split_path: Path,
    data_root: Path,
    source_filter: str,
    catalog_path: Path | None,
) -> None:
    for path in _config_source_paths(config_path):
        _require_tracked_file(path, "configuration")
    _require_tracked_file(split_path, "split")
    workbooks = discover_vle_workbooks(data_root, source_filter)
    if not workbooks:
        raise RuntimeError(f"Formal dataset contains no tracked workbooks: {data_root}")
    for workbook in workbooks:
        _require_tracked_file(workbook, "dataset workbook")
    if catalog_path is not None:
        _require_tracked_file(catalog_path, "pure-property catalog")


def _dependency_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_context(requested_device: str) -> dict[str, object]:
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cuda_unavailable"
    else:
        resolved_device = requested_device
    cuda_active = resolved_device == "cuda" and torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "unimol_tools": _dependency_version("unimol-tools"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_active else None,
        "cuda_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_active else None
        ),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_checkpoint(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_training_curves(path: Path, history: Sequence[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for entry in history:
        row: dict[str, object] = {
            "stage": entry["stage"],
            "epoch": entry["epoch"],
        }
        train = entry.get("train")
        validation = entry.get("validation")
        if isinstance(train, dict):
            row.update({f"train_{key}": value for key, value in train.items()})
        if isinstance(validation, dict):
            row.update({f"validation_{key}": value for key, value in validation.items()})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_data_root(experiment: ExperimentConfig) -> Path:
    root = Path(experiment.data.root)
    return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()


def _resolve_catalog(experiment: ExperimentConfig) -> Path | None:
    if not experiment.data.pure_property_catalog:
        return None
    path = Path(experiment.data.pure_property_catalog)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_paper_experiment(
    config_path: Path,
    split_path: Path,
    seed: int,
    run_root: Path,
    checkpoint_root: Path,
    results_root: Path,
    feature_cache: Path,
    device_name: str | None = None,
    overrides: Sequence[str] = (),
    allow_overwrite: bool = False,
    run_kind: str = "formal",
    evaluation_partition: str = "test",
    stage1_checkpoint: Path | None = None,
    generated_stage1_manifest: Path | None = None,
    aggregate_expected: bool = True,
    analysis_status: str = "confirmatory",
) -> dict[str, Any]:
    """Train/evaluate one immutable split and export every required artifact."""
    if run_kind not in {"formal", "pilot", "selection", "smoke"}:
        raise ValueError("run_kind must be formal, pilot, selection, or smoke")
    if evaluation_partition not in {"test", "validation"}:
        raise ValueError("evaluation_partition must be 'test' or 'validation'")
    if analysis_status not in {
        "confirmatory",
        "test_exposed_exploratory",
        "diagnostic",
    }:
        raise ValueError("analysis_status is invalid")
    if generated_stage1_manifest is not None and stage1_checkpoint is None:
        raise ValueError("A generated Stage 1 manifest requires a Stage 1 checkpoint")
    audited_run = run_kind in {"formal", "pilot", "selection"}
    git_commit = _git_commit()
    worktree_dirty, git_dirty, dirty_code_paths = _git_state()
    if audited_run and git_dirty:
        raise RuntimeError(
            "Audited runs require committed code/config/splits; dirty paths: "
            + ", ".join(dirty_code_paths[:10])
        )
    experiment = load_experiment_config(config_path, overrides)
    training = replace(experiment.training, seed=seed)
    experiment = replace(experiment, seed=seed, training=training)
    requested_device = device_name or experiment.runtime.device
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    else:
        device = torch.device(requested_device)

    data_root = _resolve_data_root(experiment)
    catalog_path = _resolve_catalog(experiment)
    pure_property_catalog_sha256 = (
        artifact_sha256(catalog_path) if catalog_path is not None else None
    )
    if audited_run:
        _validate_formal_inputs(
            config_path,
            split_path,
            data_root,
            experiment.data.source_filter,
            catalog_path,
        )
        if stage1_checkpoint is not None and generated_stage1_manifest is not None:
            _require_generated_stage1_checkpoint(
                stage1_checkpoint,
                generated_stage1_manifest,
                expected_git_commit=git_commit,
                expected_seed=seed,
            )
        elif stage1_checkpoint is not None:
            _require_committed_file(stage1_checkpoint, "Stage 1 checkpoint")
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
    if split.seed != seed:
        raise ValueError(f"Split seed {split.seed} does not match run seed {seed}")
    split_protocol = validate_protocol_name(split.protocol)
    if experiment.protocol is not None:
        experiment.protocol.validate_request(split_protocol, seed, evaluation_partition)
    protocol = result_protocol_name(experiment.name, split_protocol)
    if audited_run:
        expected_split_path = (
            PROJECT_ROOT / 'datasets/splits/vle' / split_protocol / f"seed_{seed}.json"
        ).resolve()
        if split_path.resolve() != expected_split_path:
            raise RuntimeError(
                f"Audited split must use the registered protocol path: {expected_split_path}"
            )
    run_dir = run_root / protocol / f"seed_{seed}"
    checkpoint_dir = checkpoint_root / protocol / f"seed_{seed}"
    result_dir = results_root / protocol / f"seed_{seed}"
    completion = result_dir / "manifest.json"
    if completion.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Completed result already exists: {completion}; pass allow_overwrite=True explicitly"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    running_manifest = {
        "status": "running",
        "protocol": protocol,
        "seed": seed,
        "git_commit": git_commit,
        "run_kind": run_kind,
        "analysis_status": analysis_status,
    }
    invalidated_aggregate = {
            "status": "invalidated",
            "protocol": protocol,
            "invalidated_by_seed": seed,
            "git_commit": git_commit,
            "reason": "a seed run started; aggregate is invalid until all formal seeds are revalidated",
    }
    if aggregate_expected:
        for marker in ("aggregate_manifest.json", "diagnostic_aggregate_manifest.json"):
            _atomic_json(result_dir.parent / marker, invalidated_aggregate)
    # Invalidate every old completion marker before overwriting any artifact.
    # A crash can therefore never leave an old "completed" manifest pointing
    # at a partially replaced checkpoint or CSV.
    _atomic_json(run_dir / "manifest.json", running_manifest)
    _atomic_json(completion, running_manifest)

    unique_smiles = sorted({value for sample in samples for value in sample.smiles})
    encoder = build_molecular_encoder(
        experiment.encoder,
        feature_cache,
        use_cuda=device.type == "cuda",
    )
    seed_everything(seed)
    train_smiles = sorted({value for sample in split.train for value in sample.smiles})
    prepared_features = prepare_partition_features(
        encoder,
        unique_smiles,
        train_smiles,
    )
    feature_map = prepared_features.values
    feature_cache_sha256 = _file_digest(feature_cache)
    feature_subset_sha256 = _feature_subset_digest(feature_map)
    feature_definition_sha256 = str(
        prepared_features.metadata["feature_definition_sha256"]
    )
    source_cache_sha256 = prepared_features.metadata.get("source_cache_sha256", {})
    rdkit_scaler = prepared_features.metadata.get("rdkit_scaler", {})
    feature_dim = int(next(iter(feature_map.values())).shape[0])
    if experiment.model.feature_dim not in (None, feature_dim):
        raise ValueError(
            f"Configured feature_dim {experiment.model.feature_dim} != molecular features {feature_dim}"
        )
    view_dimensions = prepared_features.view_dimensions
    multiview = experiment.encoder.fusion_mode != "legacy"
    model_config = replace(
        experiment.model,
        feature_dim=feature_dim,
        fusion_mode=experiment.encoder.fusion_mode,
        rdkit_feature_dim=(view_dimensions["rdkit_2d"] if multiview else 0),
        unimol_feature_dim=(view_dimensions["unimol_v2"] if multiview else 0),
        functional_group_feature_dim=(
            view_dimensions["functional_groups"] if multiview else 0
        ),
        chemical_attention_bias=experiment.encoder.chemical_attention_bias,
        context_pair_interaction=experiment.encoder.context_pair_interaction,
    )
    resolved_experiment = replace(experiment, model=model_config)
    resolved_payload = resolved_experiment.to_dict()
    resolved_config_sha256 = _normalized_experiment_digest(resolved_experiment)
    split_sha256 = _file_digest(split_path)
    stage0_recipe = (
        _formal_c1_stage0_recipe(
            experiment,
            split_path,
            split_protocol,
            seed,
            dataset_digest(samples),
            feature_cache_sha256,
            pure_property_catalog_sha256,
        )
        if (
            run_kind == "formal"
            and stage1_checkpoint is None
            and evaluation_partition == "validation"
        )
        else None
    )
    stage0_recipe_sha256 = (
        _stage0_recipe_digest(stage0_recipe) if stage0_recipe is not None else None
    )
    request_sha256 = requested_run_fingerprint(
        config_path,
        split_path,
        seed,
        feature_cache,
        device_name,
        overrides,
        run_kind,
        evaluation_partition,
        stage1_checkpoint,
        aggregate_expected,
        analysis_status=analysis_status,
    )
    runtime_context = _runtime_context(requested_device)
    environment_sha256 = _json_digest(runtime_context)
    resolved_payload["paper_protocol"] = {
        "name": protocol,
        "split_protocol": split_protocol,
        "seed": seed,
        "split_file": portable_artifact_path(split_path),
        "analysis_status": analysis_status,
        "dataset_sha256": dataset_digest(samples),
    }
    resolved_payload["molecular_feature_preprocessing"] = prepared_features.metadata
    _atomic_json(run_dir / "resolved_config.json", resolved_payload)
    _atomic_json(result_dir / "resolved_config.json", resolved_payload)

    # Uni-Mol inference/conformer generation may consume framework RNG state on
    # a cache miss.  Reseed immediately before model construction so a cache hit
    # and a cache miss produce identical ThermoFormer initialization/training.
    seed_everything(seed)
    model = ThermoFormer(model_config)
    initially_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    catalog = (
        load_pure_property_catalog(catalog_path)
        if catalog_path is not None
        else empty_pure_property_catalog()
    )
    stage1_checkpoint_sha256 = None
    if experiment.direct_ge_supervision is not None:
        if stage1_checkpoint is None:
            raise ValueError("Direct-GE training requires the fixed C1 baseline checkpoint")
        if experiment.physics_finetuning is None:
            raise ValueError("Direct-GE Stage 3 requires physics_finetuning configuration")
        stage1_checkpoint_sha256 = load_stage1_checkpoint(
            model,
            stage1_checkpoint,
            expected_provenance={
                "dataset_sha256": dataset_digest(samples),
                "split_sha256": split_sha256,
                "feature_subset_sha256": feature_subset_sha256,
            },
        )
        baseline_state = cpu_state_dict(model)
        result = fit_direct_ge_stages(
            model,
            split.train,
            feature_map,
            training,
            experiment.direct_ge_supervision,
            experiment.physics_finetuning,
            device,
            validation_samples=split.validation,
            pure_property_catalog=catalog,
            baseline_state=baseline_state,
        )
    elif stage1_checkpoint is None:
        result = fit_model(
            model,
            split.train,
            feature_map,
            training,
            device,
            validation_samples=split.validation,
            pure_property_catalog=catalog,
        )
    else:
        if experiment.physics_finetuning is None:
            raise ValueError("Stage 1 checkpoint requires physics_finetuning configuration")
        stage1_checkpoint_sha256 = load_stage1_checkpoint(
            model,
            stage1_checkpoint,
            expected_provenance={
                "dataset_sha256": dataset_digest(samples),
                "split_sha256": split_sha256,
                "feature_subset_sha256": feature_subset_sha256,
                "feature_definition_sha256": feature_definition_sha256,
            },
        )
        result = fit_physics_stage(
            model,
            split.train,
            feature_map,
            training,
            experiment.physics_finetuning,
            device,
            validation_samples=split.validation,
            pure_property_catalog=catalog,
        )
    training_seconds = time.perf_counter() - started
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    inference_started = time.perf_counter()
    evaluation_samples = (
        split.test if evaluation_partition == "test" else split.validation
    )
    stage_comparison: dict[str, object] | None = None
    stage_predictions: dict[str, list[dict[str, Any]]] = {}
    trained_candidate_stage: str | None = None
    is_direct_ge = experiment.direct_ge_supervision is not None
    direct_stage_names = ("stage0", "stage1", "stage2", "stage3")
    if stage1_checkpoint is not None:
        stage_rows: dict[str, object] = {}
        if is_direct_ge:
            missing_stages = tuple(
                stage_name
                for stage_name in direct_stage_names
                if stage_name not in result.stage_states
            )
            if missing_stages:
                raise RuntimeError(
                    "Direct-GE result is missing persisted stage state(s): "
                    + ", ".join(missing_stages)
                )
            trained_candidate_stage = min(
                (stage_name for stage_name in direct_stage_names if stage_name != "stage0"),
                key=lambda stage_name: result.stage_validation_losses[stage_name],
            )
            # Every stage is evaluated only for post-selection diagnostics. The
            # selected checkpoint remains the one chosen using validation data.
            comparison_stage_names = direct_stage_names
        else:
            comparison_stage_names = tuple(result.stage_states)
        for stage_name in comparison_stage_names:
            stage_state = result.stage_states[stage_name]
            model.load_state_dict(stage_state)
            current_predictions = predict_vle(
                model,
                evaluation_samples,
                feature_map,
                batch_size=training.batch_size,
                device=device,
                solver_iterations=training.solver_iterations_eval,
                pure_property_catalog=catalog,
            )
            stage_predictions[stage_name] = current_predictions
            validation_score = result.stage_validation_losses[stage_name]
            stage_rows[stage_name] = {
                "metrics": prediction_metric_rows(current_predictions),
                "physics_residuals": evaluate_physics_residuals(
                    model,
                    evaluation_samples,
                    feature_map,
                    training,
                    device,
                    pure_property_catalog=catalog,
                ),
                "validation_loss": validation_score,
                "validation_score": validation_score,
                "selected_as_final": stage_name == result.selected_stage,
            }
            if is_direct_ge:
                stage_rows[stage_name]["best_epoch"] = int(
                    result.stage_best_epochs[stage_name]
                )
                stage_rows[stage_name]["validation_metrics"] = (
                    result.stage_validation_metrics[stage_name]
                )
                stage_rows[stage_name]["validation_label_metrics"] = (
                    result.stage_label_metrics[stage_name]
                )
                stage_rows[stage_name]["evaluation_label_metrics"] = (
                    evaluate_direct_label_metrics(
                        model,
                        evaluation_samples,
                        feature_map,
                        training,
                        device,
                        catalog,
                        minimum_fraction=experiment.direct_ge_supervision.minimum_fraction,
                    )
                )
        stage_comparison = {
            "selection_partition": "validation",
            "evaluation_partition": evaluation_partition,
            "evaluation_purpose": "diagnostic_only",
            "test_metrics_used_for_selection": False,
            "selected_stage": result.selected_stage,
            "trained_candidate_stage": trained_candidate_stage,
            "stages": stage_rows,
            "parameter_summary": result.parameter_summary,
            "stage1_checkpoint": portable_artifact_path(stage1_checkpoint),
            "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
            "thermodynamic_loss_weights": (
                {
                    "excess_gibbs": experiment.direct_ge_supervision.excess_gibbs_weight,
                    "activity_coefficient": experiment.direct_ge_supervision.activity_coefficient_weight,
                    "vle": experiment.direct_ge_supervision.vle_weight,
                    "teacher_forced_fugacity": experiment.direct_ge_supervision.fugacity_weight,
                }
                if is_direct_ge
                else {
                    "teacher_forced_fugacity": (
                        experiment.physics_finetuning.teacher_forced_fugacity_weight
                    ),
                }
            ),
        }
        model.load_state_dict(result.state_dict)
        predictions = stage_predictions[result.selected_stage]
    else:
        predictions = predict_vle(
            model,
            evaluation_samples,
            feature_map,
            batch_size=training.batch_size,
            device=device,
            solver_iterations=training.solver_iterations_eval,
            pure_property_catalog=catalog,
        )
    inference_seconds = time.perf_counter() - inference_started
    if split_protocol.startswith("binary_to_ternary"):
        subsystem_coverage = {
            str(row["ternary_system_id"]): int(row["covered_binary_subsystems"])
            for row in ternary_subsystem_rows(
                evaluation_samples, binary_reference_samples=split.train
            )
        }
        for record in predictions:
            record["binary_subsystem_coverage"] = subsystem_coverage.get(
                str(record["system_id"])
            )
    if split_protocol == "unseen_component" and evaluation_partition == "test":
        strict_ids = set(split.metadata.get("strict_unseen_sample_ids", []))
        for record in predictions:
            record["strict_unseen"] = record["sample_id"] in strict_ids
    metric_rows = prediction_metric_rows(predictions)
    physical_consistency = (
        evaluate_thermodynamic_consistency(
            model,
            split.test,
            feature_map,
            device,
            prediction_records=predictions,
            solver_iterations=training.solver_iterations_eval,
            grid_points=5 if run_kind == "smoke" else 21,
            max_systems=2 if run_kind == "smoke" else None,
            pure_reference_samples=split.train,
            pure_property_catalog=catalog,
        )
        if evaluation_partition == "test"
        else {
            "status": "not_evaluated",
            "reason": "validation-only architecture selection",
        }
    )
    checkpoint_payload = {
        "model_name": "ThermoFormer",
        "model": result.state_dict,
        "model_config": model_config.to_dict(),
        "training_config": asdict(training),
        "protocol": protocol,
        "split_protocol": split_protocol,
        "seed": seed,
        "split_file": str(split_path.resolve()),
        "dataset_sha256": dataset_digest(samples),
        "split_sha256": split_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "feature_cache_sha256": feature_cache_sha256,
        "feature_subset_sha256": feature_subset_sha256,
        "feature_definition_sha256": feature_definition_sha256,
        "rdkit_descriptor_list_sha256": prepared_features.metadata.get(
            "rdkit_descriptor_definition_sha256"
        ),
        "rdkit_scaler_sha256": (
            rdkit_scaler.get("sha256") if isinstance(rdkit_scaler, dict) else None
        ),
        "functional_group_vocabulary_sha256": prepared_features.metadata.get(
            "functional_group_vocabulary_sha256"
        ),
        "pure_property_catalog_sha256": pure_property_catalog_sha256,
        "unimol_cache_sha256": (
            source_cache_sha256.get("unimol_v2")
            if isinstance(source_cache_sha256, dict)
            else None
        ),
        "molecular_feature_preprocessing": prepared_features.metadata,
        "environment_sha256": environment_sha256,
        "request_sha256": request_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_worktree_dirty": worktree_dirty,
        "dirty_code_paths": dirty_code_paths,
        "run_kind": run_kind,
        "analysis_status": analysis_status,
        "evaluation_partition": evaluation_partition,
        "selection_partition": "validation",
        "test_metrics_used_for_selection": False,
        "stage0_recipe": stage0_recipe,
        "stage0_recipe_sha256": stage0_recipe_sha256,
        "best_validation_loss": result.best_validation_loss,
        "stage_validation_scores": (
            dict(result.stage_validation_losses) if is_direct_ge else None
        ),
        "stage_best_epochs": (
            dict(result.stage_best_epochs) if is_direct_ge else None
        ),
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "selected_stage": getattr(result, "selected_stage", "experimental"),
        "physics_parameter_summary": getattr(result, "parameter_summary", None),
        "aggregate_expected": aggregate_expected,
        "units": {"temperature": "K", "pressure": "kPa"},
    }
    checkpoint_path = checkpoint_dir / "best_model.pt"
    _atomic_checkpoint(checkpoint_path, checkpoint_payload)
    stage_checkpoint_paths: dict[str, Path] = {}
    stage2_checkpoint_path = checkpoint_dir / "stage2_best_model.pt"
    if is_direct_ge:
        for stage_name in direct_stage_names:
            stage_checkpoint_path = checkpoint_dir / f"{stage_name}_best_model.pt"
            stage_payload = {
                **checkpoint_payload,
                "model": result.stage_states[stage_name],
                "checkpoint_role": "best_stage_by_validation",
                "stage": stage_name,
                "stage_best_epoch": result.stage_best_epochs[stage_name],
                "validation_loss": result.stage_validation_losses[stage_name],
                "validation_score": result.stage_validation_losses[stage_name],
                "selected_as_final": stage_name == result.selected_stage,
            }
            if stage_name == "stage2":
                stage_payload["legacy_checkpoint_role"] = "best_physics_epoch_by_validation"
            _atomic_checkpoint(stage_checkpoint_path, stage_payload)
            stage_checkpoint_paths[stage_name] = stage_checkpoint_path
    elif stage1_checkpoint is not None:
        # Retain the historical Stage 2 artifact for the two-stage physics
        # fine-tuning runner.
        stage2_payload = {
            **checkpoint_payload,
            "model": result.stage_states["stage2"],
            "checkpoint_role": "best_physics_epoch_by_validation",
            "selected_as_final": result.selected_stage == "stage2",
            "validation_loss": result.stage_validation_losses["stage2"],
        }
        _atomic_checkpoint(stage2_checkpoint_path, stage2_payload)
        stage_checkpoint_paths["stage2"] = stage2_checkpoint_path
    trained_candidate_checkpoint_path = checkpoint_dir / "trained_candidate_best_model.pt"
    if trained_candidate_stage is not None:
        trained_candidate_payload = {
            **checkpoint_payload,
            "model": result.stage_states[trained_candidate_stage],
            "checkpoint_role": "best_trained_stage_by_validation",
            "trained_candidate_stage": trained_candidate_stage,
            "selected_as_final": result.selected_stage == trained_candidate_stage,
            "validation_loss": result.stage_validation_losses[trained_candidate_stage],
        }
        _atomic_checkpoint(
            trained_candidate_checkpoint_path, trained_candidate_payload
        )
    if is_direct_ge and stage_comparison is not None:
        stage_entries = stage_comparison["stages"]
        if not isinstance(stage_entries, dict):
            raise RuntimeError("Direct-GE stage comparison is malformed")
        for stage_name, stage_checkpoint_path in stage_checkpoint_paths.items():
            stage_entry = stage_entries.get(stage_name)
            if not isinstance(stage_entry, dict):
                raise RuntimeError(f"Direct-GE stage entry is missing: {stage_name}")
            stage_entry["checkpoint"] = portable_artifact_path(stage_checkpoint_path)
            stage_entry["checkpoint_sha256"] = _file_digest(stage_checkpoint_path)
    history_path = run_dir / "history.json"
    curves_path = run_dir / "training_curves.csv"
    predictions_path = result_dir / "predictions.csv"
    metrics_path = result_dir / "metrics.json"
    physical_consistency_path = result_dir / "physical_consistency.json"
    stage_comparison_path = result_dir / "stage_comparison.json"
    result_config_path = result_dir / "resolved_config.json"
    _atomic_json(history_path, result.history)
    _write_training_curves(curves_path, result.history)
    write_prediction_csv(predictions_path, predictions)
    stage_prediction_paths: dict[str, Path] = {}
    if is_direct_ge and stage_comparison is not None:
        stage_entries = stage_comparison["stages"]
        if not isinstance(stage_entries, dict):
            raise RuntimeError("Direct-GE stage comparison is malformed")
        for stage_name in direct_stage_names:
            stage_prediction_path = result_dir / f"{stage_name}_predictions.csv"
            write_prediction_csv(stage_prediction_path, stage_predictions[stage_name])
            stage_prediction_paths[stage_name] = stage_prediction_path
            stage_entry = stage_entries.get(stage_name)
            if not isinstance(stage_entry, dict):
                raise RuntimeError(f"Direct-GE stage entry is missing: {stage_name}")
            stage_entry["predictions"] = portable_artifact_path(stage_prediction_path)
            stage_entry["predictions_sha256"] = _file_digest(stage_prediction_path)
    _atomic_json(metrics_path, metric_rows)

    _atomic_json(physical_consistency_path, physical_consistency)
    if stage_comparison is not None:
        _atomic_json(stage_comparison_path, stage_comparison)
    artifact_paths = {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "training_curves": curves_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "physical_consistency": physical_consistency_path,
        "resolved_config": result_config_path,
    }
    if stage_comparison is not None:
        artifact_paths["stage_comparison"] = stage_comparison_path
        if is_direct_ge:
            for stage_name, stage_checkpoint_path in stage_checkpoint_paths.items():
                artifact_paths[f"{stage_name}_checkpoint"] = stage_checkpoint_path
            for stage_name, stage_prediction_path in stage_prediction_paths.items():
                artifact_paths[f"{stage_name}_predictions"] = stage_prediction_path
        else:
            artifact_paths["stage2_checkpoint"] = stage2_checkpoint_path
        if trained_candidate_stage is not None:
            artifact_paths["trained_candidate_checkpoint"] = (
                trained_candidate_checkpoint_path
            )
    artifacts = {
        name: {"path": portable_artifact_path(path), "sha256": _file_digest(path)}
        for name, path in artifact_paths.items()
    }
    manifest: dict[str, Any] = {
        "status": "smoke" if run_kind == "smoke" else "completed",
        "protocol": protocol,
        "split_protocol": split_protocol,
        "seed": seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_worktree_dirty": worktree_dirty,
        "dirty_code_paths": dirty_code_paths,
        "run_kind": run_kind,
        "analysis_status": analysis_status,
        "evaluation_partition": evaluation_partition,
        "selection_partition": "validation",
        "test_metrics_used_for_selection": False,
        "stage0_recipe": stage0_recipe,
        "stage0_recipe_sha256": stage0_recipe_sha256,
        "dataset_sha256": dataset_digest(samples),
        "split_sha256": split_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "feature_cache_sha256": feature_cache_sha256,
        "feature_subset_sha256": feature_subset_sha256,
        "feature_definition_sha256": feature_definition_sha256,
        "rdkit_descriptor_list_sha256": prepared_features.metadata.get(
            "rdkit_descriptor_definition_sha256"
        ),
        "rdkit_scaler_sha256": (
            rdkit_scaler.get("sha256") if isinstance(rdkit_scaler, dict) else None
        ),
        "functional_group_vocabulary_sha256": prepared_features.metadata.get(
            "functional_group_vocabulary_sha256"
        ),
        "pure_property_catalog_sha256": pure_property_catalog_sha256,
        "unimol_cache_sha256": (
            source_cache_sha256.get("unimol_v2")
            if isinstance(source_cache_sha256, dict)
            else None
        ),
        "molecular_feature_preprocessing": prepared_features.metadata,
        "environment_sha256": environment_sha256,
        "request_sha256": request_sha256,
        "split_file": portable_artifact_path(split_path),
        "split_metadata": split.metadata,
        "rows": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "evaluated_partition": evaluation_partition,
            "evaluated_rows": len(evaluation_samples),
            "prediction_attempts": len(predictions),
        },
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_attempt": 1000.0 * inference_seconds / max(1, len(predictions)),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "total_parameters": total_parameters,
        "initially_trainable_parameters": initially_trainable_parameters,
        "trainable_parameters": (
            result.parameter_summary["trainable_parameters"]
            if getattr(result, "parameter_summary", None) is not None
            else initially_trainable_parameters
        ),
        "best_validation_loss": result.best_validation_loss,
        "stage1_checkpoint": (
            portable_artifact_path(stage1_checkpoint)
            if stage1_checkpoint is not None
            else None
        ),
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "selected_stage": getattr(result, "selected_stage", "experimental"),
        "physics_parameter_summary": getattr(result, "parameter_summary", None),
        "checkpoint": portable_artifact_path(checkpoint_path),
        "artifact_path_schema": "project-relative-v1",
        "artifacts": artifacts,
        "runtime": {**runtime_context, "device": str(device)},
        "headline_metrics": next(row for row in metric_rows if row["scope"] == "all"),
        "physical_consistency": physical_consistency,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    _atomic_json(completion, manifest)
    return manifest
