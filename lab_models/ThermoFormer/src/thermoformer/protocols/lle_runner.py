"""Independent LLE experiment runner; it never mixes VLE and LLE batches."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from ..configuration import PhysicsFineTuningConfig, load_experiment_config
from ..data import lle_dataset_digest, load_lle_dataset, load_lle_split
from ..features import build_molecular_encoder, prepare_partition_features
from ..models import ThermoFormer
from ..reporting import artifact_sha256, atomic_write_json
from ..training.lle import evaluate_lle, fit_lle_stages


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha(path: Path) -> str:
    return artifact_sha256(path)


def run_lle_experiment(
    config_path: Path, split_path: Path, seed: int, run_root: Path, checkpoint_root: Path,
    results_root: Path, feature_cache: Path, *, device_name: str | None = None,
    overrides: Sequence[str] = (), allow_overwrite: bool = False, run_kind: str = "formal",
) -> dict[str, Any]:
    experiment = load_experiment_config(config_path, overrides)
    if experiment.task_mode != "lle" or experiment.lle is None:
        raise ValueError("LLE runner requires task_mode=lle")
    if experiment.lle.component_count != 3:
        raise ValueError("The initial LLE runner is restricted to ternary_train_ternary_test")
    training = replace(experiment.training, seed=seed)
    experiment = replace(experiment, seed=seed, training=training)
    requested_device = device_name or experiment.runtime.device
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    data_root = _resolve(PROJECT_ROOT, experiment.data.root)
    loaded = load_lle_dataset(
        data_root, binary_workbook=experiment.lle.binary_workbook,
        ternary_workbook=experiment.lle.ternary_workbook, component_count=3,
        failed_weight=experiment.data.failed_weight,
    )
    split = load_lle_split(split_path, loaded.samples)
    if split.seed != seed:
        raise ValueError("LLE split seed does not match requested seed")
    if run_kind == "smoke":
        split = replace(
            split, train=split.train[:8], validation=split.validation[:4], test=split.test[:4]
        )
    protocol = f"{experiment.name}.on.{split.protocol}"
    run_dir, checkpoint_dir, result_dir = (root / protocol / f"seed_{seed}" for root in (run_root, checkpoint_root, results_root))
    completion = result_dir / "manifest.json"
    if completion.exists() and not allow_overwrite:
        raise FileExistsError(f"Completed LLE result exists: {completion}")
    for directory in (run_dir, checkpoint_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    encoder = build_molecular_encoder(experiment.encoder, feature_cache, use_cuda=device.type == "cuda")
    all_smiles = sorted({smiles for sample in loaded.samples for smiles in sample.smiles})
    train_smiles = sorted({smiles for sample in split.train for smiles in sample.smiles})
    features = prepare_partition_features(encoder, all_smiles, train_smiles)
    dimensions = features.view_dimensions
    model_config = replace(
        experiment.model, feature_dim=int(next(iter(features.values.values())).shape[0]),
        fusion_mode=experiment.encoder.fusion_mode,
        rdkit_feature_dim=dimensions["rdkit_2d"], unimol_feature_dim=dimensions["unimol_v2"],
        functional_group_feature_dim=dimensions["functional_groups"],
        chemical_attention_bias=experiment.encoder.chemical_attention_bias,
        context_pair_interaction=experiment.encoder.context_pair_interaction,
    )
    model = ThermoFormer(model_config)
    checkpoint = _resolve(PROJECT_ROOT, experiment.lle.stage1_checkpoint_template.format(seed=seed))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required VLE Stage 1 checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_config") != asdict(model_config):
        raise ValueError("VLE Stage 1 checkpoint model configuration does not match LLE C1")
    model.load_state_dict(payload["model"], strict=True)
    result = fit_lle_stages(
        model, split.train, split.validation, features.values, training, experiment.lle,
        experiment.physics_finetuning or PhysicsFineTuningConfig(), device,
    )
    metrics, predictions = evaluate_lle(model, split.test, features.values, training, experiment.lle, device)
    dataset_sha256 = lle_dataset_digest(loaded.samples)
    input_hashes = {
        "binary_workbook": _sha(data_root / experiment.lle.binary_workbook),
        "ternary_workbook": _sha(data_root / experiment.lle.ternary_workbook),
        "split": _sha(split_path),
        "stage1_checkpoint": _sha(checkpoint),
    }
    checkpoint_payload = {
        "model_name": "ThermoFormer", "task_mode": "lle", "model": result.state_dict,
        "model_config": model_config.to_dict(), "seed": seed, "split_protocol": split.protocol,
        "run_kind": run_kind, "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_sha256": input_hashes["stage1_checkpoint"],
        "selected_stage": result.selected_stage, "stage_validation": result.stage_validation,
        "stage_best_epochs": result.stage_best_epochs, "parameter_summary": result.parameter_summary,
        "dataset_sha256": dataset_sha256, "input_sha256": input_hashes,
    }
    stage_parent = {"stage1": "preloaded_vle_stage1", "stage2": "stage1", "stage3": "stage2"}
    stage_paths = {}
    for stage, state in result.stage_states.items():
        stage_path = checkpoint_dir / f"{stage}_best_model.pt"
        torch.save({**checkpoint_payload, "model": state, "stage": stage, "lineage_parent": stage_parent[stage]}, stage_path)
        stage_paths[stage] = stage_path
    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save({**checkpoint_payload, "lineage_parent": result.selected_stage}, checkpoint_path)
    prediction_path = result_dir / "predictions.json"
    metric_path = result_dir / "metrics.json"
    history_path = run_dir / "history.json"
    atomic_write_json(prediction_path, predictions)
    atomic_write_json(metric_path, metrics)
    atomic_write_json(history_path, result.history)
    manifest = {
        "status": "smoke" if run_kind == "smoke" else "completed", "task_mode": "lle",
        "protocol": protocol, "split_protocol": split.protocol, "seed": seed,
        "rows": {"train_raw_tielines": len(split.train), "validation_raw_tielines": len(split.validation), "test_raw_tielines": len(split.test)},
        "metrics": metrics, "stage_validation": result.stage_validation, "selected_stage": result.selected_stage,
        "input_sha256": input_hashes,
        "dataset_sha256": dataset_sha256,
        "artifacts": {name: {"path": str(path), "sha256": _sha(path)} for name, path in {
            "checkpoint": checkpoint_path, "stage1_checkpoint": stage_paths["stage1"],
            "stage2_checkpoint": stage_paths["stage2"], "stage3_checkpoint": stage_paths["stage3"],
            "predictions": prediction_path, "metrics": metric_path, "history": history_path,
        }.items()},
    }
    atomic_write_json(completion, manifest)
    return manifest
