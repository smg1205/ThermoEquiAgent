"""Run a validation-only smoke test for SPT-NRTL adapted (ThermoFormer-train)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_external_activity_baselines_smoke import _json_finite, _validation_rows
from src.thermoformer.baselines.machine_learning.formal import FormalConfig, _fixed_external_activity_result
from src.thermoformer.baselines.machine_learning.spt_nrtl_adapted import (
    AdaptedSPTConfig,
    AdaptedSPTNRTLPredictor,
    fit_pair_label,
    train_adapted_spt,
)
from src.thermoformer.baselines.thermodynamic_fitting import fit_vapor_pressure_correlations
from src.thermoformer.data.splitting import canonical_smiles, load_split_assignment
from src.thermoformer.baselines.machine_learning.protocols import load_registered_vle_dataset
from src.thermoformer.reporting.artifacts import artifact_sha256, atomic_write_json


IMPLEMENTATION_FILES = (
    "src/thermoformer/baselines/machine_learning/external_activity.py",
    "src/thermoformer/baselines/machine_learning/formal.py",
    "src/thermoformer/baselines/machine_learning/spt_nrtl.py",
    "src/thermoformer/baselines/machine_learning/spt_nrtl_adapted.py",
    "scripts/run_spt_nrtl_adapted_smoke.py",
    "scripts/run_ml_baselines_formal.py",
)


def _labels(rows, vapor_pressure, config, limit):
    grouped = defaultdict(list)
    for row in rows:
        if row.component_count == 2:
            grouped[tuple(sorted(canonical_smiles(value) for value in row.smiles))].append(row)
    labels = []
    for pair in sorted(grouped, key=lambda value: (-len(grouped[value]), value)):
        label = fit_pair_label(grouped[pair], vapor_pressure, config)
        if label is not None:
            labels.append(label)
        if len(labels) >= limit:
            break
    return tuple(labels)


def main() -> None:
    samples = load_registered_vle_dataset(PROJECT_ROOT)
    split = load_split_assignment(
        PROJECT_ROOT / 'datasets/splits/vle/vle_overall_binary_ternary/seed_0.json', samples
    )
    train, selected_validation = _validation_rows()
    vapor_pressure, _ = fit_vapor_pressure_correlations(train)
    config = AdaptedSPTConfig(
        max_sequence_length=128,
        embedding_dimension=32,
        attention_heads=4,
        transformer_layers=1,
        feedforward_dimension=64,
        dropout=0.0,
        batch_size=4,
        epochs=2,
        patience=2,
        minimum_pair_rows=5,
        label_fit_evaluations=30,
        label_regularization=1.0e-3,
    )
    training_labels = _labels(split.train, vapor_pressure, config, 8)
    validation_labels = _labels(split.validation, vapor_pressure, config, 3)
    trained = train_adapted_spt(training_labels, validation_labels, config, torch.device("cpu"))
    predictor = AdaptedSPTNRTLPredictor(trained, torch.device("cpu"))
    result = _fixed_external_activity_result(
        "spt_nrtl_adapted",
        predictor,
        train,
        selected_validation,
        "joint_train_joint_test",
        FormalConfig(device="cpu"),
        {},
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = _json_finite({
        "schema_version": 1,
        "status": "smoke_passed" if len(result.predictions) == 4 else "smoke_failed",
        "run_kind": "smoke",
        "analysis_status": "diagnostic",
        "selection_partition": "validation",
        "evaluation_partition": "validation",
        "test_rows_used": False,
        "baseline": "spt_nrtl_adapted",
        "registered_cells": 4,
        "valid_predictions": len(result.predictions),
        "training_pair_labels": len(training_labels),
        "validation_pair_labels": len(validation_labels),
        "best_validation_loss": trained.best_validation_loss,
        "trainable_parameters": trained.trainable_parameters,
        "metrics": list(result.rows),
        "git_commit": commit,
        "requirements_sha256": artifact_sha256(PROJECT_ROOT / "requirements.txt"),
        "environment_sha256": artifact_sha256(PROJECT_ROOT / "environment.yml"),
        "implementation_sha256": {
            path: artifact_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION_FILES
        },
    })
    destination = (
        PROJECT_ROOT / 'experiments/vle/comparison/diagnostic_records/external_activity_baselines_smoke'
        / "spt_nrtl_adapted" / "smoke_result.json"
    )
    atomic_write_json(destination, payload)
    print(json.dumps({"output": str(destination), "status": payload["status"], "valid_predictions": len(result.predictions)}, indent=2))


if __name__ == "__main__":
    main()

