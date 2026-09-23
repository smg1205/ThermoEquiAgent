"""Formal train/validation/test runner for audited ML VLE baselines."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ...data.loading import VLESample, discover_vle_workbooks
from ...data.splitting import canonical_smiles, dataset_digest, load_split_assignment, sample_id, system_id
from ...reporting.artifacts import artifact_sha256, atomic_write_json
from ...thermodynamics import solve_isobaric, solve_isothermal
from ..thermodynamic_fitting import fit_vapor_pressure_correlations
from .artifacts import aggregate_seed_metrics, write_seed_metrics
from .hanna import HANNA_SOURCE_REVISION, OfficialHANNAAssets, OfficialHANNAInference
from .external_activity import ExternalActivityThermodynamicAdapter, psat_coefficient_tensor
from .models import GDIGNN, GEGNN, SMILESRNN, SolvGNN, UALFGNN, trainable_parameter_count
from .protocols import benchmark_for_baseline, load_registered_vle_dataset
from .schema import BASELINE_CAPABILITIES
from .smoke import _graph_tensors, _hydrogen_bond_edges
from .spt_nrtl import (
    SPT_NRTL_DATABASE_REVISION,
    SPT_NRTL_SOURCE_REVISION,
    SPTNRTLDatabase,
    SPTNRTLPredictor,
    SPTParameterFileUnavailable,
    SPTParameterRowUnavailable,
)
from .spt_nrtl_adapted import (
    AdaptedSPTConfig,
    AdaptedSPTNRTLPredictor,
    fit_partition_pair_labels,
    train_adapted_spt,
)
from .tennet_sac import (
    CudaTeNNetSACPredictor,
    TENNETSAC_PACKAGE_VERSION,
    TENNETSAC_SOURCE_REVISION,
    TENNETSAC_WHEEL_SHA256,
    TeNNetSACPredictor,
    installed_tennetsac_asset_audit,
    verified_tennetsac_wheel,
)


FORMAL_SCHEMA_VERSION = 1
TRAINABLE_BASELINES = frozenset({"smiles_rnn", "ualf_gnn", "solvgnn", "gdi_gnn", "ge_gnn", "spt_nrtl_adapted"})
FIXED_EXTERNAL_BASELINES = frozenset({"hanna", "tennet_sac", "spt_nrtl"})
EXECUTABLE_BASELINES = TRAINABLE_BASELINES | FIXED_EXTERNAL_BASELINES


def _analysis_status(baseline: str, completed: bool) -> str:
    if not completed:
        return "coverage_audit"
    return (
        "external_pretrained_overlap_unknown"
        if baseline in FIXED_EXTERNAL_BASELINES
        else "confirmatory"
    )


def _selection_partition(baseline: str) -> str:
    if baseline == "hanna":
        return "not_applicable_official_pretrained"
    if baseline in {"tennet_sac", "spt_nrtl"}:
        return "not_applicable_fixed_external_model"
    return "validation"


def _runtime_provenance(baseline: str, device: str) -> dict[str, object]:
    """Describe the actual execution environment without claiming upstream parity."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "cuda": torch.version.cuda or "unavailable",
        "cudnn": str(torch.backends.cudnn.version() or "unavailable"),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else "N/A",
        "execution_profile": (
            "thermoformer_ggnn39_compatibility"
            if baseline == "hanna"
            else "thermoformer_ggnn39_external_compatibility"
            if baseline in {"tennet_sac", "spt_nrtl"}
            else "native_project_environment"
        ),
        "upstream_environment_parity": (
            "not_established" if baseline in FIXED_EXTERNAL_BASELINES else "not_applicable"
        ),
        "hanna_dependencies": (
            {
                package: importlib.metadata.version(package)
                for package in (
                    "transformers", "tokenizers", "safetensors", "scikit-learn", "rdkit", "pandas"
                )
            }
            if baseline == "hanna" else {}
        ),
        "tennetsac_dependencies": (
            {
                package: importlib.metadata.version(package)
                for package in ("tennetsac", "transformers", "tokenizers", "rdkit", "numpy")
            }
            if baseline == "tennet_sac" else {}
        ),
    }


@dataclass(frozen=True)
class FormalConfig:
    epochs: int = 200
    learning_rate: float = 1e-3
    patience: int = 25
    mc_dropout_samples: int = 32
    fixed_temperature_tolerance_k: float = 0.5
    minimum_composition: float = 1e-5
    gdi_weight: float = 1e-2
    device: str = "cuda"
    spt_adapted_max_sequence_length: int = 128
    spt_adapted_embedding_dimension: int = 128
    spt_adapted_attention_heads: int = 4
    spt_adapted_transformer_layers: int = 2
    spt_adapted_feedforward_dimension: int = 256
    spt_adapted_dropout: float = 0.10
    spt_adapted_batch_size: int = 32
    spt_adapted_epochs: int = 300
    spt_adapted_learning_rate: float = 3.0e-4
    spt_adapted_patience: int = 30
    spt_adapted_minimum_pair_rows: int = 5
    spt_adapted_label_fit_evaluations: int = 250
    spt_adapted_label_regularization: float = 1.0e-3

    def adapted_spt(self) -> AdaptedSPTConfig:
        return AdaptedSPTConfig(
            max_sequence_length=self.spt_adapted_max_sequence_length,
            embedding_dimension=self.spt_adapted_embedding_dimension,
            attention_heads=self.spt_adapted_attention_heads,
            transformer_layers=self.spt_adapted_transformer_layers,
            feedforward_dimension=self.spt_adapted_feedforward_dimension,
            dropout=self.spt_adapted_dropout,
            batch_size=self.spt_adapted_batch_size,
            epochs=self.spt_adapted_epochs,
            learning_rate=self.spt_adapted_learning_rate,
            patience=self.spt_adapted_patience,
            minimum_pair_rows=self.spt_adapted_minimum_pair_rows,
            label_fit_evaluations=self.spt_adapted_label_fit_evaluations,
            label_regularization=self.spt_adapted_label_regularization,
        )


@dataclass(frozen=True)
class DirectionResult:
    rows: tuple[dict[str, object], ...]
    predictions: tuple[dict[str, object], ...]
    checkpoint: dict[str, object] | None
    history: tuple[dict[str, float], ...]
    trainable_parameters: int


def _atomic_json(path: Path, payload: object) -> None:
    atomic_write_json(path, payload)


def _atomic_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV output requires at least one row")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eligible(rows: Sequence[VLESample], direction: str, count: int) -> tuple[VLESample, ...]:
    return tuple(
        row for row in rows
        if row.component_count == count and row.experiment_mode in (direction, "full_state")
    )


def _metric_values(observed: Sequence[float], predicted: Sequence[float]) -> tuple[float, float, float]:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(estimate)
    truth, estimate = truth[finite], estimate[finite]
    if not len(truth):
        return math.nan, math.nan, math.nan
    residual = estimate - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if len(truth) >= 2 and denominator > 0 else math.nan
    return mae, rmse, r2


def _metric_row(
    baseline: str,
    benchmark: str,
    direction: str,
    count: int | str,
    total: int,
    predictions: Sequence[dict[str, object]],
    status: str = "evaluated",
    solver_failures: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "baseline": baseline,
        "benchmark": benchmark,
        "direction": direction,
        "component_count": count,
        "status": status,
        "valid_coverage": len(predictions) / total if total else 0.0,
    }
    if solver_failures is not None:
        row["solver_failure_rate"] = solver_failures / total if total else 0.0
    if predictions:
        state = "pressure" if direction == "isothermal" else "temperature"
        mae, rmse, r2 = _metric_values(
            [float(value[f"observed_{state}"]) for value in predictions],
            [float(value[f"predicted_{state}"]) for value in predictions],
        )
        prefix = "pressure" if direction == "isothermal" else "temperature"
        row[f"{prefix}_mae_{'kpa' if prefix == 'pressure' else 'k'}"] = mae
        row[f"{prefix}_rmse_{'kpa' if prefix == 'pressure' else 'k'}"] = rmse
        row[f"{prefix}_r2"] = r2
        y_observed = [number for value in predictions for number in value["observed_y"]]
        y_predicted = [number for value in predictions for number in value["predicted_y"]]
        row["y_mae"], row["y_rmse"], row["y_r2"] = _metric_values(y_observed, y_predicted)
        row["nonphysical_rate"] = float(np.mean([bool(value["nonphysical"]) for value in predictions]))
    return row


class SmilesEncoder:
    """Train-only character vocabulary matching the source length-one RNN input."""

    def __init__(self, rows: Sequence[VLESample]) -> None:
        self.characters = tuple(sorted({character for row in rows for value in row.smiles[:2] for character in value}))
        self.index = {value: index for index, value in enumerate(self.characters)}
        self.maximum_length = max(len(value) for row in rows for value in row.smiles[:2])
        self.width_per_molecule = self.maximum_length * (len(self.characters) + 1)

    @property
    def width(self) -> int:
        return self.width_per_molecule * 2 + 2

    def transform(self, rows: Sequence[VLESample], direction: str) -> Tensor:
        values = torch.zeros(len(rows), self.width)
        unknown = len(self.characters)
        for row_index, row in enumerate(rows):
            for component, smiles in enumerate(row.smiles[:2]):
                for position, character in enumerate(smiles[: self.maximum_length]):
                    offset = component * self.width_per_molecule + position * (len(self.characters) + 1)
                    values[row_index, offset + self.index.get(character, unknown)] = 1.0
            condition = row.temperature_k / 500.0 if direction == "isothermal" else row.pressure_kpa / 500.0
            values[row_index, -2:] = torch.tensor([condition, row.liquid_composition[0]])
        return values


def _direct_targets(rows: Sequence[VLESample], direction: str) -> Tensor:
    return torch.tensor([
        [
            row.pressure_kpa / 500.0 if direction == "isothermal" else row.temperature_k / 500.0,
            row.vapor_composition[0],
        ]
        for row in rows
    ], dtype=torch.float32)


def _fit_with_validation(
    model: nn.Module,
    train_loss: Callable[[], Tensor],
    validation_loss: Callable[[], Tensor],
    config: FormalConfig,
) -> tuple[dict[str, Tensor], tuple[dict[str, float], ...]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best = copy.deepcopy(model.state_dict())
    best_value = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        model.eval()
        value = float(validation_loss().detach())
        history.append({"epoch": float(epoch + 1), "train_loss": float(loss.detach()), "validation_loss": value})
        if value < best_value - 1e-10:
            best_value = value
            best = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    model.load_state_dict(best)
    return best, tuple(history)


def _direct_result(
    baseline: str,
    direction: str,
    train: Sequence[VLESample],
    validation: Sequence[VLESample],
    test: Sequence[VLESample],
    benchmark: str,
    config: FormalConfig,
) -> DirectionResult:
    device = torch.device(config.device)
    train_rows = _eligible(train, direction, 2)
    validation_rows = _eligible(validation, direction, 2)
    test_rows = _eligible(test, direction, 2)
    if not train_rows or not validation_rows:
        row = _metric_row(baseline, benchmark, direction, 2, len(test_rows), (), "not_evaluated_no_training_coverage")
        return DirectionResult((row,), (), None, (), 0)
    if baseline == "smiles_rnn":
        encoder = SmilesEncoder(train_rows)
        train_input = encoder.transform(train_rows, direction).to(device)
        validation_input = encoder.transform(validation_rows, direction).to(device)
        model: nn.Module = SMILESRNN(encoder.width).to(device)
        train_target = _direct_targets(train_rows, direction).to(device)
        validation_target = _direct_targets(validation_rows, direction).to(device)
        train_loss = lambda: torch.mean((model(train_input) - train_target) ** 2)
        validation_loss = lambda: torch.mean((model(validation_input) - validation_target) ** 2)
        checkpoint_extra = {"characters": encoder.characters, "maximum_length": encoder.maximum_length, "direction": direction}
    else:
        if direction != "isobaric":
            row = _metric_row(baseline, benchmark, direction, 2, len(test_rows), (), "not_applicable")
            return DirectionResult((row,), (), None, (), 0)
        train_atoms, train_adjacency, train_mask = (value.to(device) for value in _graph_tensors(train_rows))
        val_atoms, val_adjacency, val_mask = (value.to(device) for value in _graph_tensors(validation_rows))
        train_condition = torch.tensor([[row.pressure_kpa / 500.0, row.liquid_composition[0]] for row in train_rows]).to(device)
        val_condition = torch.tensor([[row.pressure_kpa / 500.0, row.liquid_composition[0]] for row in validation_rows]).to(device)
        train_target = _direct_targets(train_rows, direction).to(device)
        validation_target = _direct_targets(validation_rows, direction).to(device)
        model = UALFGNN().to(device)
        train_loss = lambda: model.heteroscedastic_loss(*model(train_atoms, train_adjacency, train_mask, train_condition), train_target)
        validation_loss = lambda: torch.mean((model(val_atoms, val_adjacency, val_mask, val_condition)[0] - validation_target) ** 2)
        checkpoint_extra = {"direction": direction}
    best, history = _fit_with_validation(model, train_loss, validation_loss, config)
    predictions: list[dict[str, object]] = []
    model.eval()
    if test_rows:
        if baseline == "smiles_rnn":
            output = model(encoder.transform(test_rows, direction).to(device)).detach().cpu().numpy()
        else:
            atoms, adjacency, mask = (value.to(device) for value in _graph_tensors(test_rows))
            condition = torch.tensor([[row.pressure_kpa / 500.0, row.liquid_composition[0]] for row in test_rows]).to(device)
            output = model.mc_predict(atoms, adjacency, mask, condition, samples=config.mc_dropout_samples)[0].cpu().numpy()
        for sample, estimate in zip(test_rows, output):
            state_prediction = float(estimate[0] * 500.0)
            y1 = float(estimate[1])
            nonphysical = bool(
                not np.isfinite(estimate).all()
                or state_prediction <= 0.0
                or y1 < 0.0 or y1 > 1.0
            )
            predictions.append({
                "sample_id": sample_id(sample), "system_id": system_id(sample), "baseline": baseline,
                "benchmark": benchmark, "direction": direction, "component_count": 2,
                "observed_pressure": sample.pressure_kpa, "predicted_pressure": state_prediction if direction == "isothermal" else "",
                "observed_temperature": sample.temperature_k, "predicted_temperature": state_prediction if direction == "isobaric" else "",
                "observed_y": list(sample.vapor_composition), "predicted_y": [y1, 1.0 - y1],
                "nonphysical": nonphysical,
            })
    row = _metric_row(baseline, benchmark, direction, 2, len(test_rows), predictions)
    checkpoint = {"model_state_dict": best, "model": baseline, "direction": direction, "model_config": checkpoint_extra}
    return DirectionResult((row,), tuple(predictions), checkpoint, history, trainable_parameter_count(model))


def _psat_for(sample: VLESample, fitted: dict[str, Any]) -> np.ndarray | None:
    try:
        values = np.asarray([fitted[canonical_smiles(value)].pressure_kpa(sample.temperature_k) for value in sample.smiles])
    except KeyError:
        return None
    return values if np.isfinite(values).all() and np.all(values > 0.0) else None


def _gamma_rows(rows: Sequence[VLESample], fitted: dict[str, Any], tolerance: float, minimum_x: float) -> tuple[VLESample, ...]:
    selected = []
    for row in rows:
        if abs(row.temperature_k - 298.15) > tolerance:
            continue
        psat = _psat_for(row, fitted)
        x, y = np.asarray(row.liquid_composition), np.asarray(row.vapor_composition)
        if psat is not None and np.all(x > minimum_x) and np.all(y > minimum_x):
            log_gamma = np.log(y * row.pressure_kpa / (x * psat))
            if np.isfinite(log_gamma).all() and np.all(np.abs(log_gamma) <= 12.0):
                selected.append(row)
    return tuple(selected)


def _graph_batch(rows: Sequence[VLESample], count: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    atoms, adjacency, atom_mask = _graph_tensors(rows, count, atom_dim=74)
    x = torch.tensor([row.liquid_composition for row in rows], dtype=torch.float32)
    mixture_mask = torch.ones_like(x)
    edges = _hydrogen_bond_edges(rows, count)
    return tuple(value.to(device) for value in (atoms, adjacency, atom_mask, x, mixture_mask, edges))


def _graph_targets(rows: Sequence[VLESample], fitted: dict[str, Any]) -> Tensor:
    values = []
    for row in rows:
        psat = _psat_for(row, fitted)
        assert psat is not None
        values.append(np.log(np.asarray(row.vapor_composition) * row.pressure_kpa / (np.asarray(row.liquid_composition) * psat)))
    return torch.tensor(np.asarray(values), dtype=torch.float32)


def _graph_result(
    baseline: str,
    train: Sequence[VLESample],
    validation: Sequence[VLESample],
    test: Sequence[VLESample],
    benchmark: str,
    config: FormalConfig,
) -> DirectionResult:
    device = torch.device(config.device)
    vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(train)
    counts = benchmark_for_baseline(baseline).test_component_counts
    prepared: dict[int, dict[str, object]] = {}
    for count in counts:
        eligible_test = _eligible(test, "isothermal", count)
        training_rows = _gamma_rows(_eligible(train, "isothermal", count), vapor_pressure, config.fixed_temperature_tolerance_k, config.minimum_composition)
        validation_rows = _gamma_rows(_eligible(validation, "isothermal", count), vapor_pressure, config.fixed_temperature_tolerance_k, config.minimum_composition)
        evaluation_rows = _gamma_rows(eligible_test, vapor_pressure, config.fixed_temperature_tolerance_k, config.minimum_composition)
        prepared[count] = {
            "eligible_test": eligible_test,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "evaluation_rows": evaluation_rows,
        }
        if training_rows and validation_rows:
            prepared[count]["train_batch"] = _graph_batch(training_rows, count, device)
            prepared[count]["validation_batch"] = _graph_batch(validation_rows, count, device)
            prepared[count]["train_target"] = _graph_targets(training_rows, vapor_pressure).to(device)
            prepared[count]["validation_target"] = _graph_targets(validation_rows, vapor_pressure).to(device)

    usable_counts = tuple(
        count for count, values in prepared.items()
        if "train_batch" in values and not (baseline in {"gdi_gnn", "ge_gnn"} and count != 2)
    )
    if not usable_counts:
        rows = tuple(
            _metric_row(
                baseline, benchmark, "isothermal", count,
                len(prepared[count]["eligible_test"]), (),
                "not_applicable" if baseline in {"gdi_gnn", "ge_gnn"} and count != 2 else "not_evaluated_no_298k_coverage",
            )
            for count in counts
        )
        return DirectionResult(rows, (), None, (), 0)

    # SolvGNN uses one shared parameter set across every native cardinality with
    # available rows. Cardinality-specific tensors are separate batches only
    # because their shapes differ; their losses update the same model.
    model: nn.Module = SolvGNN() if baseline == "solvgnn" else (GDIGNN() if baseline == "gdi_gnn" else GEGNN())
    model = model.to(device)

    def forward(batch: tuple[Tensor, ...], training: bool) -> Tensor:
        atoms, adjacency, atom_mask, x, mask, edges = batch
        if baseline == "ge_gnn":
            return model.forward_graphs(atoms, adjacency, atom_mask, x, edges)
        if baseline == "gdi_gnn" and training:
            x = x.detach().clone().requires_grad_(True)
        return model(atoms, adjacency, atom_mask, x, mask, edges)

    def train_loss() -> Tensor:
        numerator = torch.zeros((), device=device)
        denominator = 0
        for count in usable_counts:
            values = prepared[count]
            batch = values["train_batch"]
            target = values["train_target"]
            prediction = forward(batch, True)
            loss = torch.mean((prediction - target) ** 2)
            if baseline == "gdi_gnn":
                x = batch[3].detach().clone().requires_grad_(True)
                prediction = model(batch[0], batch[1], batch[2], x, batch[4], batch[5])
                loss = torch.mean((prediction - target) ** 2) + config.gdi_weight * torch.mean(GDIGNN.gibbs_duhem_residual(prediction, x) ** 2)
            weight = len(values["training_rows"])
            numerator = numerator + loss * weight
            denominator += weight
        return numerator / max(denominator, 1)

    def validation_loss() -> Tensor:
        numerator = torch.zeros((), device=device)
        denominator = 0
        for count in usable_counts:
            values = prepared[count]
            loss = torch.mean((forward(values["validation_batch"], False) - values["validation_target"]) ** 2)
            weight = len(values["validation_rows"])
            numerator = numerator + loss * weight
            denominator += weight
        return numerator / max(denominator, 1)

    best, history = _fit_with_validation(model, train_loss, validation_loss, config)
    all_rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    for count in counts:
        values = prepared[count]
        eligible_test = values["eligible_test"]
        evaluation_rows = values["evaluation_rows"]
        if baseline in {"gdi_gnn", "ge_gnn"} and count != 2:
            all_rows.append(_metric_row(baseline, benchmark, "isothermal", count, len(eligible_test), (), "not_applicable"))
            continue
        if count not in usable_counts:
            all_rows.append(_metric_row(baseline, benchmark, "isothermal", count, len(eligible_test), (), "not_evaluated_no_298k_coverage"))
            continue
        predictions: list[dict[str, object]] = []
        if evaluation_rows:
            output = forward(_graph_batch(evaluation_rows, count, device), False).detach().cpu().numpy()
            for sample, log_gamma in zip(evaluation_rows, output):
                psat = _psat_for(sample, vapor_pressure)
                assert psat is not None
                contribution = np.asarray(sample.liquid_composition) * np.exp(log_gamma) * psat
                pressure = float(contribution.sum())
                vapor = contribution / max(pressure, 1e-12)
                predictions.append({
                    "sample_id": sample_id(sample), "system_id": system_id(sample), "baseline": baseline,
                    "benchmark": benchmark, "direction": "isothermal", "component_count": count,
                    "observed_pressure": sample.pressure_kpa, "predicted_pressure": pressure,
                    "observed_temperature": sample.temperature_k, "predicted_temperature": "",
                    "observed_y": list(sample.vapor_composition), "predicted_y": vapor.tolist(),
                    "nonphysical": bool(
                        not math.isfinite(pressure) or pressure <= 0.0
                        or not np.isfinite(vapor).all()
                        or np.any(vapor < 0.0) or np.any(vapor > 1.0)
                    ),
                })
        all_predictions.extend(predictions)
        status = "evaluated" if predictions else "not_evaluated_no_native_coverage"
        all_rows.append(_metric_row(baseline, benchmark, "isothermal", count, len(eligible_test), predictions, status))

    if len(counts) > 1:
        total_eligible = sum(len(prepared[count]["eligible_test"]) for count in counts)
        combined_status = "evaluated" if all_predictions else "not_evaluated_no_native_coverage"
        all_rows.append(
            _metric_row(
                baseline, benchmark, "isothermal", "2+3", total_eligible,
                all_predictions, combined_status,
            )
        )

    checkpoint = {
        "model": baseline,
        "model_state_dict": best,
        "trained_component_counts": list(usable_counts),
        "vapor_pressure_audit": vapor_audit,
        "native_temperature_k": 298.15,
    }
    return DirectionResult(
        tuple(all_rows), tuple(all_predictions), checkpoint, history,
        trainable_parameter_count(model),
    )


def _hanna_result(
    project_root: Path,
    train: Sequence[VLESample],
    test: Sequence[VLESample],
    benchmark: str,
    config: FormalConfig,
) -> DirectionResult:
    """Evaluate the unchanged official ensemble with train-only Psat fits."""
    device = torch.device(config.device)
    assets = OfficialHANNAAssets.default(project_root)
    inference = OfficialHANNAInference(assets, device)
    vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(train)
    counts = benchmark_for_baseline("hanna").test_component_counts
    rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    combined: dict[str, dict[str, object]] = {
        direction: {"total": 0, "failures": 0, "predictions": []}
        for direction in ("isothermal", "isobaric")
    }
    batch_size = 128
    for direction in ("isothermal", "isobaric"):
        for count in counts:
            eligible = _eligible(test, direction, count)
            covered = tuple(
                sample for sample in eligible
                if all(canonical_smiles(smiles) in vapor_pressure for smiles in sample.smiles)
            )
            predictions: list[dict[str, object]] = []
            failures = 0
            for start in range(0, len(covered), batch_size):
                samples = covered[start : start + batch_size]
                molecules = inference.molecule_tensor(samples, vapor_pressure)
                x = torch.tensor(
                    [sample.liquid_composition for sample in samples],
                    dtype=torch.float32,
                    device=device,
                )
                mask = torch.ones_like(x)
                with torch.enable_grad():
                    if direction == "isothermal":
                        state = solve_isothermal(
                            inference.model,
                            molecules,
                            torch.tensor(
                                [[sample.temperature_k] for sample in samples],
                                dtype=torch.float32,
                                device=device,
                            ),
                            x,
                            mask,
                            iterations=1,
                            damping=1.0,
                            strict=False,
                        )
                    else:
                        state = solve_isobaric(
                            inference.model,
                            molecules,
                            torch.tensor(
                                [[sample.pressure_kpa] for sample in samples],
                                dtype=torch.float32,
                                device=device,
                            ),
                            x,
                            mask,
                            strict=False,
                        )
                converged = state.converged.detach().cpu().reshape(-1).numpy().astype(bool)
                predicted_pressure = state.pressure_kpa.detach().cpu().reshape(-1).numpy()
                predicted_temperature = state.temperature_k.detach().cpu().reshape(-1).numpy()
                predicted_y = state.y.detach().cpu().numpy()
                for index, sample in enumerate(samples):
                    if not converged[index]:
                        failures += 1
                        continue
                    state_value = (
                        float(predicted_pressure[index])
                        if direction == "isothermal"
                        else float(predicted_temperature[index])
                    )
                    vapor = predicted_y[index]
                    nonphysical = bool(
                        not math.isfinite(state_value)
                        or state_value <= 0.0
                        or not np.isfinite(vapor).all()
                        or np.any(vapor < 0.0)
                        or np.any(vapor > 1.0)
                    )
                    predictions.append({
                        "sample_id": sample_id(sample),
                        "system_id": system_id(sample),
                        "baseline": "hanna",
                        "benchmark": benchmark,
                        "direction": direction,
                        "component_count": count,
                        "observed_pressure": sample.pressure_kpa,
                        "predicted_pressure": state_value if direction == "isothermal" else "",
                        "observed_temperature": sample.temperature_k,
                        "predicted_temperature": state_value if direction == "isobaric" else "",
                        "observed_y": list(sample.vapor_composition),
                        "predicted_y": vapor.tolist(),
                        "nonphysical": nonphysical,
                    })
            status = "evaluated" if predictions else "not_evaluated_no_psat_or_solver_coverage"
            rows.append(
                _metric_row(
                    "hanna", benchmark, direction, count, len(eligible), predictions,
                    status, failures,
                )
            )
            all_predictions.extend(predictions)
            combined[direction]["total"] += len(eligible)
            combined[direction]["failures"] += failures
            combined[direction]["predictions"].extend(predictions)
        direction_values = combined[direction]
        direction_predictions = direction_values["predictions"]
        rows.append(
            _metric_row(
                "hanna",
                benchmark,
                direction,
                "2+3",
                int(direction_values["total"]),
                direction_predictions,
                "evaluated" if direction_predictions else "not_evaluated_no_psat_or_solver_coverage",
                int(direction_values["failures"]),
            )
        )
    checkpoint = {
        "model": "hanna_official_pretrained",
        "official_source_revision": HANNA_SOURCE_REVISION,
        "official_asset_sha256": assets.hashes(project_root),
        "training_regime": "official_binary_training_with_muggianu_multicomponent_projection",
        "training_system_overlap": "unknown_official_training_inventory_not_published",
        "vapor_pressure_audit": vapor_audit,
        "parameter_accounting": {
            "ensemble_total": sum(parameter.numel() for parameter in inference.model.ensemble.parameters()),
            "chemberta_total": sum(parameter.numel() for parameter in inference.chemberta.parameters()),
            "optimized": 0,
        },
    }
    return DirectionResult(tuple(rows), tuple(all_predictions), checkpoint, (), 0)


def _fixed_external_activity_result(
    baseline: str,
    predictor: object,
    train: Sequence[VLESample],
    test: Sequence[VLESample],
    benchmark: str,
    config: FormalConfig,
    model_assets: dict[str, object],
) -> DirectionResult:
    """Evaluate a frozen external log-gamma source with the shared Psat/solver path."""
    device = torch.device(config.device)
    vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(train)
    counts = benchmark_for_baseline(baseline).test_component_counts
    rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    unavailable = {
        "missing_psat": 0,
        "missing_parameter_file": 0,
        "missing_parameter_row": 0,
        "solver_failure": 0,
    }
    combined: dict[str, dict[str, object]] = {
        direction: {"total": 0, "failures": 0, "predictions": []}
        for direction in ("isothermal", "isobaric")
    }
    for direction in ("isothermal", "isobaric"):
        for count in counts:
            eligible = _eligible(test, direction, count)
            predictions: list[dict[str, object]] = []
            failures = 0
            for sample in eligible:
                if not all(canonical_smiles(value) in vapor_pressure for value in sample.smiles):
                    unavailable["missing_psat"] += 1
                    continue
                coverage_check = getattr(predictor, "require_coverage", None)
                if coverage_check is not None:
                    try:
                        coverage_check(sample.smiles)
                    except SPTParameterFileUnavailable:
                        unavailable["missing_parameter_file"] += 1
                        continue
                    except SPTParameterRowUnavailable:
                        unavailable["missing_parameter_row"] += 1
                        continue
                try:
                    adapter = ExternalActivityThermodynamicAdapter(predictor, [sample.smiles]).to(device)
                    molecules = psat_coefficient_tensor(
                        [sample.smiles], vapor_pressure, canonical_smiles, device
                    )
                    x = torch.tensor([sample.liquid_composition], dtype=torch.float32, device=device)
                    mask = torch.ones_like(x)
                    # TeNNet-SAC differentiates its segment free-energy network
                    # internally to obtain activity coefficients, so inference must
                    # retain autograd even though no external weights are optimized.
                    with torch.enable_grad():
                        if direction == "isothermal":
                            state = solve_isothermal(
                                adapter,
                                molecules,
                                torch.tensor([[sample.temperature_k]], dtype=torch.float32, device=device),
                                x,
                                mask,
                                strict=False,
                            )
                        else:
                            state = solve_isobaric(
                                adapter,
                                molecules,
                                torch.tensor([[sample.pressure_kpa]], dtype=torch.float32, device=device),
                                x,
                                mask,
                                strict=False,
                            )
                except (RuntimeError, ValueError):
                    failures += 1
                    unavailable["solver_failure"] += 1
                    continue
                if not bool(state.converged.detach().cpu().reshape(-1)[0]):
                    failures += 1
                    unavailable["solver_failure"] += 1
                    continue
                predicted_pressure = float(state.pressure_kpa.detach().cpu().reshape(-1)[0])
                predicted_temperature = float(state.temperature_k.detach().cpu().reshape(-1)[0])
                vapor = state.y.detach().cpu().numpy()[0]
                state_value = predicted_pressure if direction == "isothermal" else predicted_temperature
                nonphysical = bool(
                    not math.isfinite(state_value)
                    or state_value <= 0.0
                    or not np.isfinite(vapor).all()
                    or np.any(vapor < 0.0)
                    or np.any(vapor > 1.0)
                )
                predictions.append({
                    "sample_id": sample_id(sample),
                    "system_id": system_id(sample),
                    "baseline": baseline,
                    "benchmark": benchmark,
                    "direction": direction,
                    "component_count": count,
                    "observed_pressure": sample.pressure_kpa,
                    "predicted_pressure": predicted_pressure if direction == "isothermal" else "",
                    "observed_temperature": sample.temperature_k,
                    "predicted_temperature": predicted_temperature if direction == "isobaric" else "",
                    "observed_y": list(sample.vapor_composition),
                    "predicted_y": vapor.tolist(),
                    "nonphysical": nonphysical,
                })
            status = "evaluated" if predictions else "not_evaluated_no_external_or_psat_coverage"
            rows.append(
                _metric_row(
                    baseline, benchmark, direction, count, len(eligible), predictions, status, failures
                )
            )
            all_predictions.extend(predictions)
            combined[direction]["total"] += len(eligible)
            combined[direction]["failures"] += failures
            combined[direction]["predictions"].extend(predictions)
        values = combined[direction]
        direction_predictions = values["predictions"]
        rows.append(
            _metric_row(
                baseline,
                benchmark,
                direction,
                "2+3",
                int(values["total"]),
                direction_predictions,
                "evaluated" if direction_predictions else "not_evaluated_no_external_or_psat_coverage",
                int(values["failures"]),
            )
        )
    checkpoint = {
        "model": baseline,
        "training_regime": "fixed_external_model_no_test_training_or_tuning",
        "training_system_overlap": "unknown_external_training_inventory",
        "vapor_pressure_audit": vapor_audit,
        "coverage_audit": unavailable,
        "model_assets": model_assets,
        "parameter_accounting": {"optimized": 0},
    }
    return DirectionResult(tuple(rows), tuple(all_predictions), checkpoint, (), 0)


def _adapted_spt_result(
    train: Sequence[VLESample],
    validation: Sequence[VLESample],
    test: Sequence[VLESample],
    benchmark: str,
    config: FormalConfig,
) -> DirectionResult:
    """Fit train-only NRTL labels, select a pair Transformer on validation, then test."""
    adapted = config.adapted_spt()
    adapted.validate()
    vapor_pressure, vapor_audit = fit_vapor_pressure_correlations(train)
    training_labels, training_audit = fit_partition_pair_labels(train, vapor_pressure, adapted)
    validation_labels, validation_audit = fit_partition_pair_labels(
        validation, vapor_pressure, adapted
    )
    trained = train_adapted_spt(
        training_labels,
        validation_labels,
        adapted,
        torch.device(config.device),
    )
    predictor = AdaptedSPTNRTLPredictor(trained, torch.device(config.device))
    evaluated = _fixed_external_activity_result(
        "spt_nrtl_adapted",
        predictor,
        train,
        test,
        benchmark,
        config,
        {},
    )
    checkpoint = {
        "model": "spt_nrtl_adapted",
        "display_name": "SPT-NRTL adapted (ThermoFormer-train)",
        "training_regime": (
            "binary_train_pair_parameter_fit_then_character_transformer; "
            "validation_parameter_labels_for_checkpoint_selection; no_test_fit"
        ),
        "training_system_overlap": "registered_training_partition_only",
        "test_labels_used_for_training_or_selection": False,
        "vapor_pressure_audit": vapor_audit,
        "training_pair_label_audit": training_audit,
        "validation_pair_label_audit": validation_audit,
        "model_state": trained.model_state,
        "vocabulary": trained.vocabulary,
        "parameter_mean": trained.parameter_mean,
        "parameter_scale": trained.parameter_scale,
        "adapted_config": trained.config,
        "best_validation_loss": trained.best_validation_loss,
        "best_epoch": trained.best_epoch,
        "parameter_accounting": {
            "optimized": trained.trainable_parameters,
            "pair_label_parameters": 10,
        },
        "adaptation_disclosure": (
            "Paper architecture reimplemented without author weights or confidential COSMO "
            "pretraining; fitted NRTL labels use ThermoFormer train/validation partitions only."
        ),
    }
    return DirectionResult(
        evaluated.rows,
        evaluated.predictions,
        checkpoint,
        trained.history,
        trained.trainable_parameters,
    )


def _blocked_rows(baseline: str, benchmark: str, test: Sequence[VLESample], reason: str) -> tuple[dict[str, object], ...]:
    capability = BASELINE_CAPABILITIES[baseline]
    rows = []
    for count in benchmark_for_baseline(baseline).test_component_counts:
        for direction in ("isothermal", "isobaric"):
            eligible = _eligible(test, direction, count)
            status = "blocked_" + reason if direction in capability.native_directions else "not_applicable"
            rows.append(_metric_row(baseline, benchmark, direction, count, len(eligible), (), status))
    if len(benchmark_for_baseline(baseline).test_component_counts) > 1:
        for direction in ("isothermal", "isobaric"):
            total = sum(
                len(_eligible(test, direction, count))
                for count in benchmark_for_baseline(baseline).test_component_counts
            )
            status = "blocked_" + reason if direction in capability.native_directions else "not_applicable"
            rows.append(_metric_row(baseline, benchmark, direction, "2+3", total, (), status))
    return tuple(rows)


def run_formal_seed(
    project_root: Path,
    baseline: str,
    seed: int,
    output_dir: Path,
    checkpoint_dir: Path,
    config: FormalConfig,
    settings_sha256: str,
    source_commit: str | None = None,
) -> dict[str, object]:
    if seed not in (0, 1, 2, 3, 4):
        raise ValueError("Formal baseline seeds are frozen to 0--4")
    capability = BASELINE_CAPABILITIES[baseline]
    benchmark = benchmark_for_baseline(baseline)
    _set_seed(seed)
    samples = load_registered_vle_dataset(project_root)
    split_path = project_root / 'datasets/splits/vle' / benchmark.split_protocol / f"seed_{seed}.json"
    split = load_split_assignment(split_path, samples)
    train = tuple(row for row in split.train if row.component_count in benchmark.train_component_counts)
    validation = tuple(row for row in split.validation if row.component_count in benchmark.test_component_counts)
    test = tuple(row for row in split.test if row.component_count in benchmark.test_component_counts)
    test_ids = {sample_id(row) for row in test}
    if test_ids & {sample_id(row) for row in (*train, *validation)}:
        raise RuntimeError("Formal baseline partitions overlap")
    if baseline not in EXECUTABLE_BASELINES:
        reason = "external_assets" if capability.external_assets_required else "unavailable_author_features"
        rows = _blocked_rows(baseline, benchmark.key, test, reason)
        result = DirectionResult(rows, (), None, (), 0)
    elif baseline == "hanna":
        result = _hanna_result(project_root, train, test, benchmark.key, config)
    elif baseline == "tennet_sac":
        result = _fixed_external_activity_result(
            baseline,
            CudaTeNNetSACPredictor(torch.device(config.device))
            if config.device.startswith("cuda")
            else TeNNetSACPredictor(),
            train,
            test,
            benchmark.key,
            config,
            {
                "package": f"tennetsac=={TENNETSAC_PACKAGE_VERSION}",
                "wheel_sha256": TENNETSAC_WHEEL_SHA256,
                "verified_wheel": verified_tennetsac_wheel(
                    project_root / 'experiments/run_records/cache/tennetsac'
                ),
                "source_revision": TENNETSAC_SOURCE_REVISION,
                "model_version": "tuned_mean_of_10_experimental_finetuned_heads",
                "training_system_overlap": "unknown_official_training_inventory_not_fully_published",
                "installed_assets": installed_tennetsac_asset_audit(),
            },
        )
    elif baseline == "spt_nrtl":
        database = SPTNRTLDatabase(project_root / 'experiments/run_records/cache/spt_nrtl')
        result = _fixed_external_activity_result(
            baseline,
            SPTNRTLPredictor(database),
            train,
            test,
            benchmark.key,
            config,
            {
                "database_revision": SPT_NRTL_DATABASE_REVISION,
                "source_revision": SPT_NRTL_SOURCE_REVISION,
                "database_download_attempts": database.download_attempts,
                "parameter_policy": "exact_canonical_smiles_lookup_no_test_fit_or_imputation",
                "database_query_audit": database.audit,
                "database_pair_audit": database.pair_audit,
                "database_valid_temperature_range": "not_published",
                "database_license": "not_declared_at_fixed_revision_publication_use_requires_review",
                "training_system_overlap": "unknown_database_model_training_inventory",
            },
        )
    elif baseline == "spt_nrtl_adapted":
        result = _adapted_spt_result(train, validation, test, benchmark.key, config)
    elif baseline in {"smiles_rnn", "ualf_gnn"}:
        directions = capability.native_directions
        pieces = [_direct_result(baseline, direction, train, validation, test, benchmark.key, config) for direction in directions]
        result = DirectionResult(
            tuple(row for piece in pieces for row in piece.rows),
            tuple(row for piece in pieces for row in piece.predictions),
            {piece.rows[0]["direction"]: piece.checkpoint for piece in pieces if piece.checkpoint},
            tuple(row for piece in pieces for row in piece.history),
            sum(piece.trainable_parameters for piece in pieces),
        )
    else:
        result = _graph_result(baseline, train, validation, test, benchmark.key, config)
        isobaric_rows = tuple(
            _metric_row(baseline, benchmark.key, "isobaric", count, len(_eligible(test, "isobaric", count)), (), "not_applicable")
            for count in benchmark.test_component_counts
        )
        if len(benchmark.test_component_counts) > 1:
            isobaric_rows += (
                _metric_row(
                    baseline, benchmark.key, "isobaric", "2+3",
                    sum(len(_eligible(test, "isobaric", count)) for count in benchmark.test_component_counts),
                    (), "not_applicable",
                ),
            )
        result = DirectionResult(result.rows + isobaric_rows, result.predictions, result.checkpoint, result.history, result.trainable_parameters)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, predictions_path, history_path = output_dir / "metrics.csv", output_dir / "predictions.csv", output_dir / "history.csv"
    write_seed_metrics(metrics_path, result.rows, seed)
    _atomic_csv(predictions_path, result.predictions or ({"status": "no_predictions"},))
    _atomic_csv(history_path, result.history or ({"status": "not_trained"},))
    checkpoint_path = checkpoint_dir / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if result.checkpoint is not None:
        torch.save({
            **result.checkpoint,
            "seed": seed,
            "benchmark": benchmark.key,
            "selection_partition": _selection_partition(baseline),
        }, checkpoint_path)
    if source_commit is None:
        try:
            source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = "unavailable"
    artifacts = {
        "metrics": {"path": "metrics.csv", "sha256": artifact_sha256(metrics_path)},
        "predictions": {"path": "predictions.csv", "sha256": artifact_sha256(predictions_path)},
        "history": {"path": "history.csv", "sha256": artifact_sha256(history_path)},
    }
    if checkpoint_path.is_file():
        artifacts["checkpoint"] = {"path": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"), "sha256": artifact_sha256(checkpoint_path)}
    source_workbooks = {
        str(path.relative_to(project_root)).replace("\\", "/"): artifact_sha256(path)
        for path in discover_vle_workbooks(project_root / "datasets" / "vle", "_vle_")
    }
    feature_definitions = {
        str(path.relative_to(project_root)).replace("\\", "/"): artifact_sha256(path)
        for path in (
            Path(__file__).with_name("models.py"),
            Path(__file__).with_name("formal.py"),
            Path(__file__).with_name("smoke.py"),
            *([Path(__file__).with_name("hanna.py")] if baseline == "hanna" else []),
            *(
                [
                    Path(__file__).with_name("external_activity.py"),
                    Path(__file__).with_name("tennet_sac.py"),
                ]
                if baseline == "tennet_sac" else []
            ),
            *(
                [
                    Path(__file__).with_name("external_activity.py"),
                    Path(__file__).with_name("spt_nrtl.py"),
                ]
                if baseline == "spt_nrtl" else []
            ),
            *(
                [
                    Path(__file__).with_name("external_activity.py"),
                    Path(__file__).with_name("spt_nrtl.py"),
                    Path(__file__).with_name("spt_nrtl_adapted.py"),
                ]
                if baseline == "spt_nrtl_adapted" else []
            ),
        )
    }
    environment_path = project_root / "environment.yml"
    requirements_path = project_root / "requirements.txt"
    manifest_status = (
        "completed"
        if result.checkpoint is not None and (
            baseline not in {"tennet_sac", "spt_nrtl", "spt_nrtl_adapted"} or bool(result.predictions)
        )
        else "coverage_audit"
        if baseline in {"tennet_sac", "spt_nrtl"}
        else "not_evaluated"
    )
    model_assets = (
        result.checkpoint.get("model_assets", {})
        if baseline in {"tennet_sac", "spt_nrtl"} and result.checkpoint is not None
        else {
            "source_revision": HANNA_SOURCE_REVISION,
            "files": OfficialHANNAAssets.default(project_root).hashes(project_root),
            "training_system_overlap": "unknown_official_training_inventory_not_published",
        }
        if baseline == "hanna" else {}
    )
    manifest = {
        "schema_version": FORMAL_SCHEMA_VERSION, "status": manifest_status,
        "analysis_status": _analysis_status(baseline, manifest_status == "completed"),
        "baseline": baseline, "benchmark": benchmark.key, "split_protocol": benchmark.split_protocol,
        "seed": seed, "selection_partition": _selection_partition(baseline), "evaluation_partition": "test",
        "test_labels_used_for_selection": False, "git_commit": source_commit,
        "dataset_sha256": dataset_digest(samples), "split_sha256": artifact_sha256(split_path),
        "settings_sha256": settings_sha256, "config": asdict(config),
        "source_workbooks": source_workbooks,
        "feature_definitions": feature_definitions,
        "feature_cache": {"used": False, "policy": "deterministic_on_the_fly_from_registered_rows"},
        "model_assets": model_assets,
        "environment": {
            "path": str(environment_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": artifact_sha256(environment_path),
            "requirements_path": str(requirements_path.relative_to(project_root)).replace("\\", "/"),
            "requirements_sha256": artifact_sha256(requirements_path),
        },
        "train_sample_ids_sha256": hashlib.sha256("\n".join(map(sample_id, train)).encode()).hexdigest(),
        "validation_sample_ids_sha256": hashlib.sha256("\n".join(map(sample_id, validation)).encode()).hexdigest(),
        "test_sample_ids_sha256": hashlib.sha256("\n".join(map(sample_id, test)).encode()).hexdigest(),
        "train_system_ids_sha256": hashlib.sha256("\n".join(sorted({system_id(row) for row in train})).encode()).hexdigest(),
        "validation_system_ids_sha256": hashlib.sha256("\n".join(sorted({system_id(row) for row in validation})).encode()).hexdigest(),
        "test_system_ids_sha256": hashlib.sha256("\n".join(sorted({system_id(row) for row in test})).encode()).hexdigest(),
        "trainable_parameters": result.trainable_parameters, "artifacts": artifacts,
        "parameter_accounting": (
            result.checkpoint.get("parameter_accounting", {})
            if result.checkpoint is not None else {}
        ),
        "runtime": _runtime_provenance(baseline, config.device),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def aggregate_formal_baseline(project_root: Path, baseline: str, result_root: Path) -> list[dict[str, object]]:
    benchmark = benchmark_for_baseline(baseline)
    experiment_root = result_root / f"{baseline}.on.{benchmark.key}"
    seed_files: list[Path] = []
    input_manifests: dict[str, dict[str, str]] = {}
    invariant: dict[str, object] | None = None
    for seed in range(5):
        seed_root = experiment_root / f"seed_{seed}"
        manifest_path = seed_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_status = manifest.get("status")
        allowed_statuses = (
            {"completed", "not_evaluated", "coverage_audit"}
            if baseline in EXECUTABLE_BASELINES
            else {"not_evaluated"}
        )
        if actual_status not in allowed_statuses:
            raise ValueError(f"Seed {seed} manifest has stale status")
        if actual_status == "not_evaluated":
            metric_rows = list(csv.DictReader((seed_root / "metrics.csv").open(encoding="utf-8")))
            if any(row.get("status") == "evaluated" for row in metric_rows):
                raise ValueError(f"Seed {seed} is marked not evaluated but contains evaluated metrics")
        expected = {
            "analysis_status": _analysis_status(baseline, actual_status == "completed"),
            "baseline": baseline, "benchmark": benchmark.key, "seed": seed,
            "selection_partition": _selection_partition(baseline), "evaluation_partition": "test",
            "test_labels_used_for_selection": False,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"Seed {seed} manifest has stale {field}")
        split_path = project_root / 'datasets/splits/vle' / benchmark.split_protocol / f"seed_{seed}.json"
        if artifact_sha256(split_path) != manifest.get("split_sha256"):
            raise ValueError(f"Seed {seed} split assignment has changed")
        for path_text, digest in manifest.get("source_workbooks", {}).items():
            path = project_root / path_text
            if not path.is_file() or artifact_sha256(path) != digest:
                raise ValueError(f"Seed {seed} source workbook has changed: {path_text}")
        for path_text, digest in manifest.get("feature_definitions", {}).items():
            path = project_root / path_text
            if not path.is_file() or artifact_sha256(path) != digest:
                raise ValueError(f"Seed {seed} feature definition has changed: {path_text}")
        if baseline == "hanna" and manifest.get("model_assets", {}).get("files") != OfficialHANNAAssets.default(project_root).hashes(project_root):
            raise ValueError(f"Seed {seed} official HANNA assets have changed")
        if baseline == "spt_nrtl":
            assets = manifest.get("model_assets", {})
            if (
                assets.get("database_revision") != SPT_NRTL_DATABASE_REVISION
                or assets.get("database_download_attempts") != 4
                or assets.get("database_license")
                != "not_declared_at_fixed_revision_publication_use_requires_review"
            ):
                raise ValueError(f"Seed {seed} SPT-NRTL database revision has changed")
            for query in assets.get("database_query_audit", {}).values():
                path = project_root / 'experiments/run_records/cache/spt_nrtl' / query["cache_relative_path"]
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != query["sha256"]
                ):
                    raise ValueError(f"Seed {seed} SPT-NRTL cached parameter source has changed")
            source_hashes = {
                query["sha256"] for query in assets.get("database_query_audit", {}).values()
            }
            for pair in assets.get("database_pair_audit", {}).values():
                if (
                    pair.get("lookup_direction") not in {"forward", "reverse"}
                    or pair.get("source_sha256") not in source_hashes
                    or len(str(pair.get("selected_row_sha256", ""))) != 64
                ):
                    raise ValueError(f"Seed {seed} SPT-NRTL selected pair provenance is invalid")
        if baseline == "tennet_sac":
            assets = manifest.get("model_assets", {})
            if (
                assets.get("source_revision") != TENNETSAC_SOURCE_REVISION
                or assets.get("wheel_sha256") != TENNETSAC_WHEEL_SHA256
                or assets.get("verified_wheel") != verified_tennetsac_wheel(
                    project_root / 'experiments/run_records/cache/tennetsac'
                )
                or assets.get("installed_assets") != installed_tennetsac_asset_audit()
            ):
                raise ValueError(f"Seed {seed} TeNNet-SAC asset identity has changed")
        environment = manifest.get("environment", {})
        environment_path = project_root / str(environment.get("path", ""))
        if not environment_path.is_file() or artifact_sha256(environment_path) != environment.get("sha256"):
            raise ValueError(f"Seed {seed} runtime environment definition has changed")
        requirements_path = project_root / str(environment.get("requirements_path", ""))
        if not requirements_path.is_file() or artifact_sha256(requirements_path) != environment.get("requirements_sha256"):
            raise ValueError(f"Seed {seed} runtime requirements have changed")
        comparable_assets = copy.deepcopy(manifest.get("model_assets", {}))
        if baseline == "spt_nrtl":
            comparable_assets.pop("database_query_audit", None)
            comparable_assets.pop("database_pair_audit", None)
        current = {
            field: manifest[field]
            for field in (
                "dataset_sha256", "settings_sha256", "git_commit", "split_protocol",
                "config", "runtime", "source_workbooks", "feature_definitions",
                "feature_cache", "environment", "parameter_accounting",
            )
        }
        current["model_assets"] = comparable_assets
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise ValueError("Formal seed manifests do not share comparable provenance")
        for name, artifact in manifest["artifacts"].items():
            path = (
                project_root / artifact["path"]
                if name == "checkpoint"
                else seed_root / artifact["path"]
            )
            if not path.is_file() or artifact_sha256(path) != artifact["sha256"]:
                raise ValueError(f"Seed {seed} artifact failed verification: {name}")
        seed_files.append(seed_root / "metrics.csv")
        input_manifests[str(seed)] = {
            "path": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": artifact_sha256(manifest_path),
        }
    output = experiment_root / "metrics_summary.csv"
    rows = aggregate_seed_metrics(seed_files, output)
    _atomic_json(
        experiment_root / "aggregate_manifest.json",
        {
            "schema_version": FORMAL_SCHEMA_VERSION,
            "status": "completed" if baseline in EXECUTABLE_BASELINES else "not_evaluated",
            "analysis_status": _analysis_status(baseline, baseline in EXECUTABLE_BASELINES),
            "baseline": baseline,
            "benchmark": benchmark.key,
            "seeds": list(range(5)),
            "provenance": invariant,
            "input_manifests": input_manifests,
            "output": {"path": "metrics_summary.csv", "sha256": artifact_sha256(output)},
        },
    )
    return rows

