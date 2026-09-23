"""Row-level VLE inference and paper-grade point/system metric aggregation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import VLESample, VLETensorDataset, collate_vle
from ..thermodynamics.vapor_pressure import PurePropertyCatalog
from ..data.splitting import canonical_smiles, component_id, sample_id, system_id
from ..models import DirectVLEOutputs
from ..thermodynamics import EquilibriumState, solve_batch_modes


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if actual.size < 2:
        return None
    variance = float(np.sum((actual - actual.mean()) ** 2))
    if variance <= 1e-15:
        return None
    return float(1.0 - np.sum((actual - predicted) ** 2) / variance)


def _scalar_metrics(
    rows: Sequence[dict[str, Any]],
    actual_name: str,
    predicted_name: str,
    prefix: str,
    unit_suffix: str,
) -> dict[str, float | int | None]:
    selected = [
        row
        for row in rows
        if row.get(actual_name) is not None
        and row.get(predicted_name) is not None
        and math.isfinite(float(row[actual_name]))
        and math.isfinite(float(row[predicted_name]))
    ]
    if not selected:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mae{unit_suffix}": None,
            f"{prefix}_rmse{unit_suffix}": None,
            f"{prefix}_r2": None,
            f"{prefix}_system_macro_mae{unit_suffix}": None,
            f"{prefix}_system_macro_rmse{unit_suffix}": None,
        }
    actual = np.asarray([float(row[actual_name]) for row in selected])
    predicted = np.asarray([float(row[predicted_name]) for row in selected])
    errors = predicted - actual
    by_system: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(selected, errors):
        by_system[str(row["system_id"])].append(float(error))
    system_mae = [np.mean(np.abs(values)) for values in by_system.values()]
    system_mse = [np.mean(np.square(values)) for values in by_system.values()]
    return {
        f"{prefix}_count": len(selected),
        f"{prefix}_mae{unit_suffix}": float(np.mean(np.abs(errors))),
        f"{prefix}_rmse{unit_suffix}": float(np.sqrt(np.mean(errors**2))),
        f"{prefix}_r2": _r2(actual, predicted),
        f"{prefix}_system_macro_mae{unit_suffix}": float(np.mean(system_mae)),
        f"{prefix}_system_macro_rmse{unit_suffix}": float(np.sqrt(np.mean(system_mse))),
    }


def _y_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    observations: list[tuple[str, str, str, float, float]] = []
    sample_errors: dict[str, list[float]] = defaultdict(list)
    system_errors: dict[str, list[float]] = defaultdict(list)
    component_errors: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for index in range(1, int(row["component_count"]) + 1):
            actual = row.get(f"y_true_{index}")
            predicted = row.get(f"y_pred_{index}")
            component = row.get(f"component_id_{index}")
            if actual is None or predicted is None or component is None:
                continue
            actual_value = float(actual)
            predicted_value = float(predicted)
            if not math.isfinite(actual_value) or not math.isfinite(predicted_value):
                continue
            error = predicted_value - actual_value
            observations.append(
                (
                    str(row["sample_id"]),
                    str(row["system_id"]),
                    str(component),
                    actual_value,
                    predicted_value,
                )
            )
            sample_errors[str(row["sample_id"])].append(error)
            system_errors[str(row["system_id"])].append(error)
            component_errors[str(component)].append(error)
    if not observations:
        return {
            "y_component_values": 0,
            "y_mae": None,
            "y_rmse": None,
            "y_r2": None,
            "y_sample_macro_mae": None,
            "y_system_macro_mae": None,
            "y_system_macro_rmse": None,
            "y_component_macro_mae": None,
        }
    actual = np.asarray([row[3] for row in observations])
    predicted = np.asarray([row[4] for row in observations])
    errors = predicted - actual
    return {
        "y_component_values": len(observations),
        "y_mae": float(np.mean(np.abs(errors))),
        "y_rmse": float(np.sqrt(np.mean(errors**2))),
        "y_r2": _r2(actual, predicted),
        "y_sample_macro_mae": float(
            np.mean([np.mean(np.abs(values)) for values in sample_errors.values()])
        ),
        "y_system_macro_mae": float(
            np.mean([np.mean(np.abs(values)) for values in system_errors.values()])
        ),
        "y_system_macro_rmse": float(
            np.sqrt(np.mean([np.mean(np.square(values)) for values in system_errors.values()]))
        ),
        "y_component_macro_mae": float(
            np.mean([np.mean(np.abs(values)) for values in component_errors.values()])
        ),
    }


def _metrics_for_group(
    rows: Sequence[dict[str, Any]],
    scope: str,
    direction: str | None,
    component_count: int | None,
    subgroup: str | None = None,
) -> dict[str, Any]:
    attempted = len(rows)
    converged = sum(bool(row.get("converged")) for row in rows)
    nonphysical = sum(bool(row.get("nonphysical")) for row in rows)
    valid = [
        row
        for row in rows
        if bool(row.get("converged")) and not bool(row.get("nonphysical"))
    ]
    result: dict[str, Any] = {
        "scope": scope,
        "direction": direction,
        "component_count": component_count,
        "subgroup": subgroup,
        "attempted_samples": attempted,
        "valid_samples": len(valid),
        "valid_coverage": len(valid) / attempted if attempted else 0.0,
        "solver_failure_rate": (attempted - converged) / attempted if attempted else 0.0,
        "nonphysical_rate": nonphysical / attempted if attempted else 0.0,
        "systems": len({row["system_id"] for row in rows}),
    }
    result.update(
        _scalar_metrics(
            valid,
            "target_pressure_kpa",
            "predicted_pressure_kpa",
            "pressure",
            "_kpa",
        )
    )
    result.update(
        _scalar_metrics(
            valid,
            "target_temperature_k",
            "predicted_temperature_k",
            "temperature",
            "_k",
        )
    )
    result.update(_y_metrics(valid))
    return result


def prediction_metric_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate inference records without hiding solver/nonphysical failures."""
    if not records:
        raise ValueError("At least one prediction record is required")
    rows = list(records)
    result = [_metrics_for_group(rows, "all", None, None)]
    directions = sorted({str(row["direction"]) for row in rows})
    cardinalities = sorted({int(row["component_count"]) for row in rows})
    for direction in directions:
        selected = [row for row in rows if row["direction"] == direction]
        result.append(_metrics_for_group(selected, "direction", direction, None))
    for cardinality in cardinalities:
        selected = [row for row in rows if int(row["component_count"]) == cardinality]
        result.append(_metrics_for_group(selected, "cardinality", None, cardinality))
    for direction in directions:
        for cardinality in cardinalities:
            selected = [
                row
                for row in rows
                if row["direction"] == direction
                and int(row["component_count"]) == cardinality
            ]
            if selected:
                result.append(
                    _metrics_for_group(
                        selected,
                        "direction_cardinality",
                        direction,
                        cardinality,
                    )
                )
    coverage_values = sorted(
        {
            int(row["binary_subsystem_coverage"])
            for row in rows
            if row.get("binary_subsystem_coverage") is not None
        }
    )
    for coverage in coverage_values:
        selected = [
            row for row in rows if row.get("binary_subsystem_coverage") == coverage
        ]
        result.append(
            _metrics_for_group(
                selected,
                "binary_subsystem_coverage",
                None,
                None,
                f"{coverage}/3",
            )
        )
    unseen_rows = [row for row in rows if row.get("strict_unseen") is not None]
    if unseen_rows:
        result.append(
            _metrics_for_group(
                unseen_rows,
                "unseen_subset",
                None,
                None,
                "at_least_one_component_unseen",
            )
        )
        strict_rows = [row for row in unseen_rows if bool(row["strict_unseen"])]
        if strict_rows:
            result.append(
                _metrics_for_group(
                    strict_rows,
                    "unseen_subset",
                    None,
                    None,
                    "strict_all_components_unseen",
                )
            )
    return result


def _nonphysical_state(state: EquilibriumState, index: int, count: int) -> bool:
    values = torch.cat(
        [
            state.temperature_k[index].reshape(-1),
            state.pressure_kpa[index].reshape(-1),
            state.y[index, :count],
            state.gamma[index, :count],
            state.psat_kpa[index, :count],
        ]
    ).detach().cpu()
    if not bool(torch.isfinite(values).all()):
        return True
    temperature = float(state.temperature_k[index].reshape(-1)[0].detach().cpu())
    pressure = float(state.pressure_kpa[index].reshape(-1)[0].detach().cpu())
    y = state.y[index, :count].detach().cpu()
    gamma = state.gamma[index, :count].detach().cpu()
    psat = state.psat_kpa[index, :count].detach().cpu()
    return (
        not 150.0 <= temperature <= 1500.0
        or pressure <= 0.0
        or bool((y < -1e-6).any())
        or bool((y > 1.0 + 1e-6).any())
        or abs(float(y.sum()) - 1.0) > 1e-4
        or bool((gamma <= 0.0).any())
        or bool((psat <= 0.0).any())
    )


def _prediction_record(
    sample: VLESample,
    state: EquilibriumState,
    state_index: int,
    direction: str,
) -> dict[str, Any]:
    order = sorted(
        range(sample.component_count),
        key=lambda index: canonical_smiles(sample.smiles[index]),
    )
    converged = (
        bool(state.converged[state_index].reshape(-1)[0].detach().cpu())
        if state.converged is not None
        else False
    )
    record: dict[str, Any] = {
        "sample_id": sample_id(sample),
        "system_id": system_id(sample),
        "component_count": sample.component_count,
        "source": Path(sample.source).name,
        "doi": sample.doi,
        "quality_status": sample.quality_status,
        "quality_weight": sample.quality_weight,
        "experiment_mode": sample.experiment_mode,
        "experiment_mode_confidence": sample.experiment_mode_confidence,
        "direction": direction,
        "target_temperature_k": sample.temperature_k if direction == "isobaric" else None,
        "predicted_temperature_k": (
            float(state.temperature_k[state_index].reshape(-1)[0].detach().cpu())
            if direction == "isobaric"
            else None
        ),
        "target_pressure_kpa": sample.pressure_kpa if direction == "isothermal" else None,
        "predicted_pressure_kpa": (
            float(state.pressure_kpa[state_index].reshape(-1)[0].detach().cpu())
            if direction == "isothermal"
            else None
        ),
        "pressure_residual_kpa": float(
            state.pressure_residual_kpa[state_index].reshape(-1)[0].detach().cpu()
        ),
        "converged": converged,
        "nonphysical": _nonphysical_state(state, state_index, sample.component_count),
        "iterations": state.iterations,
    }
    for output_index in range(3):
        suffix = output_index + 1
        if output_index < sample.component_count:
            source_index = order[output_index]
            canonical = canonical_smiles(sample.smiles[source_index])
            record.update(
                {
                    f"component_smiles_{suffix}": canonical,
                    f"component_id_{suffix}": component_id(canonical),
                    f"x_{suffix}": sample.liquid_composition[source_index],
                    f"y_true_{suffix}": sample.vapor_composition[source_index],
                    f"y_pred_{suffix}": float(
                        state.y[state_index, source_index].detach().cpu()
                    ),
                    f"gamma_pred_{suffix}": float(
                        state.gamma[state_index, source_index].detach().cpu()
                    ),
                    f"psat_pred_kpa_{suffix}": float(
                        state.psat_kpa[state_index, source_index].detach().cpu()
                    ),
                }
            )
        else:
            for name in (
                "component_smiles",
                "component_id",
                "x",
                "y_true",
                "y_pred",
                "gamma_pred",
                "psat_pred_kpa",
            ):
                record[f"{name}_{suffix}"] = None
    return record


def _direct_prediction_record(
    sample: VLESample,
    prediction: DirectVLEOutputs,
    state_index: int,
    direction: str,
) -> dict[str, Any]:
    order = sorted(
        range(sample.component_count),
        key=lambda index: canonical_smiles(sample.smiles[index]),
    )
    temperature = float(prediction.temperature_k[state_index, 0].detach().cpu())
    pressure = float(prediction.pressure_kpa[state_index, 0].detach().cpu())
    y = prediction.y[state_index, : sample.component_count].detach().cpu()
    nonphysical = (
        not math.isfinite(temperature)
        or not math.isfinite(pressure)
        or pressure <= 0.0
        or not bool(torch.isfinite(y).all())
        or bool((y < -1e-6).any())
        or bool((y > 1.0 + 1e-6).any())
        or abs(float(y.sum()) - 1.0) > 1e-4
    )
    record: dict[str, Any] = {
        "sample_id": sample_id(sample),
        "system_id": system_id(sample),
        "component_count": sample.component_count,
        "source": Path(sample.source).name,
        "doi": sample.doi,
        "quality_status": sample.quality_status,
        "quality_weight": sample.quality_weight,
        "experiment_mode": sample.experiment_mode,
        "experiment_mode_confidence": sample.experiment_mode_confidence,
        "direction": direction,
        "target_temperature_k": sample.temperature_k if direction == "isobaric" else None,
        "predicted_temperature_k": temperature if direction == "isobaric" else None,
        "target_pressure_kpa": sample.pressure_kpa if direction == "isothermal" else None,
        "predicted_pressure_kpa": pressure if direction == "isothermal" else None,
        "pressure_residual_kpa": None,
        "converged": True,
        "nonphysical": nonphysical,
        "iterations": 0,
    }
    for output_index in range(3):
        suffix = output_index + 1
        if output_index < sample.component_count:
            source_index = order[output_index]
            canonical = canonical_smiles(sample.smiles[source_index])
            record.update(
                {
                    f"component_smiles_{suffix}": canonical,
                    f"component_id_{suffix}": component_id(canonical),
                    f"x_{suffix}": sample.liquid_composition[source_index],
                    f"y_true_{suffix}": sample.vapor_composition[source_index],
                    f"y_pred_{suffix}": float(prediction.y[state_index, source_index].detach().cpu()),
                    f"gamma_pred_{suffix}": None,
                    f"psat_pred_kpa_{suffix}": None,
                }
            )
        else:
            for name in (
                "component_smiles",
                "component_id",
                "x",
                "y_true",
                "y_pred",
                "gamma_pred",
                "psat_pred_kpa",
            ):
                record[f"{name}_{suffix}"] = None
    return record


def predict_vle(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    solver_iterations: int = 24,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> list[dict[str, Any]]:
    """Run the label-independent solver entry points and retain every attempt."""
    if not samples:
        raise ValueError("Prediction set is empty")
    dataset = VLETensorDataset(
        samples,
        feature_map,
        pure_property_catalog=pure_property_catalog,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_vle)
    model.to(device).eval()
    records: list[dict[str, Any]] = []
    offset = 0
    direct = getattr(getattr(model, "config", None), "decoder_mode", None) == "direct_vle"
    with torch.no_grad():
        for host_batch in loader:
            batch_samples = samples[offset : offset + host_batch.x.shape[0]]
            offset += host_batch.x.shape[0]
            batch = host_batch.to(device)
            if direct:
                isothermal_rows = batch.experiment_mode != 1
                if bool(isothermal_rows.any()):
                    prediction = model.predict_direct(
                        batch.molecules[isothermal_rows],
                        batch.temperature_k[isothermal_rows],
                        batch.pressure_kpa[isothermal_rows],
                        batch.x[isothermal_rows],
                        batch.mask[isothermal_rows],
                        direction="isothermal",
                    )
                    local_rows = torch.nonzero(
                        isothermal_rows, as_tuple=False
                    ).flatten().tolist()
                    for state_index, local_index in enumerate(local_rows):
                        records.append(
                            _direct_prediction_record(
                                batch_samples[local_index],
                                prediction,
                                state_index,
                                "isothermal",
                            )
                        )
                isobaric_rows = batch.experiment_mode != 0
                if bool(isobaric_rows.any()):
                    prediction = model.predict_direct(
                        batch.molecules[isobaric_rows],
                        batch.temperature_k[isobaric_rows],
                        batch.pressure_kpa[isobaric_rows],
                        batch.x[isobaric_rows],
                        batch.mask[isobaric_rows],
                        direction="isobaric",
                    )
                    local_rows = torch.nonzero(
                        isobaric_rows, as_tuple=False
                    ).flatten().tolist()
                    for state_index, local_index in enumerate(local_rows):
                        records.append(
                            _direct_prediction_record(
                                batch_samples[local_index],
                                prediction,
                                state_index,
                                "isobaric",
                            )
                        )
                continue
            solutions = solve_batch_modes(
                model,
                batch,
                iterations=solver_iterations,
                strict=False,
            )
            if solutions.isothermal is not None:
                local_rows = torch.nonzero(solutions.isothermal_rows, as_tuple=False).flatten().tolist()
                for state_index, local_index in enumerate(local_rows):
                    records.append(
                        _prediction_record(
                            batch_samples[local_index],
                            solutions.isothermal,
                            state_index,
                            "isothermal",
                        )
                    )
            if solutions.isobaric is not None:
                local_rows = torch.nonzero(solutions.isobaric_rows, as_tuple=False).flatten().tolist()
                for state_index, local_index in enumerate(local_rows):
                    records.append(
                        _prediction_record(
                            batch_samples[local_index],
                            solutions.isobaric,
                            state_index,
                            "isobaric",
                        )
                    )
    return records


def write_prediction_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Cannot write an empty prediction file")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
