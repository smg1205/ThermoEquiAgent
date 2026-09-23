"""Row-level prediction records for fitted thermodynamic activity models."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..data import VLESample
from ..data.splitting import canonical_smiles, component_id, sample_id, system_id
from .thermodynamic_fitting import FittedActivityModel
from .thermodynamic_models import (
    BubbleState,
    LogLinearVaporPressure,
    solve_isobaric_state,
    solve_isothermal_state,
)


def _directions(sample: VLESample) -> tuple[str, ...]:
    if sample.experiment_mode == "isothermal":
        return ("isothermal",)
    if sample.experiment_mode == "isobaric":
        return ("isobaric",)
    return ("isothermal", "isobaric")


def _nonphysical(state: BubbleState) -> bool:
    if not state.converged:
        return False
    values = np.concatenate(
        [
            np.asarray([state.temperature_k, state.pressure_kpa]),
            state.vapor_composition,
            state.activity_coefficients,
            state.vapor_pressures_kpa,
        ]
    )
    return bool(
        not np.isfinite(values).all()
        or not 150.0 <= state.temperature_k <= 1500.0
        or state.pressure_kpa <= 0.0
        or np.any(state.vapor_composition < -1e-6)
        or np.any(state.vapor_composition > 1.0 + 1e-6)
        or abs(float(state.vapor_composition.sum()) - 1.0) > 1e-4
        or np.any(state.activity_coefficients <= 0.0)
        or np.any(state.vapor_pressures_kpa <= 0.0)
    )


def _record(
    sample: VLESample,
    direction: str,
    model_name: str,
    state: BubbleState | None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    order = sorted(
        range(sample.component_count),
        key=lambda index: canonical_smiles(sample.smiles[index]),
    )
    converged = bool(state is not None and state.converged)
    record: dict[str, object] = {
        "sample_id": sample_id(sample),
        "system_id": system_id(sample),
        "component_count": sample.component_count,
        "source": sample.source,
        "doi": sample.doi,
        "quality_status": sample.quality_status,
        "quality_weight": sample.quality_weight,
        "experiment_mode": sample.experiment_mode,
        "experiment_mode_confidence": sample.experiment_mode_confidence,
        "direction": direction,
        "baseline_model": model_name,
        "target_temperature_k": sample.temperature_k if direction == "isobaric" else None,
        "predicted_temperature_k": (
            state.temperature_k if converged and direction == "isobaric" else None
        ),
        "target_pressure_kpa": sample.pressure_kpa if direction == "isothermal" else None,
        "predicted_pressure_kpa": (
            state.pressure_kpa if converged and direction == "isothermal" else None
        ),
        "pressure_residual_kpa": state.residual_kpa if state is not None else None,
        "converged": converged,
        "nonphysical": _nonphysical(state) if state is not None else False,
        "iterations": state.iterations if state is not None else 0,
        "failure_reason": (
            failure_reason
            if failure_reason is not None
            else (state.failure_reason if state is not None else "model_unavailable")
        ),
    }
    if state is not None and converged:
        predicted_by_smiles = {
            value: index for index, value in enumerate(sorted(canonical_smiles(s) for s in sample.smiles))
        }
    else:
        predicted_by_smiles = {}
    for output_index in range(3):
        suffix = output_index + 1
        if output_index < sample.component_count:
            source_index = order[output_index]
            smiles = canonical_smiles(sample.smiles[source_index])
            predicted_index = predicted_by_smiles.get(smiles)
            record.update(
                {
                    f"component_smiles_{suffix}": smiles,
                    f"component_id_{suffix}": component_id(smiles),
                    f"x_{suffix}": sample.liquid_composition[source_index],
                    f"y_true_{suffix}": sample.vapor_composition[source_index],
                    f"y_pred_{suffix}": (
                        float(state.vapor_composition[predicted_index])
                        if state is not None and predicted_index is not None
                        else None
                    ),
                    f"gamma_pred_{suffix}": (
                        float(state.activity_coefficients[predicted_index])
                        if state is not None and predicted_index is not None
                        else None
                    ),
                    f"psat_pred_kpa_{suffix}": (
                        float(state.vapor_pressures_kpa[predicted_index])
                        if state is not None and predicted_index is not None
                        else None
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


def predict_fitted_activity_model(
    samples: Sequence[VLESample],
    fitted: FittedActivityModel,
    vapor_pressure: Mapping[str, LogLinearVaporPressure],
    *,
    temperature_bounds_k: tuple[float, float] = (150.0, 1500.0),
) -> list[dict[str, object]]:
    """Predict all mode-appropriate bubble states for one fitted system."""

    correlations = tuple(vapor_pressure[value] for value in fitted.components)
    records: list[dict[str, object]] = []
    for sample in samples:
        positions = {canonical_smiles(value): index for index, value in enumerate(sample.smiles)}
        x = np.asarray(
            [sample.liquid_composition[positions[value]] for value in fitted.components],
            dtype=float,
        )
        for direction in _directions(sample):
            try:
                if direction == "isothermal":
                    state = solve_isothermal_state(
                        temperature_k=sample.temperature_k,
                        liquid_composition=x,
                        vapor_pressure=correlations,
                        parameters=fitted.parameters,
                    )
                else:
                    state = solve_isobaric_state(
                        pressure_kpa=sample.pressure_kpa,
                        liquid_composition=x,
                        vapor_pressure=correlations,
                        parameters=fitted.parameters,
                        temperature_bounds_k=temperature_bounds_k,
                    )
                records.append(_record(sample, direction, fitted.model, state))
            except (ArithmeticError, RuntimeError, ValueError) as error:
                records.append(
                    _record(
                        sample,
                        direction,
                        fitted.model,
                        None,
                        f"prediction_error:{str(error).splitlines()[0][:120]}",
                    )
                )
    return records


def unavailable_prediction_records(
    samples: Sequence[VLESample], model_name: str, reason: str
) -> list[dict[str, object]]:
    return [
        _record(sample, direction, model_name, None, reason)
        for sample in samples
        for direction in _directions(sample)
    ]
