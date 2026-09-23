"""Data and physics objectives for thermodynamic-state training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..thermodynamics.vle_solver import EquilibriumState

if TYPE_CHECKING:
    from ..data import VLEBatch


@dataclass
class Objective:
    total: Tensor
    vapor_composition: Tensor
    pressure: Tensor
    temperature: Tensor
    pure_vapor_pressure: Tensor
    teacher_forced_fugacity: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "vapor_composition": float(self.vapor_composition.detach().cpu()),
            "pressure": float(self.pressure.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
            "pure_vapor_pressure": float(self.pure_vapor_pressure.detach().cpu()),
            "teacher_forced_fugacity": float(
                self.teacher_forced_fugacity.detach().cpu()
            ),
        }


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    flattened_weights = weights.reshape(-1).to(values)
    return torch.sum(values.reshape(-1) * flattened_weights) / flattened_weights.sum().clamp_min(1e-12)


def experimental_objective(
    state: EquilibriumState,
    observed_y: Tensor,
    observed_pressure_kpa: Tensor,
    quality_weight: Tensor,
    mask: Tensor,
    vapor_weight: float = 1.0,
    pressure_weight: float = 1.0,
    pure_weight: float = 0.5,
) -> Objective:
    """Supervise y, the bubble equation, and pure-component Psat anchors."""
    component_count = mask.sum(-1).clamp_min(1.0)
    vapor_per_sample = (((state.y - observed_y) ** 2) * mask).sum(-1) / component_count
    vapor_loss = _weighted_mean(vapor_per_sample, quality_weight)

    log_pressure_error = (
        torch.log(state.calculated_pressure_kpa.clamp_min(1e-12))
        - torch.log(observed_pressure_kpa.clamp_min(1e-12))
    ).squeeze(-1) ** 2
    pressure_loss = _weighted_mean(log_pressure_error, quality_weight)

    dominant_fraction, dominant_index = state.x.max(-1)
    endpoint = (dominant_fraction >= 0.999).to(state.x)
    predicted_pure = state.psat_kpa.gather(1, dominant_index.unsqueeze(-1)).squeeze(-1)
    pure_error = (
        torch.log(predicted_pure.clamp_min(1e-12))
        - torch.log(observed_pressure_kpa.squeeze(-1).clamp_min(1e-12))
    ) ** 2
    endpoint_weights = quality_weight.reshape(-1).to(state.x) * endpoint
    pure_loss = torch.sum(pure_error * endpoint_weights) / endpoint_weights.sum().clamp_min(1e-12)

    zero = state.y.sum() * 0.0
    total = vapor_weight * vapor_loss + pressure_weight * pressure_loss + pure_weight * pure_loss
    return Objective(
        total=total,
        vapor_composition=vapor_loss,
        pressure=pressure_loss,
        temperature=zero,
        pure_vapor_pressure=pure_loss,
        teacher_forced_fugacity=zero,
    )


def direct_vle_objective(
    model: nn.Module,
    batch: "VLEBatch",
    vapor_weight: float = 1.0,
    pressure_weight: float = 1.0,
    temperature_weight: float = 1.0,
) -> Objective:
    """Supervise a black-box direct head in both label-independent task directions."""
    zero = batch.y.sum() * 0.0
    vapor_terms: list[Tensor] = []
    isothermal_rows = batch.experiment_mode != 1
    if bool(isothermal_rows.any()):
        predicted = model.predict_direct(
            batch.molecules[isothermal_rows],
            batch.temperature_k[isothermal_rows],
            batch.pressure_kpa[isothermal_rows],
            batch.x[isothermal_rows],
            batch.mask[isothermal_rows],
            direction="isothermal",
        )
        component_count = batch.mask[isothermal_rows].sum(-1).clamp_min(1.0)
        vapor_error = (
            (predicted.y - batch.y[isothermal_rows]).square()
            * batch.mask[isothermal_rows]
        ).sum(-1) / component_count
        vapor_terms.append(
            _weighted_mean(vapor_error, batch.quality_weight[isothermal_rows])
        )
        pressure_error = (
            torch.log(predicted.pressure_kpa.clamp_min(1e-12))
            - torch.log(batch.pressure_kpa[isothermal_rows].clamp_min(1e-12))
        ).squeeze(-1).square()
        pressure = _weighted_mean(
            pressure_error, batch.quality_weight[isothermal_rows]
        )
    else:
        pressure = zero
    isobaric_rows = batch.experiment_mode != 0
    if bool(isobaric_rows.any()):
        predicted = model.predict_direct(
            batch.molecules[isobaric_rows],
            batch.temperature_k[isobaric_rows],
            batch.pressure_kpa[isobaric_rows],
            batch.x[isobaric_rows],
            batch.mask[isobaric_rows],
            direction="isobaric",
        )
        component_count = batch.mask[isobaric_rows].sum(-1).clamp_min(1.0)
        vapor_error = (
            (predicted.y - batch.y[isobaric_rows]).square()
            * batch.mask[isobaric_rows]
        ).sum(-1) / component_count
        vapor_terms.append(
            _weighted_mean(vapor_error, batch.quality_weight[isobaric_rows])
        )
        temperature_error = (
            (predicted.temperature_k - batch.temperature_k[isobaric_rows]) / 100.0
        ).squeeze(-1).square()
        temperature = _weighted_mean(
            temperature_error, batch.quality_weight[isobaric_rows]
        )
    else:
        temperature = zero
    vapor = torch.stack(vapor_terms).mean() if vapor_terms else zero
    total = (
        vapor_weight * vapor
        + pressure_weight * pressure
        + temperature_weight * temperature
    )
    return Objective(
        total=total,
        vapor_composition=vapor,
        pressure=pressure,
        temperature=temperature,
        pure_vapor_pressure=zero,
        teacher_forced_fugacity=zero,
    )


def with_teacher_forced_fugacity_equilibrium(
    objective: Objective,
    state: EquilibriumState,
    batch: "VLEBatch",
    weight: float,
) -> Objective:
    """Penalize component fugacity imbalance at the observed T-P-x-y state.

    The low-pressure vapor phase is treated as ideal, so the component balance
    is x_i gamma_i Psat_i = y_i P.  Dividing by P makes the residual
    dimensionless and stable at zero-composition endpoints.
    """

    if weight <= 0.0:
        return objective
    liquid_fugacity_kpa = state.x * state.gamma * state.psat_kpa
    vapor_fugacity_kpa = batch.y * batch.pressure_kpa
    normalized_residual = (
        liquid_fugacity_kpa - vapor_fugacity_kpa
    ) / batch.pressure_kpa.clamp_min(1e-12)
    component_count = batch.mask.sum(-1).clamp_min(1.0)
    per_sample = (normalized_residual.square() * batch.mask).sum(-1) / component_count
    fugacity = _weighted_mean(per_sample, batch.quality_weight)
    return replace(
        objective,
        total=objective.total + weight * fugacity,
        teacher_forced_fugacity=fugacity,
    )
