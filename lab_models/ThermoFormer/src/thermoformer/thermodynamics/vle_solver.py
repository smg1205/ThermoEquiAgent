"""Differentiable low-pressure activity-coefficient bubble-state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .vapor_pressure import (
    CORRELATION_PARAMETER_COUNT,
    CORRELATION_TYPE_ANTOINE,
    CORRELATION_TYPE_DIPPR101,
)

if TYPE_CHECKING:
    from ..data import VLEBatch


@dataclass
class EquilibriumState:
    temperature_k: Tensor
    pressure_kpa: Tensor
    calculated_pressure_kpa: Tensor
    pressure_residual_kpa: Tensor
    x: Tensor
    y: Tensor
    gamma: Tensor
    psat_kpa: Tensor
    attention_bias: Tensor | None = None
    attention_bias_penalty: Tensor | None = None
    converged: Tensor | None = None
    iterations: int = 0


@dataclass
class ModeEquilibria:
    """Mode masks and label-independent bubble solves for one VLE batch."""

    isothermal_rows: Tensor
    isothermal: EquilibriumState | None
    isobaric_rows: Tensor
    isobaric: EquilibriumState | None


class ConvergenceError(RuntimeError):
    """Raised when a requested bubble-state solve does not converge."""


def _mark_convergence(
    state: EquilibriumState,
    iterations: int,
    absolute_tolerance_kpa: float,
    relative_tolerance: float,
) -> EquilibriumState:
    tolerance = absolute_tolerance_kpa + relative_tolerance * state.pressure_kpa.abs()
    converged = state.pressure_residual_kpa.abs() <= tolerance
    return replace(state, converged=converged, iterations=iterations)


def _raise_unless_converged(state: EquilibriumState, solver: str, strict: bool) -> None:
    if strict and state.converged is not None and not bool(state.converged.all().detach().cpu()):
        maximum = float(state.pressure_residual_kpa.abs().max().detach().cpu())
        raise ConvergenceError(f"{solver} did not converge; maximum pressure residual={maximum:.6g} kPa")


def _normalize(x: Tensor, mask: Tensor) -> Tensor:
    bounded = x.clamp_min(0.0) * mask
    return bounded / bounded.sum(-1, keepdim=True).clamp_min(1e-12)


def _apply_pure_property_correlations(
    learned_log_psat: Tensor,
    temperature_k: Tensor,
    mask: Tensor,
    pure_property_parameters: Tensor | None,
) -> Tensor:
    if pure_property_parameters is None:
        return learned_log_psat
    expected_shape = (*learned_log_psat.shape, CORRELATION_PARAMETER_COUNT)
    if pure_property_parameters.shape != expected_shape:
        raise ValueError(
            "pure_property_parameters must have shape "
            f"[batch, components, {CORRELATION_PARAMETER_COUNT}]"
        )
    coefficients = pure_property_parameters.to(learned_log_psat)
    temperature = temperature_k.expand_as(learned_log_psat).clamp_min(1e-6)
    correlation_type = coefficients[..., 0]
    a, b, c, d, e = (coefficients[..., index] for index in range(1, 6))
    minimum_temperature = coefficients[..., 6]
    maximum_temperature = coefficients[..., 7]
    temperature_offset = coefficients[..., 8]
    pressure_scale = coefficients[..., 9].clamp_min(1e-30)
    available = coefficients[..., 10] > 0.5
    in_range = (
        available
        & (temperature >= minimum_temperature)
        & (temperature <= maximum_temperature)
        & mask.bool()
    )

    antoine_temperature = temperature - temperature_offset
    denominator = c + antoine_temperature
    safe_denominator = torch.where(
        denominator.abs() > 1e-6,
        denominator,
        torch.where(
            denominator >= 0.0,
            torch.full_like(denominator, 1e-6),
            torch.full_like(denominator, -1e-6),
        ),
    )
    log_ten = torch.log(torch.tensor(10.0, dtype=temperature.dtype, device=temperature.device))
    log_pressure_scale = torch.log(pressure_scale)
    antoine_log_psat = log_ten * (a - b / safe_denominator) + log_pressure_scale

    temperature_power = torch.exp(
        (e * torch.log(temperature)).clamp(min=-80.0, max=80.0)
    )
    dippr_log_psat = (
        a + b / temperature + c * torch.log(temperature) + d * temperature_power
        + log_pressure_scale
    )
    is_antoine = correlation_type == CORRELATION_TYPE_ANTOINE
    is_dippr101 = correlation_type == CORRELATION_TYPE_DIPPR101
    correlation_log_psat = torch.where(is_antoine, antoine_log_psat, dippr_log_psat)
    valid = in_range & (is_dippr101 | (is_antoine & (denominator.abs() > 1e-6)))
    return torch.where(valid, correlation_log_psat, learned_log_psat)


def equilibrium_at_tp(
    model: nn.Module,
    molecules: Tensor,
    temperature_k: Tensor,
    pressure_kpa: Tensor,
    x: Tensor,
    mask: Tensor,
    pure_property_parameters: Tensor | None = None,
) -> EquilibriumState:
    """Evaluate the modified Raoult equation at an observed T-P-x state."""
    liquid = _normalize(x, mask)
    outputs = model(molecules, temperature_k, pressure_kpa, liquid, mask)
    gamma = torch.exp(outputs.log_gamma.clamp(-18.42068074, 18.42068074)) * mask
    log_psat = _apply_pure_property_correlations(
        outputs.log_psat,
        temperature_k,
        mask,
        pure_property_parameters,
    )
    psat = torch.exp(log_psat.clamp(-27.63102112, 27.63102112)) * mask
    partial_pressures = liquid * gamma * psat
    calculated_pressure = partial_pressures.sum(-1, keepdim=True)
    vapor = partial_pressures / calculated_pressure.clamp_min(1e-12)
    vapor = vapor * mask
    return EquilibriumState(
        temperature_k=temperature_k,
        pressure_kpa=pressure_kpa,
        calculated_pressure_kpa=calculated_pressure,
        pressure_residual_kpa=calculated_pressure - pressure_kpa,
        x=liquid,
        y=vapor,
        gamma=gamma,
        psat_kpa=psat,
        attention_bias=outputs.attention_bias,
        attention_bias_penalty=outputs.attention_bias_penalty,
    )


def solve_isothermal(
    model: nn.Module,
    molecules: Tensor,
    temperature_k: Tensor,
    x: Tensor,
    mask: Tensor,
    initial_pressure_kpa: Tensor | None = None,
    iterations: int = 24,
    damping: float = 0.5,
    absolute_tolerance_kpa: float = 1e-4,
    relative_tolerance: float = 1e-5,
    strict: bool = True,
    pure_property_parameters: Tensor | None = None,
) -> EquilibriumState:
    """Solve bubble pressure by a fixed-count differentiable fixed point."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    pressure = (
        torch.full_like(temperature_k, 101.325)
        if initial_pressure_kpa is None
        else initial_pressure_kpa
    )
    for _ in range(iterations):
        state = equilibrium_at_tp(
            model, molecules, temperature_k, pressure, x, mask, pure_property_parameters
        )
        pressure = (1.0 - damping) * pressure + damping * state.calculated_pressure_kpa
    result = equilibrium_at_tp(
        model, molecules, temperature_k, pressure, x, mask, pure_property_parameters
    )
    result = _mark_convergence(
        result, iterations, absolute_tolerance_kpa, relative_tolerance
    )
    _raise_unless_converged(result, "isothermal bubble-pressure solver", strict)
    return result


def solve_isobaric(
    model: nn.Module,
    molecules: Tensor,
    pressure_kpa: Tensor,
    x: Tensor,
    mask: Tensor,
    initial_temperature_k: Tensor | None = None,
    iterations: int = 16,
    finite_difference_k: float = 0.5,
    minimum_temperature_k: float = 150.0,
    maximum_temperature_k: float = 1500.0,
    absolute_tolerance_kpa: float = 1e-3,
    relative_tolerance: float = 1e-5,
    strict: bool = True,
    pure_property_parameters: Tensor | None = None,
) -> EquilibriumState:
    """Solve bubble temperature with bracketed, differentiable Newton steps."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if finite_difference_k <= 0.0:
        raise ValueError("finite_difference_k must be positive")
    if minimum_temperature_k >= maximum_temperature_k:
        raise ValueError("minimum_temperature_k must be below maximum_temperature_k")
    temperature = (
        torch.full_like(pressure_kpa, 350.0)
        if initial_temperature_k is None
        else initial_temperature_k
    ).clamp(minimum_temperature_k, maximum_temperature_k)
    lower_temperature = torch.full_like(pressure_kpa, minimum_temperature_k)
    upper_temperature = torch.full_like(pressure_kpa, maximum_temperature_k)
    lower_residual = equilibrium_at_tp(
        model, molecules, lower_temperature, pressure_kpa, x, mask, pure_property_parameters
    ).pressure_residual_kpa
    upper_residual = equilibrium_at_tp(
        model, molecules, upper_temperature, pressure_kpa, x, mask, pure_property_parameters
    ).pressure_residual_kpa
    bracketed = lower_residual * upper_residual <= 0.0
    delta = finite_difference_k
    for _ in range(iterations):
        center = equilibrium_at_tp(
            model, molecules, temperature, pressure_kpa, x, mask, pure_property_parameters
        )
        upper = equilibrium_at_tp(
            model, molecules, temperature + delta, pressure_kpa, x, mask, pure_property_parameters
        )
        lower = equilibrium_at_tp(
            model, molecules, temperature - delta, pressure_kpa, x, mask, pure_property_parameters
        )
        derivative = (upper.pressure_residual_kpa - lower.pressure_residual_kpa) / (2.0 * delta)
        safe_derivative = torch.where(
            derivative.abs() >= 1e-6,
            derivative,
            torch.where(derivative >= 0.0, torch.full_like(derivative, 1e-6), torch.full_like(derivative, -1e-6)),
        )
        newton = (temperature - center.pressure_residual_kpa / safe_derivative).clamp(
            minimum_temperature_k, maximum_temperature_k
        )
        midpoint = 0.5 * (lower_temperature + upper_temperature)
        newton_inside_bracket = (newton > lower_temperature) & (newton < upper_temperature)
        candidate = torch.where(
            bracketed,
            torch.where(newton_inside_bracket, newton, midpoint),
            newton,
        )
        center_tolerance = absolute_tolerance_kpa + relative_tolerance * pressure_kpa.abs()
        candidate = torch.where(
            center.pressure_residual_kpa.abs() <= center_tolerance,
            temperature,
            candidate,
        )
        candidate_residual = equilibrium_at_tp(
            model, molecules, candidate, pressure_kpa, x, mask, pure_property_parameters
        ).pressure_residual_kpa
        same_side_as_lower = lower_residual * candidate_residual > 0.0
        update_lower = bracketed & same_side_as_lower
        update_upper = bracketed & ~same_side_as_lower
        lower_temperature = torch.where(update_lower, candidate, lower_temperature)
        lower_residual = torch.where(update_lower, candidate_residual, lower_residual)
        upper_temperature = torch.where(update_upper, candidate, upper_temperature)
        upper_residual = torch.where(update_upper, candidate_residual, upper_residual)
        temperature = candidate
    result = equilibrium_at_tp(
        model, molecules, temperature, pressure_kpa, x, mask, pure_property_parameters
    )
    result = _mark_convergence(
        result, iterations, absolute_tolerance_kpa, relative_tolerance
    )
    _raise_unless_converged(result, "isobaric bubble-temperature solver", strict)
    return result


def solve_batch_modes(
    model: nn.Module,
    batch: "VLEBatch",
    iterations: int,
    strict: bool = False,
) -> ModeEquilibria:
    """Run the appropriate inference solve without using observed labels as initials."""
    isothermal_rows = batch.experiment_mode != 1
    isobaric_rows = batch.experiment_mode != 0
    isothermal = (
        solve_isothermal(
            model,
            batch.molecules[isothermal_rows],
            batch.temperature_k[isothermal_rows],
            batch.x[isothermal_rows],
            batch.mask[isothermal_rows],
            iterations=iterations,
            strict=strict,
            pure_property_parameters=batch.pure_property_parameters[isothermal_rows],
        )
        if bool(isothermal_rows.any())
        else None
    )
    isobaric = (
        solve_isobaric(
            model,
            batch.molecules[isobaric_rows],
            batch.pressure_kpa[isobaric_rows],
            batch.x[isobaric_rows],
            batch.mask[isobaric_rows],
            iterations=iterations,
            strict=strict,
            pure_property_parameters=batch.pure_property_parameters[isobaric_rows],
        )
        if bool(isobaric_rows.any())
        else None
    )
    return ModeEquilibria(
        isothermal_rows=isothermal_rows,
        isothermal=isothermal,
        isobaric_rows=isobaric_rows,
        isobaric=isobaric,
    )
