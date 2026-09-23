"""Unified low-pressure VLE interface for local-composition activity models.

The three adapters share the same pure-component vapor-pressure calibration,
ideal-vapor assumption, and bubble-point solvers. This isolates the activity
coefficient model as the only scientific difference between comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import brentq
from thermo.nrtl import NRTL_gammas
from thermo.uniquac import UNIQUAC_gammas
from thermo.wilson import Wilson_gammas


ActivityModelName = Literal["nrtl", "wilson", "uniquac"]


@dataclass(frozen=True)
class LogLinearVaporPressure:
    """Training-endpoint fit of ``ln(Psat/kPa) = a + b/T``."""

    intercept: float
    inverse_temperature: float

    def pressure_kpa(self, temperature_k: float) -> float:
        if not math.isfinite(temperature_k) or temperature_k <= 0.0:
            return math.nan
        log_pressure = self.intercept + self.inverse_temperature / temperature_k
        return float(math.exp(min(50.0, max(-50.0, log_pressure))))


@dataclass(frozen=True)
class ActivityModelParameters:
    """Temperature-dependent ordered-pair parameters shared by all adapters."""

    model: ActivityModelName
    interaction_b: np.ndarray
    uniquac_r: np.ndarray | None = None
    uniquac_q: np.ndarray | None = None
    nrtl_alpha: float = 0.3

    def __post_init__(self) -> None:
        interaction = np.asarray(self.interaction_b, dtype=float)
        if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
            raise ValueError("interaction_b must be a square matrix")
        if not np.isfinite(interaction).all():
            raise ValueError("interaction_b must be finite")
        if self.model not in {"nrtl", "wilson", "uniquac"}:
            raise ValueError(f"Unsupported activity model: {self.model}")
        if self.model == "uniquac":
            if self.uniquac_r is None or self.uniquac_q is None:
                raise ValueError("UNIQUAC requires r and q molecular size parameters")
            r = np.asarray(self.uniquac_r, dtype=float)
            q = np.asarray(self.uniquac_q, dtype=float)
            if r.shape != (interaction.shape[0],) or q.shape != r.shape:
                raise ValueError("UNIQUAC r and q must match the component count")
            if not np.isfinite(r).all() or not np.isfinite(q).all():
                raise ValueError("UNIQUAC r and q must be finite")
            if np.any(r <= 0.0) or np.any(q <= 0.0):
                raise ValueError("UNIQUAC r and q must be positive")


@dataclass(frozen=True)
class BubbleState:
    temperature_k: float
    pressure_kpa: float
    liquid_composition: np.ndarray
    vapor_composition: np.ndarray
    activity_coefficients: np.ndarray
    vapor_pressures_kpa: np.ndarray
    converged: bool
    residual_kpa: float
    iterations: int
    failure_reason: str | None = None


def _closed_composition(composition: Sequence[float]) -> np.ndarray:
    values = np.asarray(composition, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("composition must contain at least two components")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("composition must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("composition sum must be positive")
    return values / total


def activity_coefficients(
    temperature_k: float,
    liquid_composition: Sequence[float],
    parameters: ActivityModelParameters,
) -> np.ndarray:
    """Evaluate NRTL, Wilson, or UNIQUAC through the authoritative thermo API."""

    x = _closed_composition(liquid_composition)
    if parameters.interaction_b.shape != (x.size, x.size):
        raise ValueError("activity parameters do not match the composition")
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        return np.full(x.size, np.nan)
    evaluation_x = np.clip(x, 1e-15, None)
    evaluation_x /= evaluation_x.sum()
    reduced = np.clip(parameters.interaction_b / temperature_k, -50.0, 50.0)
    if parameters.model == "nrtl":
        alpha = np.full((x.size, x.size), parameters.nrtl_alpha, dtype=float)
        np.fill_diagonal(alpha, 0.0)
        gamma = NRTL_gammas(evaluation_x.tolist(), reduced.tolist(), alpha.tolist())
    elif parameters.model == "wilson":
        lambdas = np.exp(reduced)
        np.fill_diagonal(lambdas, 1.0)
        gamma = Wilson_gammas(evaluation_x.tolist(), lambdas.tolist())
    else:
        taus = np.exp(reduced)
        np.fill_diagonal(taus, 1.0)
        gamma = UNIQUAC_gammas(
            evaluation_x.tolist(),
            np.asarray(parameters.uniquac_r, dtype=float).tolist(),
            np.asarray(parameters.uniquac_q, dtype=float).tolist(),
            taus.tolist(),
        )
    values = np.asarray(gamma, dtype=float)
    return values if values.shape == x.shape else np.full(x.size, np.nan)


def solve_isothermal_state(
    *,
    temperature_k: float,
    liquid_composition: Sequence[float],
    vapor_pressure: Sequence[LogLinearVaporPressure],
    parameters: ActivityModelParameters,
) -> BubbleState:
    """Return low-pressure bubble pressure and vapor composition at fixed T,x."""

    x = _closed_composition(liquid_composition)
    if len(vapor_pressure) != x.size:
        raise ValueError("vapor-pressure correlations must match the component count")
    gamma = activity_coefficients(temperature_k, x, parameters)
    psat = np.asarray([model.pressure_kpa(temperature_k) for model in vapor_pressure])
    contributions = x * gamma * psat
    pressure = float(np.sum(contributions))
    valid = (
        np.isfinite(gamma).all()
        and np.isfinite(psat).all()
        and np.isfinite(pressure)
        and pressure > 0.0
        and np.all(gamma > 0.0)
        and np.all(psat > 0.0)
    )
    y = contributions / pressure if valid else np.full(x.size, np.nan)
    valid = valid and np.isfinite(y).all() and np.all(y >= 0.0)
    return BubbleState(
        temperature_k=float(temperature_k),
        pressure_kpa=pressure,
        liquid_composition=x,
        vapor_composition=y,
        activity_coefficients=gamma,
        vapor_pressures_kpa=psat,
        converged=bool(valid),
        residual_kpa=0.0 if valid else math.nan,
        iterations=1,
        failure_reason=None if valid else "nonfinite_isothermal_state",
    )


def solve_isobaric_state(
    *,
    pressure_kpa: float,
    liquid_composition: Sequence[float],
    vapor_pressure: Sequence[LogLinearVaporPressure],
    parameters: ActivityModelParameters,
    temperature_bounds_k: tuple[float, float] = (150.0, 1500.0),
    grid_points: int = 257,
    absolute_tolerance_kpa: float = 1e-6,
) -> BubbleState:
    """Solve bubble temperature at fixed P,x without an observed-T initial value."""

    x = _closed_composition(liquid_composition)
    lower, upper = map(float, temperature_bounds_k)
    if not math.isfinite(pressure_kpa) or pressure_kpa <= 0.0:
        raise ValueError("pressure_kpa must be positive and finite")
    if not 0.0 < lower < upper or grid_points < 2:
        raise ValueError("temperature bounds and grid_points are invalid")

    def residual(temperature: float) -> float:
        return (
            solve_isothermal_state(
                temperature_k=temperature,
                liquid_composition=x,
                vapor_pressure=vapor_pressure,
                parameters=parameters,
            ).pressure_kpa
            - pressure_kpa
        )

    grid = np.linspace(lower, upper, grid_points)
    residuals = np.asarray([residual(float(value)) for value in grid])
    bracket: tuple[float, float] | None = None
    for index in range(grid_points - 1):
        left, right = residuals[index : index + 2]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            bracket = (float(grid[index]), float(grid[index]))
            break
        if left * right <= 0.0:
            bracket = (float(grid[index]), float(grid[index + 1]))
            break
    if bracket is None:
        return BubbleState(
            temperature_k=math.nan,
            pressure_kpa=float(pressure_kpa),
            liquid_composition=x,
            vapor_composition=np.full(x.size, np.nan),
            activity_coefficients=np.full(x.size, np.nan),
            vapor_pressures_kpa=np.full(x.size, np.nan),
            converged=False,
            residual_kpa=math.nan,
            iterations=grid_points,
            failure_reason="bubble_temperature_not_bracketed",
        )
    temperature = (
        bracket[0]
        if bracket[0] == bracket[1]
        else float(
            brentq(
                residual,
                bracket[0],
                bracket[1],
                xtol=1e-9,
                rtol=1e-12,
                maxiter=100,
            )
        )
    )
    state = solve_isothermal_state(
        temperature_k=temperature,
        liquid_composition=x,
        vapor_pressure=vapor_pressure,
        parameters=parameters,
    )
    final_residual = state.pressure_kpa - pressure_kpa
    converged = state.converged and abs(final_residual) <= max(
        absolute_tolerance_kpa, 1e-8 * pressure_kpa
    )
    return BubbleState(
        temperature_k=temperature,
        pressure_kpa=float(pressure_kpa),
        liquid_composition=x,
        vapor_composition=state.vapor_composition,
        activity_coefficients=state.activity_coefficients,
        vapor_pressures_kpa=state.vapor_pressures_kpa,
        converged=converged,
        residual_kpa=final_residual,
        iterations=grid_points + 1,
        failure_reason=None if converged else "bubble_temperature_residual",
    )
