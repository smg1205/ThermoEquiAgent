"""Training-only parameter estimation for thermodynamic comparison models."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
import torch

from ..data import VLESample
from ..data.splitting import canonical_smiles
from .thermodynamic_models import (
    ActivityModelName,
    ActivityModelParameters,
    LogLinearVaporPressure,
    activity_coefficients,
)


@dataclass(frozen=True)
class FittedActivityModel:
    model: ActivityModelName
    components: tuple[str, ...]
    parameters: ActivityModelParameters
    success: bool
    training_samples: int
    fitted_component_observations: int
    rmse_log_gamma: float | None
    optimizer_status: int
    optimizer_message: str


@dataclass(frozen=True)
class SharedActivityModel:
    """One ordered interaction parameter per molecular pair in a split."""

    model: ActivityModelName
    pair_parameters_k: dict[tuple[str, str], float]
    success: bool
    training_samples: int
    fitted_pairs: int
    initial_training_loss: float
    final_training_loss: float
    optimizer_message: str
    nrtl_alpha: float = 0.3
    uniquac_sizes: Mapping[str, tuple[float, float]] | None = None

    def for_components(self, components: Sequence[str]) -> FittedActivityModel | None:
        ordered = tuple(sorted(canonical_smiles(value) for value in components))
        count = len(ordered)
        matrix = np.zeros((count, count), dtype=float)
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                key = (ordered[i], ordered[j])
                if key not in self.pair_parameters_k:
                    return None
                matrix[i, j] = self.pair_parameters_k[key]
        sizes = (
            _resolved_uniquac_sizes(ordered, self.uniquac_sizes)
            if self.model == "uniquac" else None
        )
        if self.model == "uniquac" and sizes is None:
            return None
        r, q = sizes if sizes is not None else (None, None)
        return FittedActivityModel(
            model=self.model,
            components=ordered,
            parameters=ActivityModelParameters(
                model=self.model,
                interaction_b=matrix,
                uniquac_r=r,
                uniquac_q=q,
                nrtl_alpha=self.nrtl_alpha,
            ),
            success=self.success,
            training_samples=self.training_samples,
            fitted_component_observations=0,
            rmse_log_gamma=None,
            optimizer_status=0 if self.success else -1,
            optimizer_message=self.optimizer_message,
        )


def _ordered_sample(sample: VLESample, components: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    positions = {canonical_smiles(value): index for index, value in enumerate(sample.smiles)}
    if set(positions) != set(components):
        raise ValueError("sample components do not match the fitted system")
    order = [positions[value] for value in components]
    return (
        np.asarray([sample.liquid_composition[index] for index in order], dtype=float),
        np.asarray([sample.vapor_composition[index] for index in order], dtype=float),
    )


def fit_vapor_pressure_correlations(
    samples: Sequence[VLESample],
    *,
    endpoint_threshold: float = 0.999,
) -> tuple[dict[str, LogLinearVaporPressure], dict[str, object]]:
    """Fit ``ln(Psat/kPa)=a+b/T`` using pure endpoints in training only."""

    observations: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    all_components = {canonical_smiles(value) for row in samples for value in row.smiles}
    for sample in samples:
        x = np.asarray(sample.liquid_composition, dtype=float)
        dominant = int(np.argmax(x))
        if x[dominant] < endpoint_threshold:
            continue
        temperature = float(sample.temperature_k)
        pressure = float(sample.pressure_kpa)
        if temperature <= 0.0 or pressure <= 0.0:
            continue
        observations[canonical_smiles(sample.smiles[dominant])].append(
            (temperature, pressure, float(sample.quality_weight))
        )

    fitted: dict[str, LogLinearVaporPressure] = {}
    details: dict[str, dict[str, float | int | None]] = {}
    for component in sorted(all_components):
        rows = observations.get(component, [])
        unique_temperatures = sorted({round(row[0], 10) for row in rows})
        if len(unique_temperatures) < 2:
            details[component] = {
                "endpoint_rows": len(rows),
                "unique_temperatures": len(unique_temperatures),
                "rmse_log_pressure": None,
            }
            continue
        temperature = np.asarray([row[0] for row in rows], dtype=float)
        pressure = np.asarray([row[1] for row in rows], dtype=float)
        weights = np.sqrt(np.asarray([max(row[2], 1e-12) for row in rows]))
        inverse_temperature = 1.0 / temperature
        log_pressure = np.log(pressure)
        statistical_weights = weights**2
        weight_sum = float(statistical_weights.sum())
        mean_inverse_temperature = float(
            np.sum(statistical_weights * inverse_temperature) / weight_sum
        )
        mean_log_pressure = float(
            np.sum(statistical_weights * log_pressure) / weight_sum
        )
        centered_temperature = inverse_temperature - mean_inverse_temperature
        denominator = float(np.sum(statistical_weights * centered_temperature**2))
        if denominator <= 1e-20:
            continue
        slope = float(
            np.sum(
                statistical_weights
                * centered_temperature
                * (log_pressure - mean_log_pressure)
            )
            / denominator
        )
        coefficients = np.asarray(
            [mean_log_pressure - slope * mean_inverse_temperature, slope]
        )
        design = np.column_stack([np.ones_like(temperature), inverse_temperature])
        model = LogLinearVaporPressure(
            intercept=float(coefficients[0]),
            inverse_temperature=float(coefficients[1]),
        )
        residual = design @ coefficients - np.log(pressure)
        fitted[component] = model
        details[component] = {
            "endpoint_rows": len(rows),
            "unique_temperatures": len(unique_temperatures),
            "minimum_temperature_k": float(temperature.min()),
            "maximum_temperature_k": float(temperature.max()),
            "rmse_log_pressure": float(np.sqrt(np.average(residual**2, weights=weights**2))),
        }
    return fitted, {
        "components": len(all_components),
        "covered_components": len(fitted),
        "component_coverage": len(fitted) / len(all_components) if all_components else 0.0,
        "fits": details,
    }


@lru_cache(maxsize=None)
def _uniquac_size_for_smiles(smiles: str) -> tuple[float, float] | None:
    from thermo import ChemicalConstantsPackage

    try:
        constants, _ = ChemicalConstantsPackage.from_IDs([f"smiles={smiles}"])
        r = float(constants.UNIFAC_Rs[0])
        q = float(constants.UNIFAC_Qs[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(r) or not math.isfinite(q) or r <= 0.0 or q <= 0.0:
        return None
    return r, q


def uniquac_size_parameters(
    components: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    values = [_uniquac_size_for_smiles(value) for value in components]
    if any(value is None for value in values):
        return None
    return (
        np.asarray([value[0] for value in values if value is not None]),
        np.asarray([value[1] for value in values if value is not None]),
    )


def _resolved_uniquac_sizes(
    components: Sequence[str],
    overrides: Mapping[str, tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if overrides is not None and all(value in overrides for value in components):
        pairs = [overrides[value] for value in components]
        values = np.asarray(pairs, dtype=float)
        if values.shape == (len(components), 2) and np.isfinite(values).all() and np.all(values > 0.0):
            return values[:, 0], values[:, 1]
    return uniquac_size_parameters(components)


def _interaction_matrix(values: np.ndarray, component_count: int) -> np.ndarray:
    matrix = np.zeros((component_count, component_count), dtype=float)
    pairs = [(i, j) for i in range(component_count) for j in range(component_count) if i != j]
    for value, (i, j) in zip(values, pairs):
        matrix[i, j] = value
    return matrix


def fit_activity_model(
    model: ActivityModelName,
    samples: Sequence[VLESample],
    vapor_pressure: Mapping[str, LogLinearVaporPressure],
    *,
    minimum_composition: float = 1e-5,
    maximum_evaluations: int = 200,
    nrtl_alpha: float = 0.3,
) -> FittedActivityModel:
    """Regress ordered-pair energy parameters from training VLE states only."""

    if not samples:
        raise ValueError("At least one training sample is required")
    components = tuple(sorted(canonical_smiles(value) for value in samples[0].smiles))
    if any(tuple(sorted(canonical_smiles(value) for value in row.smiles)) != components for row in samples):
        raise ValueError("All fitting samples must belong to one chemical system")
    missing_pure = [value for value in components if value not in vapor_pressure]
    if missing_pure:
        raise ValueError(f"Missing vapor-pressure fits for: {', '.join(missing_pure)}")

    sizes = uniquac_size_parameters(components) if model == "uniquac" else None
    if model == "uniquac" and sizes is None:
        raise ValueError("UNIQUAC molecular size parameters are unavailable")

    observations: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    correlations = tuple(vapor_pressure[value] for value in components)
    for sample in samples:
        x, y = _ordered_sample(sample, components)
        psat = np.asarray([value.pressure_kpa(sample.temperature_k) for value in correlations])
        valid = (
            (x >= minimum_composition)
            & (y >= minimum_composition)
            & np.isfinite(psat)
            & (psat > 0.0)
        )
        if not np.any(valid):
            continue
        log_gamma = np.zeros_like(x)
        log_gamma[valid] = np.log(
            y[valid] * sample.pressure_kpa / (x[valid] * psat[valid])
        )
        valid &= np.isfinite(log_gamma) & (np.abs(log_gamma) <= 12.0)
        if np.any(valid):
            observations.append(
                (sample.temperature_k, x, log_gamma, valid, max(sample.quality_weight, 1e-12))
            )
    component_observations = sum(int(row[3].sum()) for row in observations)
    parameter_count = len(components) * (len(components) - 1)
    if component_observations < parameter_count:
        raise ValueError("Insufficient finite activity-coefficient observations")

    r, q = sizes if sizes is not None else (None, None)

    def residual(values: np.ndarray) -> np.ndarray:
        parameters = ActivityModelParameters(
            model=model,
            interaction_b=_interaction_matrix(values, len(components)),
            uniquac_r=r,
            uniquac_q=q,
            nrtl_alpha=nrtl_alpha,
        )
        parts = []
        for temperature, x, target, valid, quality in observations:
            predicted = activity_coefficients(temperature, x, parameters)
            difference = np.log(np.clip(predicted[valid], 1e-12, 1e12)) - target[valid]
            difference = np.nan_to_num(difference, nan=1e3, posinf=1e3, neginf=-1e3)
            parts.append(math.sqrt(quality) * difference)
        return np.concatenate(parts)

    def objective(values: np.ndarray) -> float:
        scaled = residual(values) / 0.5
        return float(np.sum(0.5 * (np.sqrt(1.0 + scaled**2) - 1.0)))

    optimized = minimize(
        objective,
        np.zeros(parameter_count, dtype=float),
        method="L-BFGS-B",
        bounds=[(-5000.0, 5000.0)] * parameter_count,
        options={"maxiter": maximum_evaluations, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not optimized.success or not np.isfinite(optimized.x).all():
        optimized = minimize(
            objective,
            np.nan_to_num(optimized.x, nan=0.0, posinf=0.0, neginf=0.0),
            method="Powell",
            bounds=[(-5000.0, 5000.0)] * parameter_count,
            options={"maxiter": maximum_evaluations, "ftol": 1e-10, "xtol": 1e-7},
        )
    parameters = ActivityModelParameters(
        model=model,
        interaction_b=_interaction_matrix(optimized.x, len(components)),
        uniquac_r=r,
        uniquac_q=q,
        nrtl_alpha=nrtl_alpha,
    )
    final_residual = residual(optimized.x)
    success = bool(optimized.success and np.isfinite(optimized.x).all())
    return FittedActivityModel(
        model=model,
        components=components,
        parameters=parameters,
        success=success,
        training_samples=len(samples),
        fitted_component_observations=component_observations,
        rmse_log_gamma=(
            float(np.sqrt(np.mean(final_residual**2))) if final_residual.size else None
        ),
        optimizer_status=int(optimized.status),
        optimizer_message=str(optimized.message),
    )


def _torch_activity_coefficients(
    model: ActivityModelName,
    temperature: torch.Tensor,
    composition: torch.Tensor,
    interaction_b: torch.Tensor,
    *,
    nrtl_alpha: float,
    uniquac_r: torch.Tensor | None,
    uniquac_q: torch.Tensor | None,
) -> torch.Tensor:
    """Vectorized activity coefficients used only by the global fitter."""

    x = composition.clamp_min(1e-15)
    x = x / x.sum(dim=-1, keepdim=True)
    reduced = (interaction_b / temperature[:, None, None]).clamp(-50.0, 50.0)
    count = x.shape[-1]
    identity = torch.eye(count, dtype=x.dtype, device=x.device)[None, :, :]
    if model == "nrtl":
        alpha = nrtl_alpha * (1.0 - identity)
        g = torch.exp(-alpha * reduced)
        weighted_rows = x[:, :, None]
        denominators = (weighted_rows * g).sum(dim=1).clamp_min(1e-15)
        weighted_tau = (weighted_rows * reduced * g).sum(dim=1) / denominators
        second = (
            x[:, None, :]
            * g
            / denominators[:, None, :]
            * (reduced - weighted_tau[:, None, :])
        ).sum(dim=2)
        log_gamma = weighted_tau + second
    elif model == "wilson":
        lambdas = torch.exp(reduced) * (1.0 - identity) + identity
        row_sums = (lambdas * x[:, None, :]).sum(dim=2).clamp_min(1e-15)
        second = (x[:, :, None] * lambdas / row_sums[:, :, None]).sum(dim=1)
        log_gamma = 1.0 - torch.log(row_sums) - second
    else:
        if uniquac_r is None or uniquac_q is None:
            raise ValueError("UNIQUAC fitting requires r and q")
        r = uniquac_r
        q = uniquac_q
        phi = r * x / (r * x).sum(dim=1, keepdim=True).clamp_min(1e-15)
        theta = q * x / (q * x).sum(dim=1, keepdim=True).clamp_min(1e-15)
        coordination = 10.0
        ell = coordination * 0.5 * (r - q) - (r - 1.0)
        taus = torch.exp(reduced) * (1.0 - identity) + identity
        column_sums = (theta[:, :, None] * taus).sum(dim=1).clamp_min(1e-15)
        residual_sum = (
            theta[:, None, :] * taus / column_sums[:, None, :]
        ).sum(dim=2)
        log_gamma = (
            torch.log((phi / x).clamp_min(1e-15))
            + coordination * 0.5 * q * torch.log((theta / phi).clamp_min(1e-15))
            + ell
            - phi / x * (x * ell).sum(dim=1, keepdim=True)
            - q * torch.log(column_sums)
            + q
            - q * residual_sum
        )
    return torch.exp(log_gamma.clamp(-30.0, 30.0))


def fit_shared_activity_model(
    model: ActivityModelName,
    samples: Sequence[VLESample],
    vapor_pressure: Mapping[str, LogLinearVaporPressure],
    *,
    maximum_iterations: int = 200,
    nrtl_alpha: float = 0.3,
    uniquac_sizes: Mapping[str, tuple[float, float]] | None = None,
) -> SharedActivityModel:
    """Fit shared ordered pairs directly to training P and y VLE residuals."""

    if not samples:
        raise ValueError("At least one training sample is required")
    usable: list[tuple[VLESample, tuple[str, ...], np.ndarray, np.ndarray]] = []
    pair_keys: set[tuple[str, str]] = set()
    for sample in samples:
        components = tuple(sorted(canonical_smiles(value) for value in sample.smiles))
        if any(value not in vapor_pressure for value in components):
            continue
        sizes = _resolved_uniquac_sizes(components, uniquac_sizes) if model == "uniquac" else None
        if model == "uniquac" and sizes is None:
            continue
        x, y = _ordered_sample(sample, components)
        usable.append((sample, components, x, y))
        pair_keys.update(
            (left, right) for left in components for right in components if left != right
        )
    ordered_pairs = tuple(sorted(pair_keys))
    pair_index = {key: index for index, key in enumerate(ordered_pairs)}
    if not usable or not ordered_pairs:
        raise ValueError("No training rows have complete pure-property/model coverage")

    maximum_components = max(len(row[1]) for row in usable)
    row_count = len(usable)
    temperatures = np.zeros(row_count)
    pressures = np.zeros(row_count)
    compositions = np.zeros((row_count, maximum_components))
    vapor_compositions = np.zeros_like(compositions)
    vapor_pressures = np.ones_like(compositions)
    masks = np.zeros_like(compositions)
    pair_indices = np.full(
        (row_count, maximum_components, maximum_components), -1, dtype=np.int64
    )
    r_values = np.ones_like(compositions)
    q_values = np.ones_like(compositions)
    quality = np.zeros(row_count)
    for row_index, (sample, components, x, y) in enumerate(usable):
        count = len(components)
        temperatures[row_index] = sample.temperature_k
        pressures[row_index] = sample.pressure_kpa
        compositions[row_index, :count] = x
        vapor_compositions[row_index, :count] = y
        masks[row_index, :count] = 1.0
        quality[row_index] = max(sample.quality_weight, 1e-12)
        vapor_pressures[row_index, :count] = [
            vapor_pressure[value].pressure_kpa(sample.temperature_k) for value in components
        ]
        for i in range(count):
            for j in range(count):
                if i != j:
                    pair_indices[row_index, i, j] = pair_index[(components[i], components[j])]
        if model == "uniquac":
            sizes = _resolved_uniquac_sizes(components, uniquac_sizes)
            if sizes is None:
                raise AssertionError("UNIQUAC coverage changed during fitting")
            r_values[row_index, :count], q_values[row_index, :count] = sizes

    dtype = torch.float64
    temperature_t = torch.as_tensor(temperatures, dtype=dtype)
    pressure_t = torch.as_tensor(pressures, dtype=dtype)
    x_t = torch.as_tensor(compositions, dtype=dtype)
    y_t = torch.as_tensor(vapor_compositions, dtype=dtype)
    psat_t = torch.as_tensor(vapor_pressures, dtype=dtype)
    mask_t = torch.as_tensor(masks, dtype=dtype)
    quality_t = torch.as_tensor(quality, dtype=dtype)
    indices_t = torch.as_tensor(pair_indices, dtype=torch.long)
    r_t = torch.as_tensor(r_values, dtype=dtype) if model == "uniquac" else None
    q_t = torch.as_tensor(q_values, dtype=dtype) if model == "uniquac" else None
    raw = torch.nn.Parameter(torch.zeros(len(ordered_pairs), dtype=dtype))

    def loss_value() -> torch.Tensor:
        bounded = 5000.0 * torch.tanh(raw)
        safe_indices = indices_t.clamp_min(0)
        matrix = bounded[safe_indices] * (indices_t >= 0).to(dtype)
        gamma = _torch_activity_coefficients(
            model,
            temperature_t,
            x_t,
            matrix,
            nrtl_alpha=nrtl_alpha,
            uniquac_r=r_t,
            uniquac_q=q_t,
        )
        contributions = x_t * gamma * psat_t * mask_t
        calculated_pressure = contributions.sum(dim=1).clamp_min(1e-15)
        calculated_y = contributions / calculated_pressure[:, None]
        pressure_residual = torch.log(calculated_pressure) - torch.log(pressure_t)
        y_residual = ((calculated_y - y_t) * mask_t).square().sum(dim=1) / mask_t.sum(
            dim=1
        ).clamp_min(1.0)
        per_row = pressure_residual.square() + y_residual
        return (per_row * quality_t).sum() / quality_t.sum()

    initial = float(loss_value().detach())
    optimizer = torch.optim.LBFGS(
        [raw],
        lr=0.8,
        max_iter=maximum_iterations,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        final = float(loss_value().detach())
        values = (5000.0 * torch.tanh(raw)).detach().cpu().numpy()
        success = (
            math.isfinite(final)
            and np.isfinite(values).all()
            and final <= initial + max(1e-12, abs(initial) * 1e-8)
        )
        message = (
            "LBFGS converged without worsening the direct-VLE objective"
            if success
            else "nonfinite_or_worsened_fit"
        )
    except (RuntimeError, ValueError) as error:
        final = math.inf
        values = np.zeros(len(ordered_pairs))
        success = False
        message = f"optimizer_error:{str(error).splitlines()[0][:160]}"
    return SharedActivityModel(
        model=model,
        pair_parameters_k={key: float(values[index]) for index, key in enumerate(ordered_pairs)},
        success=success,
        training_samples=len(usable),
        fitted_pairs=len(ordered_pairs),
        initial_training_loss=initial,
        final_training_loss=final,
        optimizer_message=message,
        nrtl_alpha=nrtl_alpha,
        uniquac_sizes=uniquac_sizes,
    )
