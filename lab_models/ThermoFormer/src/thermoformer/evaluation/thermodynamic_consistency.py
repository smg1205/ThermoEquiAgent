"""Independent physical-validity metrics for ThermoFormer ablations."""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..data import VLESample, VLETensorDataset, collate_vle
from ..thermodynamics.vapor_pressure import PurePropertyCatalog
from ..data.splitting import system_id
from ..thermodynamics import equilibrium_at_tp, solve_isobaric, solve_isothermal


def simplex_tangent_directions(
    component_count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """Return an orthonormal basis whose directions preserve ``sum(x)=1``."""
    if component_count < 2:
        raise ValueError("A composition simplex needs at least two components")
    basis = torch.zeros(
        component_count,
        component_count - 1,
        dtype=dtype,
        device=device,
    )
    for index in range(component_count - 1):
        basis[index, index] = 1.0
        basis[-1, index] = -1.0
    orthonormal, _ = torch.linalg.qr(basis, mode="reduced")
    return orthonormal.T


def gibbs_duhem_residuals(
    model: nn.Module,
    molecules: Tensor,
    temperature_k: Tensor,
    pressure_kpa: Tensor,
    x: Tensor,
    mask: Tensor,
) -> Tensor:
    """Evaluate directional ``sum_i x_i d ln(gamma_i)`` using autograd."""
    residuals: list[Tensor] = []
    chunk_size = 256
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        sample_x = x[start:stop].detach().clone().requires_grad_(True)
        output = model(
            molecules[start:stop],
            temperature_k[start:stop],
            pressure_kpa[start:stop],
            sample_x,
            mask[start:stop],
        )
        component_gradients = []
        for component in range(sample_x.shape[1]):
            value = output.log_gamma[:, component]
            gradient = (
                torch.autograd.grad(
                    value.sum(), sample_x, retain_graph=True,
                    create_graph=False, allow_unused=True,
                )[0]
                if value.requires_grad else None
            )
            component_gradients.append(
                gradient if gradient is not None else torch.zeros_like(sample_x)
            )
        jacobian = torch.stack(component_gradients, dim=1)
        chunk_mask = mask[start:stop]
        for count in (2, 3):
            rows = chunk_mask.sum(-1).eq(count)
            if not bool(rows.any()):
                continue
            directions = simplex_tangent_directions(
                count, dtype=sample_x.dtype, device=sample_x.device
            )
            directional = torch.einsum(
                "boi,di->bod", jacobian[rows, :count, :count], directions
            )
            values = torch.einsum(
                "bo,bod->bd", sample_x[rows, :count], directional
            )
            residuals.extend(values.abs().flatten().unbind())
    if not residuals:
        return torch.empty(0, dtype=x.dtype, device=x.device)
    return torch.stack(residuals)


def gibbs_duhem_residuals_finite_difference(
    model: nn.Module,
    molecules: Tensor,
    temperature_k: Tensor,
    pressure_kpa: Tensor,
    x: Tensor,
    mask: Tensor,
    step: float = 1e-4,
) -> Tensor:
    """Finite-difference validation of the tangent autograd residual."""
    residuals: list[Tensor] = []
    with torch.no_grad():
        for sample_index in range(x.shape[0]):
            count = int(mask[sample_index].sum().item())
            directions = simplex_tangent_directions(
                count, dtype=x.dtype, device=x.device
            )
            for direction in directions:
                padded = torch.zeros_like(x[sample_index])
                padded[:count] = direction
                upper_x = x[sample_index : sample_index + 1] + step * padded
                lower_x = x[sample_index : sample_index + 1] - step * padded
                if bool((upper_x[:, :count] <= 0.0).any()) or bool(
                    (lower_x[:, :count] <= 0.0).any()
                ):
                    continue
                upper = model(
                    molecules[sample_index : sample_index + 1],
                    temperature_k[sample_index : sample_index + 1],
                    pressure_kpa[sample_index : sample_index + 1],
                    upper_x,
                    mask[sample_index : sample_index + 1],
                ).log_gamma[0, :count]
                lower = model(
                    molecules[sample_index : sample_index + 1],
                    temperature_k[sample_index : sample_index + 1],
                    pressure_kpa[sample_index : sample_index + 1],
                    lower_x,
                    mask[sample_index : sample_index + 1],
                ).log_gamma[0, :count]
                derivative = (upper - lower) / (2.0 * step)
                residuals.append(
                    (x[sample_index, :count] * derivative).sum().abs()
                )
    if not residuals:
        return torch.empty(0, dtype=x.dtype, device=x.device)
    return torch.stack(residuals)


def composition_closure_metrics(x: Tensor, y: Tensor, mask: Tensor) -> dict[str, float]:
    """Report liquid/vapor closure and component-bound violations."""
    present_x = x * mask
    present_y = y * mask
    values = y[mask.bool()]
    return {
        "x_closure_mean_abs": float((present_x.sum(-1) - 1.0).abs().mean()),
        "y_closure_mean_abs": float((present_y.sum(-1) - 1.0).abs().mean()),
        "fraction_y_below_zero": float((values < 0.0).float().mean()),
        "fraction_y_above_one": float((values > 1.0).float().mean()),
    }


def phase_curve_smoothness(
    coordinate: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    """Measure smoothness without treating physically nonmonotonic curves as invalid."""
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    if coordinate.ndim != 1 or values.shape[0] != coordinate.size or coordinate.size < 4:
        raise ValueError("A phase curve needs one coordinate and at least four states")
    if values.ndim == 1:
        values = values[:, None]
    segment = np.diff(values, axis=0)
    total_variation = float(np.linalg.norm(segment, axis=1).sum())
    first = np.gradient(values, coordinate, axis=0, edge_order=2)
    first_jump = np.linalg.norm(np.diff(first, axis=0), axis=1)
    second = np.gradient(first, coordinate, axis=0, edge_order=2)
    second_magnitude = np.linalg.norm(second, axis=1)
    return {
        "total_variation": total_variation,
        "first_derivative_jump_p95": float(np.percentile(first_jump, 95.0)),
        "second_derivative_magnitude_mean": float(second_magnitude.mean()),
    }


def summarize_absolute(values: Tensor | np.ndarray, prefix: str) -> dict[str, Any]:
    """Summarize a residual distribution without hiding an empty observable."""
    array = np.asarray(
        values.detach().cpu().numpy() if isinstance(values, Tensor) else values,
        dtype=float,
    )
    array = np.abs(array[np.isfinite(array)])
    if array.size == 0:
        return {f"{prefix}_mean_abs": None, f"{prefix}_p95_abs": None, f"{prefix}_max_abs": None}
    return {
        f"{prefix}_mean_abs": float(array.mean()),
        f"{prefix}_p95_abs": float(np.percentile(array, 95.0)),
        f"{prefix}_max_abs": float(array.max()),
    }


def combined_nonphysical_rate(
    ordinary_failures: int,
    ordinary_predictions: int,
    pure_limit_errors: Sequence[float],
    pure_limit_threshold: float,
) -> float | None:
    """Combine observed-state and synthetic near-pure prediction checks."""
    pure = np.asarray(pure_limit_errors, dtype=float)
    pure_failures = int(
        np.sum(~np.isfinite(pure) | (np.abs(pure) > pure_limit_threshold))
    )
    total = int(ordinary_predictions) + int(pure.size)
    return (int(ordinary_failures) + pure_failures) / total if total else None


def _representative_samples(
    samples: Sequence[VLESample], max_systems: int | None
) -> list[VLESample]:
    by_system: "OrderedDict[str, VLESample]" = OrderedDict()
    for sample in sorted(samples, key=lambda row: (system_id(row), row.temperature_k, row.pressure_kpa)):
        by_system.setdefault(system_id(sample), sample)
    representatives = list(by_system.values())
    return representatives if max_systems is None else representatives[:max_systems]


def _batch(
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    device: torch.device,
    pure_property_catalog: PurePropertyCatalog | None = None,
):
    dataset = VLETensorDataset(
        samples, feature_map, pure_property_catalog=pure_property_catalog
    )
    return collate_vle([dataset[index] for index in range(len(dataset))]).to(device)


def _composition_grid_tensors(batch, grid_points: int) -> tuple[Tensor, ...]:
    molecules: list[Tensor] = []
    temperatures: list[Tensor] = []
    pressures: list[Tensor] = []
    compositions: list[Tensor] = []
    masks: list[Tensor] = []
    for index in range(batch.x.shape[0]):
        count = int(batch.mask[index].sum().item())
        for _, path in _phase_paths(count, grid_points, batch.x.device):
            padded = torch.zeros(grid_points, batch.x.shape[1], device=batch.x.device)
            padded[:, :count] = path
            molecules.append(batch.molecules[index : index + 1].expand(grid_points, -1, -1))
            temperatures.append(batch.temperature_k[index : index + 1].expand(grid_points, -1))
            pressures.append(batch.pressure_kpa[index : index + 1].expand(grid_points, -1))
            compositions.append(padded)
            masks.append(batch.mask[index : index + 1].expand(grid_points, -1))
    return tuple(
        torch.cat(values, dim=0)
        for values in (molecules, temperatures, pressures, compositions, masks)
    )


def _permutation_errors(
    model: nn.Module,
    batch,
    direct: bool,
    solver_iterations: int,
) -> dict[str, list[float]]:
    errors = {
        "y": [],
        "pressure_kpa": [],
        "temperature_k": [],
        "gamma": [],
        "psat_kpa": [],
    }
    for count in (2, 3):
        rows = batch.mask.sum(-1).eq(count)
        if not bool(rows.any()):
            continue
        molecules = batch.molecules[rows]
        temperature = batch.temperature_k[rows]
        pressure = batch.pressure_kpa[rows]
        x = batch.x[rows]
        mask = batch.mask[rows]
        pure_parameters = batch.pure_property_parameters[rows]
        if direct:
            originals = {
                direction: model.predict_direct(
                    molecules,
                    temperature,
                    pressure,
                    x,
                    mask,
                    direction=direction,
                )
                for direction in ("isothermal", "isobaric")
            }
        else:
            original = equilibrium_at_tp(
                model, molecules, temperature, pressure, x, mask, pure_parameters
            )
            original_isothermal = solve_isothermal(
                model, molecules, temperature, x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
            original_isobaric = solve_isobaric(
                model, molecules, pressure, x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
        for permutation in itertools.permutations(range(count)):
            if permutation == tuple(range(count)):
                continue
            order = torch.tensor(permutation, device=x.device)
            padded_order = torch.cat(
                [order, torch.arange(count, x.shape[1], device=x.device)]
            )
            if direct:
                for direction, baseline in originals.items():
                    permuted = model.predict_direct(
                        molecules[:, padded_order],
                        temperature,
                        pressure,
                        x[:, padded_order],
                        mask[:, padded_order],
                        direction=direction,
                    )
                    errors["y"].extend(
                        (permuted.y[:, :count] - baseline.y[:, order]).abs().flatten().tolist()
                    )
                    key = "pressure_kpa" if direction == "isothermal" else "temperature_k"
                    permuted_state = (
                        permuted.pressure_kpa if direction == "isothermal" else permuted.temperature_k
                    )
                    baseline_state = (
                        baseline.pressure_kpa if direction == "isothermal" else baseline.temperature_k
                    )
                    errors[key].extend((permuted_state - baseline_state).abs().flatten().tolist())
            else:
                permuted = equilibrium_at_tp(
                    model,
                    molecules[:, padded_order],
                    temperature,
                    pressure,
                    x[:, padded_order],
                    mask[:, padded_order],
                    pure_parameters[:, padded_order],
                )
                for key, permuted_value, baseline_value in (
                    ("gamma", permuted.gamma[:, :count], original.gamma[:, order]),
                    ("psat_kpa", permuted.psat_kpa[:, :count], original.psat_kpa[:, order]),
                ):
                    errors[key].extend((permuted_value - baseline_value).abs().flatten().tolist())
                permuted_isothermal = solve_isothermal(
                    model, molecules[:, padded_order], temperature, x[:, padded_order],
                    mask[:, padded_order], iterations=solver_iterations, strict=False,
                    pure_property_parameters=pure_parameters[:, padded_order],
                )
                permuted_isobaric = solve_isobaric(
                    model, molecules[:, padded_order], pressure, x[:, padded_order],
                    mask[:, padded_order], iterations=solver_iterations, strict=False,
                    pure_property_parameters=pure_parameters[:, padded_order],
                )
                errors["y"].extend(
                    (permuted_isothermal.y[:, :count] - original_isothermal.y[:, order]).abs().flatten().tolist()
                )
                errors["y"].extend(
                    (permuted_isobaric.y[:, :count] - original_isobaric.y[:, order]).abs().flatten().tolist()
                )
                errors["pressure_kpa"].extend(
                    (permuted_isothermal.pressure_kpa - original_isothermal.pressure_kpa).abs().flatten().tolist()
                )
                errors["temperature_k"].extend(
                    (permuted_isobaric.temperature_k - original_isobaric.temperature_k).abs().flatten().tolist()
                )
    return errors


def _pure_limit_errors(
    model: nn.Module,
    batch,
    direct: bool,
    solver_iterations: int,
    samples: Sequence[VLESample],
    pure_references: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    errors = {
        "log_gamma": [],
        "y": [],
        "pressure_relative": [],
        "temperature_k": [],
        "combined_normalized": [],
        "reference_available": [],
    }
    for sample_index in range(batch.x.shape[0]):
        count = int(batch.mask[sample_index].sum().item())
        compositions = []
        target_indices = []
        for epsilon in (0.01, 0.005, 0.001):
            for component in range(count):
                composition = torch.zeros(batch.x.shape[1], dtype=batch.x.dtype, device=batch.x.device)
                composition[:count] = epsilon / (count - 1)
                composition[component] = 1.0 - epsilon
                compositions.append(composition)
                target_indices.append(component)
        x = torch.stack(compositions)
        repeats = x.shape[0]
        molecules = batch.molecules[sample_index : sample_index + 1].expand(repeats, -1, -1)
        temperature = batch.temperature_k[sample_index : sample_index + 1].expand(repeats, -1)
        pressure = batch.pressure_kpa[sample_index : sample_index + 1].expand(repeats, -1)
        mask = batch.mask[sample_index : sample_index + 1].expand(repeats, -1)
        pure_parameters = batch.pure_property_parameters[
            sample_index : sample_index + 1
        ].expand(repeats, -1, -1)
        targets = torch.tensor(target_indices, device=x.device)
        pure_x = torch.zeros_like(x)
        pure_x.scatter_(1, targets[:, None], 1.0)
        if direct:
            near_isothermal = model.predict_direct(
                molecules, temperature, pressure, x, mask, direction="isothermal"
            )
            pure_isothermal = model.predict_direct(
                molecules, temperature, pressure, pure_x, mask, direction="isothermal"
            )
            near_isobaric = model.predict_direct(
                molecules, temperature, pressure, x, mask, direction="isobaric"
            )
            pure_isobaric = model.predict_direct(
                molecules, temperature, pressure, pure_x, mask, direction="isobaric"
            )
        else:
            output = model(molecules, temperature, pressure, x, mask)
            target_log_gamma = output.log_gamma.gather(1, targets[:, None]).squeeze(1)
            errors["log_gamma"].extend(target_log_gamma.abs().tolist())
            near_isothermal = solve_isothermal(
                model, molecules, temperature, x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
            pure_isothermal = solve_isothermal(
                model, molecules, temperature, pure_x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
            near_isobaric = solve_isobaric(
                model, molecules, pressure, x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
            pure_isobaric = solve_isobaric(
                model, molecules, pressure, pure_x, mask,
                iterations=solver_iterations, strict=False,
                pure_property_parameters=pure_parameters,
            )
        y_error = torch.maximum(
            (1.0 - near_isothermal.y.gather(1, targets[:, None]).squeeze(1)).abs(),
            (1.0 - near_isobaric.y.gather(1, targets[:, None]).squeeze(1)).abs(),
        )
        reference_pressure = []
        reference_temperature = []
        for target in target_indices:
            correlation = pure_references.get(samples[sample_index].smiles[target])
            if correlation is None:
                reference_pressure.append(math.nan)
                reference_temperature.append(math.nan)
                continue
            intercept, slope = correlation
            temperature_value = float(temperature[0].item())
            pressure_value = float(pressure[0].item())
            reference_pressure.append(math.exp(intercept + slope / temperature_value))
            inverse_temperature = (math.log(max(pressure_value, 1e-12)) - intercept) / slope
            reference_temperature.append(1.0 / inverse_temperature if inverse_temperature > 0.0 else math.nan)
        reference_pressure_tensor = torch.tensor(reference_pressure, device=x.device)
        reference_temperature_tensor = torch.tensor(reference_temperature, device=x.device)
        reference_available = torch.isfinite(reference_pressure_tensor) & torch.isfinite(
            reference_temperature_tensor
        )
        if not direct:
            reference_pressure_tensor = torch.where(
                torch.isfinite(reference_pressure_tensor),
                reference_pressure_tensor,
                pure_isothermal.pressure_kpa.squeeze(1),
            )
            reference_temperature_tensor = torch.where(
                torch.isfinite(reference_temperature_tensor),
                reference_temperature_tensor,
                pure_isobaric.temperature_k.squeeze(1),
            )
        pressure_error = torch.log(
            near_isothermal.pressure_kpa.squeeze(1).clamp_min(1e-12)
            / reference_pressure_tensor.clamp_min(1e-12)
        ).abs()
        temperature_error = (
            near_isobaric.temperature_k.squeeze(1) - reference_temperature_tensor
        ).abs()
        combined = y_error
        combined = torch.where(
            torch.isfinite(pressure_error), torch.maximum(combined, pressure_error), combined
        )
        combined = torch.where(
            torch.isfinite(temperature_error),
            torch.maximum(combined, temperature_error / 100.0),
            combined,
        )
        if direct:
            combined = torch.where(
                reference_available, combined, torch.full_like(combined, math.nan)
            )
        errors["y"].extend(y_error.tolist())
        errors["pressure_relative"].extend(pressure_error.tolist())
        errors["temperature_k"].extend(temperature_error.tolist())
        errors["combined_normalized"].extend(combined.tolist())
        errors["reference_available"].extend(reference_available.float().tolist())
    return errors


def _pure_reference_correlations(
    samples: Sequence[VLESample],
) -> dict[str, tuple[float, float]]:
    anchors: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        for index, fraction in enumerate(sample.liquid_composition):
            if fraction >= 0.999 and sample.temperature_k > 0.0 and sample.pressure_kpa > 0.0:
                anchors.setdefault(sample.smiles[index], []).append(
                    (sample.temperature_k, sample.pressure_kpa)
                )
    correlations: dict[str, tuple[float, float]] = {}
    for smiles, values in anchors.items():
        temperatures = np.asarray([row[0] for row in values], dtype=float)
        pressures = np.asarray([row[1] for row in values], dtype=float)
        if np.unique(temperatures).size < 2:
            continue
        slope, intercept = np.polyfit(1.0 / temperatures, np.log(pressures), 1)
        if np.isfinite(intercept) and np.isfinite(slope) and abs(slope) > 1e-12:
            correlations[smiles] = (float(intercept), float(slope))
    return correlations


def _phase_paths(count: int, grid_points: int, device: torch.device) -> list[tuple[Tensor, Tensor]]:
    coordinate = torch.linspace(0.01, 0.99, grid_points, device=device)
    if count == 2:
        return [(coordinate, torch.stack([coordinate, 1.0 - coordinate], dim=-1))]
    paths = []
    for component in range(count):
        x = torch.full((grid_points, count), 0.0, device=device)
        x[:] = ((1.0 - coordinate) / (count - 1)).unsqueeze(-1)
        x[:, component] = coordinate
        paths.append((coordinate, x))
    return paths


def _phase_metrics(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    device: torch.device,
    direct: bool,
    grid_points: int,
    solver_iterations: int,
    pure_property_catalog: PurePropertyCatalog | None,
) -> list[dict[str, float]]:
    molecule_parts: list[Tensor] = []
    temperature_parts: list[Tensor] = []
    pressure_parts: list[Tensor] = []
    composition_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    pure_parts: list[Tensor] = []
    spans: list[tuple[np.ndarray, int, int, int]] = []
    offset = 0
    for sample in samples:
        count = sample.component_count
        sample_batch = _batch([sample], feature_map, device, pure_property_catalog)
        for coordinate, path in _phase_paths(count, grid_points, device):
            padded_x = torch.zeros(grid_points, 3, device=device)
            padded_x[:, :count] = path
            molecule_parts.append(sample_batch.molecules.expand(grid_points, -1, -1))
            temperature_parts.append(sample_batch.temperature_k.expand(grid_points, -1))
            pressure_parts.append(sample_batch.pressure_kpa.expand(grid_points, -1))
            composition_parts.append(padded_x)
            mask_parts.append(sample_batch.mask.expand(grid_points, -1))
            pure_parts.append(sample_batch.pure_property_parameters.expand(grid_points, -1, -1))
            spans.append((coordinate.detach().cpu().numpy(), count, offset, offset + grid_points))
            offset += grid_points
    molecules = torch.cat(molecule_parts)
    temperature = torch.cat(temperature_parts)
    pressure = torch.cat(pressure_parts)
    compositions = torch.cat(composition_parts)
    masks = torch.cat(mask_parts)
    pure_parameters = torch.cat(pure_parts)
    if direct:
        isothermal = model.predict_direct(
            molecules, temperature, pressure, compositions, masks, direction="isothermal"
        )
        isobaric = model.predict_direct(
            molecules, temperature, pressure, compositions, masks, direction="isobaric"
        )
    else:
        isothermal = solve_isothermal(
            model, molecules, temperature, compositions, masks,
            iterations=solver_iterations, strict=False,
            pure_property_parameters=pure_parameters,
        )
        isobaric = solve_isobaric(
            model, molecules, pressure, compositions, masks,
            iterations=solver_iterations, strict=False,
            pure_property_parameters=pure_parameters,
        )
    metrics: list[dict[str, float]] = []
    for coordinate, count, start, stop in spans:
        curves = (
            torch.cat(
                [torch.log(isothermal.pressure_kpa[start:stop]), isothermal.y[start:stop, :count]],
                dim=-1,
            ),
            torch.cat(
                [isobaric.temperature_k[start:stop] / 100.0, isobaric.y[start:stop, :count]],
                dim=-1,
            ),
        )
        for curve in curves:
            metrics.append(phase_curve_smoothness(coordinate, curve.detach().cpu().numpy()))
    return metrics


def evaluate_thermodynamic_consistency(
    model: nn.Module,
    samples: Sequence[VLESample],
    feature_map: dict[str, np.ndarray],
    device: torch.device,
    *,
    prediction_records: Sequence[dict[str, Any]] | None = None,
    solver_iterations: int = 24,
    grid_points: int = 21,
    max_systems: int | None = None,
    pure_reference_samples: Sequence[VLESample] | None = None,
    pure_property_catalog: PurePropertyCatalog | None = None,
) -> dict[str, Any]:
    """Evaluate physical validity independently of target prediction errors."""
    if not samples:
        raise ValueError("Thermodynamic consistency evaluation needs test samples")
    if grid_points < 5 or (max_systems is not None and max_systems < 1):
        raise ValueError("grid_points must be at least five and max_systems positive when set")
    from . import predict_vle

    model.to(device).eval()
    direct = getattr(getattr(model, "config", None), "decoder_mode", None) == "direct_vle"
    representatives = _representative_samples(samples, max_systems)
    batch = _batch(representatives, feature_map, device, pure_property_catalog)
    grid_molecules, grid_temperature, grid_pressure, grid_x, grid_mask = (
        _composition_grid_tensors(batch, grid_points)
    )
    if direct:
        gd = torch.empty(0)
        gd_fd = torch.empty(0)
    else:
        gd = gibbs_duhem_residuals(
            model,
            grid_molecules,
            grid_temperature,
            grid_pressure,
            grid_x,
            grid_mask,
        )
        validation_rows = min(32, grid_x.shape[0])
        gd_fd = gibbs_duhem_residuals_finite_difference(
            model,
            grid_molecules[:validation_rows],
            grid_temperature[:validation_rows],
            grid_pressure[:validation_rows],
            grid_x[:validation_rows],
            grid_mask[:validation_rows],
        )
    permutation_samples = representatives if max_systems is not None else samples
    permutation_batch = _batch(
        permutation_samples, feature_map, device, pure_property_catalog
    )
    pure_references = _pure_reference_correlations(
        pure_reference_samples if pure_reference_samples is not None else samples
    )
    with torch.no_grad():
        permutation = _permutation_errors(
            model, permutation_batch, direct, solver_iterations
        )
        pure = _pure_limit_errors(
            model,
            batch,
            direct,
            solver_iterations,
            representatives,
            pure_references,
        )
        phase = _phase_metrics(
            model,
            representatives,
            feature_map,
            device,
            direct,
            grid_points,
            solver_iterations,
            pure_property_catalog,
        )
    records = list(prediction_records) if prediction_records is not None else predict_vle(
        model,
        samples,
        feature_map,
        batch_size=128,
        device=device,
        solver_iterations=solver_iterations,
    )
    x_rows = []
    y_rows = []
    masks = []
    residuals = []
    nonphysical = 0
    pure_failure_threshold = 0.05
    for record in records:
        count = int(record["component_count"])
        x_rows.append([record.get(f"x_{index + 1}") or 0.0 for index in range(3)])
        y_rows.append([record.get(f"y_pred_{index + 1}") or 0.0 for index in range(3)])
        masks.append([1.0 if index < count else 0.0 for index in range(3)])
        residual = record.get("pressure_residual_kpa")
        if residual is not None and np.isfinite(float(residual)):
            residuals.append(float(residual))
        gross_residual = residual is not None and abs(float(residual)) > 0.1
        if bool(record.get("nonphysical")) or not bool(record.get("converged")) or gross_residual:
            nonphysical += 1
    closure = composition_closure_metrics(
        torch.tensor(x_rows), torch.tensor(y_rows), torch.tensor(masks)
    )
    equilibrium = summarize_absolute(residuals, "equilibrium_residual")
    pure_vle = pure["combined_normalized"]
    pure_reference_coverage = float(np.mean(pure["reference_available"]))
    direct_reference_complete = not direct or pure_reference_coverage >= 1.0 - 1e-12
    result: dict[str, Any] = {
        "evaluated_systems": len(representatives),
        "evaluated_predictions": len(records),
        "gibbs_duhem_grid_states": int(grid_x.shape[0]) if not direct else 0,
        "permutation_states": len(permutation_samples),
        **summarize_absolute(gd, "gibbs_duhem"),
        **summarize_absolute(gd_fd, "gibbs_duhem_finite_difference"),
        **closure,
        **summarize_absolute(permutation["y"], "permutation_y"),
        **summarize_absolute(permutation["pressure_kpa"], "permutation_pressure_kpa"),
        **summarize_absolute(permutation["temperature_k"], "permutation_temperature_k"),
        **summarize_absolute(permutation["gamma"], "permutation_gamma"),
        **summarize_absolute(permutation["psat_kpa"], "permutation_psat_kpa"),
        **summarize_absolute(pure["log_gamma"], "pure_limit_log_gamma"),
        **summarize_absolute(pure["y"], "pure_limit_y"),
        **summarize_absolute(pure["pressure_relative"], "pure_limit_pressure_relative"),
        **summarize_absolute(pure["temperature_k"], "pure_limit_temperature_k"),
        **summarize_absolute(pure_vle, "pure_limit_vle"),
        "pure_limit_reference_coverage": pure_reference_coverage,
        "equilibrium_residual_mean_abs_kpa": equilibrium["equilibrium_residual_mean_abs"],
        "equilibrium_residual_p95_abs_kpa": equilibrium["equilibrium_residual_p95_abs"],
        "equilibrium_residual_max_abs_kpa": equilibrium["equilibrium_residual_max_abs"],
        "solver_convergence_failure_rate": float(
            np.mean([not bool(record.get("converged")) for record in records])
        ),
        "pure_limit_failure_rate": (
            float(np.mean(np.asarray(pure_vle) > pure_failure_threshold))
            if pure_vle and direct_reference_complete else None
        ),
        "ordinary_nonphysical_prediction_rate": (
            nonphysical / len(records) if records else None
        ),
        "nonphysical_prediction_rate": (
            combined_nonphysical_rate(
                nonphysical,
                len(records),
                pure_vle,
                pure_failure_threshold,
            )
            if direct_reference_complete else None
        ),
        "physical_criteria": {
            "gross_equilibrium_residual_kpa": 0.1,
            "pure_limit_vle_error": pure_failure_threshold,
        },
    }
    for name in (
        "total_variation",
        "first_derivative_jump_p95",
        "second_derivative_magnitude_mean",
    ):
        values = [row[name] for row in phase]
        result[f"phase_{name}_mean"] = float(np.mean(values)) if values else None
        result[f"phase_{name}_p95"] = float(np.percentile(values, 95.0)) if values else None
    return result
