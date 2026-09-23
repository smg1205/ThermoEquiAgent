"""Validated pure-component vapor-pressure correlations.

The catalog is deliberately separate from the learned ``P_i^sat(T)`` branch.
Each entry declares a correlation type, pressure unit, and valid temperature
range. Correlations are used only inside that range; the model supplies the
fallback everywhere else.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch import Tensor

CORRELATION_TYPE_ANTOINE = 1.0
CORRELATION_TYPE_DIPPR101 = 2.0
CORRELATION_PARAMETER_COUNT = 11
MMHG_TO_KPA = 0.133322368

_PRESSURE_TO_KPA = {
    "pa": 1e-3,
    "kpa": 1.0,
    "bar": 100.0,
    "mmhg": MMHG_TO_KPA,
}


def _pressure_scale(unit: str) -> float:
    normalized = unit.strip().casefold()
    try:
        return _PRESSURE_TO_KPA[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(_PRESSURE_TO_KPA))
        raise ValueError(
            f"Unsupported pressure_unit '{unit}'; choose one of: {choices}"
        ) from error


def _temperature_offset(unit: str) -> float:
    normalized = unit.strip().casefold().replace("°", "")
    if normalized in {"c", "degc", "celsius"}:
        return 273.15
    if normalized in {"k", "kelvin"}:
        return 0.0
    raise ValueError("temperature_unit must be 'C' or 'K'")


def _validate_finite_range(
    values: Sequence[float], minimum: float, maximum: float
) -> None:
    if not all(math.isfinite(value) for value in (*values, minimum, maximum)):
        raise ValueError("Pure-property correlation parameters must be finite")
    if minimum >= maximum:
        raise ValueError("Pure-property temperature range must be increasing")


@dataclass(frozen=True)
class AntoineParameters:
    """Antoine ``log10(P_unit) = A - B / (C + T_unit)`` parameters."""

    a: float
    b: float
    c: float
    minimum_temperature_k: float
    maximum_temperature_k: float
    pressure_unit: str = "mmHg"
    temperature_unit: str = "C"

    def __post_init__(self) -> None:
        _validate_finite_range(
            (self.a, self.b, self.c),
            self.minimum_temperature_k,
            self.maximum_temperature_k,
        )
        _pressure_scale(self.pressure_unit)
        _temperature_offset(self.temperature_unit)

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                CORRELATION_TYPE_ANTOINE,
                self.a,
                self.b,
                self.c,
                0.0,
                0.0,
                self.minimum_temperature_k,
                self.maximum_temperature_k,
                _temperature_offset(self.temperature_unit),
                _pressure_scale(self.pressure_unit),
                1.0,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class DIPPR101Parameters:
    """DIPPR 101 ``ln(P_unit)=A+B/T+C ln(T)+D T**E`` parameters."""

    a: float
    b: float
    c: float
    d: float
    e: float
    minimum_temperature_k: float
    maximum_temperature_k: float
    pressure_unit: str = "Pa"
    temperature_unit: str = "K"

    def __post_init__(self) -> None:
        _validate_finite_range(
            (self.a, self.b, self.c, self.d, self.e),
            self.minimum_temperature_k,
            self.maximum_temperature_k,
        )
        _pressure_scale(self.pressure_unit)
        if _temperature_offset(self.temperature_unit) != 0.0:
            raise ValueError("DIPPR 101 requires temperature_unit='K'")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                CORRELATION_TYPE_DIPPR101,
                self.a,
                self.b,
                self.c,
                self.d,
                self.e,
                self.minimum_temperature_k,
                self.maximum_temperature_k,
                0.0,
                _pressure_scale(self.pressure_unit),
                1.0,
            ],
            dtype=np.float32,
        )


PurePropertyParameters = Union[AntoineParameters, DIPPR101Parameters]


@dataclass(frozen=True)
class PurePropertyCatalog:
    entries: dict[str, PurePropertyParameters]
    source: str = ""

    @property
    def covered_smiles(self) -> frozenset[str]:
        return frozenset(self.entries)

    def parameters_for(self, smiles: Sequence[str]) -> np.ndarray:
        rows = [
            self.entries[smile].as_array()
            if smile in self.entries
            else np.zeros(CORRELATION_PARAMETER_COUNT, dtype=np.float32)
            for smile in smiles
        ]
        return np.stack(rows)


def empty_pure_property_catalog() -> PurePropertyCatalog:
    return PurePropertyCatalog(entries={})


def _entry_type(raw_parameters: dict[str, object]) -> str:
    value = str(raw_parameters.get("type", "antoine")).strip().casefold()
    aliases = {
        "antoine": "antoine",
        "dippr101": "dippr101",
        "dippr-101": "dippr101",
        "dippr_101": "dippr101",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported pure-property correlation type '{value}'")
    return aliases[value]


def _require_exact_fields(
    smiles: str,
    raw_parameters: dict[str, object],
    required: set[str],
    optional: set[str],
) -> None:
    fields = set(raw_parameters)
    missing = required - fields
    unknown = fields - required - optional
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(
            f"Invalid pure-property entry '{smiles}': {'; '.join(details)}"
        )


def load_pure_property_catalog(path: Path) -> PurePropertyCatalog:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            "Pure-property catalog root must be a SMILES-to-parameters object"
        )
    entries: dict[str, PurePropertyParameters] = {}
    common_required = {
        "a",
        "b",
        "c",
        "minimum_temperature_k",
        "maximum_temperature_k",
    }
    common_optional = {"type", "pressure_unit", "temperature_unit"}
    for raw_smiles, raw_parameters in payload.items():
        smiles = str(raw_smiles).strip()
        if not smiles:
            raise ValueError("Pure-property catalog SMILES keys cannot be empty")
        if not isinstance(raw_parameters, dict):
            raise ValueError(f"Pure-property entry '{smiles}' must be an object")
        try:
            correlation_type = _entry_type(raw_parameters)
            if correlation_type == "antoine":
                _require_exact_fields(
                    smiles, raw_parameters, common_required, common_optional
                )
                entries[smiles] = AntoineParameters(
                    a=float(raw_parameters["a"]),
                    b=float(raw_parameters["b"]),
                    c=float(raw_parameters["c"]),
                    minimum_temperature_k=float(
                        raw_parameters["minimum_temperature_k"]
                    ),
                    maximum_temperature_k=float(
                        raw_parameters["maximum_temperature_k"]
                    ),
                    pressure_unit=str(raw_parameters.get("pressure_unit", "mmHg")),
                    temperature_unit=str(raw_parameters.get("temperature_unit", "C")),
                )
            else:
                required = common_required | {"d", "e"}
                _require_exact_fields(
                    smiles, raw_parameters, required, common_optional
                )
                entries[smiles] = DIPPR101Parameters(
                    a=float(raw_parameters["a"]),
                    b=float(raw_parameters["b"]),
                    c=float(raw_parameters["c"]),
                    d=float(raw_parameters["d"]),
                    e=float(raw_parameters["e"]),
                    minimum_temperature_k=float(
                        raw_parameters["minimum_temperature_k"]
                    ),
                    maximum_temperature_k=float(
                        raw_parameters["maximum_temperature_k"]
                    ),
                    pressure_unit=str(raw_parameters.get("pressure_unit", "Pa")),
                    temperature_unit=str(raw_parameters.get("temperature_unit", "K")),
                )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid pure-property entry '{smiles}': {error}") from error
    return PurePropertyCatalog(entries=entries, source=str(path.resolve()))


# Compatibility names for existing Antoine-only integrations.
AntoineCatalog = PurePropertyCatalog


def empty_antoine_catalog() -> PurePropertyCatalog:
    return empty_pure_property_catalog()


def load_antoine_catalog(path: Path) -> PurePropertyCatalog:
    return load_pure_property_catalog(path)


@dataclass(frozen=True)
class ExternalVaporPressureEvaluation:
    """External log-Psat values and a strict per-component validity mask."""

    log_psat_kpa: Tensor
    valid: Tensor


def evaluate_pure_property_correlations(
    temperature_k: Tensor,
    mask: Tensor,
    pure_property_parameters: Tensor,
) -> ExternalVaporPressureEvaluation:
    """Evaluate declared correlations without a learned-branch fallback.

    A value is valid only when its catalog entry is available, the requested
    temperature lies inside the declared range, the pressure is finite, and
    ``d log(Psat) / dT`` is strictly positive at that temperature.
    """
    expected_shape = (*mask.shape, CORRELATION_PARAMETER_COUNT)
    if pure_property_parameters.shape != expected_shape:
        raise ValueError(
            "pure_property_parameters must have shape "
            f"[batch, components, {CORRELATION_PARAMETER_COUNT}]"
        )
    coefficients = pure_property_parameters.to(mask)
    temperature = temperature_k.to(mask).expand_as(mask).clamp_min(1e-6)
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

    denominator = c + temperature - temperature_offset
    safe_denominator = torch.where(
        denominator.abs() > 1e-6,
        denominator,
        torch.where(
            denominator >= 0.0,
            torch.full_like(denominator, 1e-6),
            torch.full_like(denominator, -1e-6),
        ),
    )
    log_ten = torch.log(
        torch.tensor(10.0, dtype=temperature.dtype, device=temperature.device)
    )
    log_pressure_scale = torch.log(pressure_scale)
    antoine_log_psat = log_ten * (a - b / safe_denominator) + log_pressure_scale
    antoine_derivative = log_ten * b / safe_denominator.square()

    temperature_power = torch.exp(
        (e * torch.log(temperature)).clamp(min=-80.0, max=80.0)
    )
    dippr_log_psat = (
        a + b / temperature + c * torch.log(temperature)
        + d * temperature_power + log_pressure_scale
    )
    dippr_derivative = (
        -b / temperature.square()
        + c / temperature
        + d * e * temperature_power / temperature
    )
    is_antoine = correlation_type == CORRELATION_TYPE_ANTOINE
    is_dippr101 = correlation_type == CORRELATION_TYPE_DIPPR101
    log_psat = torch.where(is_antoine, antoine_log_psat, dippr_log_psat)
    derivative = torch.where(is_antoine, antoine_derivative, dippr_derivative)
    valid = (
        in_range
        & (is_antoine | is_dippr101)
        & (~is_antoine | (denominator.abs() > 1e-6))
        & torch.isfinite(log_psat)
        & torch.isfinite(derivative)
        & (derivative > 0.0)
    )
    return ExternalVaporPressureEvaluation(
        log_psat_kpa=torch.where(valid, log_psat, torch.zeros_like(log_psat)),
        valid=valid,
    )
