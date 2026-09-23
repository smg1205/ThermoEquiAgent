"""Validated external pure-component vapor-pressure correlations for baselines.

The classical activity-coefficient comparisons must not infer pure properties from
mixture endpoints.  This module selects a published correlation exposed by the
``thermo`` property database, enforces its stated temperature interval, converts
the native pressure unit (Pa) to kPa, and rejects non-monotone curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Mapping

import numpy as np
from thermo import Chemical


METHOD_PRIORITY = (
    "DIPPR_PERRY_8E",
    "ANTOINE_POLING",
    "ANTOINE_WEBBOOK",
    "WAGNER_POLING",
    "WAGNER_MCGARRY",
    "VDI_PPDS",
    "HEOS_FIT",
)

METHOD_FAMILY = {
    "DIPPR_PERRY_8E": "DIPPR 101",
    "ANTOINE_POLING": "Antoine (Poling)",
    "ANTOINE_WEBBOOK": "Antoine (NIST WebBook)",
    "WAGNER_POLING": "Wagner (Poling)",
    "WAGNER_MCGARRY": "Wagner (McGarry)",
    "VDI_PPDS": "VDI PPDS",
    "HEOS_FIT": "HEOS fit",
}


@dataclass(frozen=True)
class ValidatedExternalVaporPressure:
    """One correlation with explicit units, validity range and monotonicity."""

    smiles: str
    name: str
    cas: str
    method: str
    family: str
    minimum_temperature_k: float
    maximum_temperature_k: float
    minimum_derivative_kpa_per_k: float
    _correlation: object = field(repr=False, compare=False)

    def pressure_kpa(self, temperature_k: float) -> float:
        temperature = float(temperature_k)
        if not self.minimum_temperature_k <= temperature <= self.maximum_temperature_k:
            raise ValueError(
                f"{self.name} {self.method} is invalid at {temperature:.6g} K; "
                f"valid range is [{self.minimum_temperature_k:.6g}, "
                f"{self.maximum_temperature_k:.6g}] K"
            )
        pressure_pa = float(self._correlation.calculate(temperature, self.method))
        pressure_kpa = pressure_pa / 1000.0
        if not math.isfinite(pressure_kpa) or pressure_kpa <= 0.0:
            raise ValueError(f"{self.name} {self.method} returned nonpositive pressure")
        return pressure_kpa

    def parameter_dict(self) -> dict[str, object]:
        """Export the selected correlation parameters through a stable adapter."""
        correlations = getattr(self._correlation, "correlations", {})
        raw = correlations.get(self.method)
        if raw is None:
            raise ValueError(f"Correlation parameters are unavailable for {self.method}")
        return dict(raw[1])

    def audit_record(self, required_range_k: tuple[float, float]) -> dict[str, object]:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "cas": self.cas,
            "status": "covered",
            "correlation_family": self.family,
            "correlation_method": self.method,
            "source": "thermo pure-component correlation database",
            "library": "thermo",
            "library_version": version("thermo"),
            "native_pressure_unit": "Pa",
            "output_pressure_unit": "kPa",
            "temperature_unit": "K",
            "valid_temperature_range_k": [
                self.minimum_temperature_k,
                self.maximum_temperature_k,
            ],
            "required_temperature_range_k": list(required_range_k),
            "minimum_dpsat_dt_kpa_per_k": self.minimum_derivative_kpa_per_k,
            "monotonic_in_required_range": True,
        }


def _curve_diagnostics(
    correlation: object,
    method: str,
    required_range_k: tuple[float, float],
    *,
    points: int = 257,
) -> tuple[bool, str, float | None]:
    lower, upper = map(float, required_range_k)
    grid = np.linspace(lower, upper, points)
    try:
        pressure_kpa = np.asarray(
            [float(correlation.calculate(float(value), method)) / 1000.0 for value in grid]
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return False, f"evaluation_error:{str(error).splitlines()[0][:120]}", None
    if not np.isfinite(pressure_kpa).all() or np.any(pressure_kpa <= 0.0):
        return False, "nonpositive_or_nonfinite_pressure", None
    derivative = np.diff(pressure_kpa) / np.diff(grid)
    minimum = float(derivative.min())
    if not np.isfinite(derivative).all() or np.any(derivative <= 0.0):
        return False, "nonpositive_dpsat_dt", minimum
    return True, "accepted", minimum


def validated_external_vapor_pressures(
    component_names: Mapping[str, str],
    required_ranges_k: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, ValidatedExternalVaporPressure], dict[str, object]]:
    """Return covered correlations and a complete rejection/coverage audit."""

    correlations: dict[str, ValidatedExternalVaporPressure] = {}
    records: dict[str, dict[str, object]] = {}
    for smiles, name in sorted(component_names.items()):
        required = tuple(map(float, required_ranges_k[smiles]))
        rejected: list[dict[str, object]] = []
        try:
            chemical = Chemical(name)
            vapor_pressure = chemical.VaporPressure
        except (KeyError, TypeError, ValueError) as error:
            records[smiles] = {
                "smiles": smiles,
                "name": name,
                "status": "unavailable",
                "reason": f"chemical_lookup_error:{str(error).splitlines()[0][:120]}",
                "required_temperature_range_k": list(required),
            }
            continue
        accepted: ValidatedExternalVaporPressure | None = None
        for method in METHOD_PRIORITY:
            if method not in vapor_pressure.all_methods:
                rejected.append({"method": method, "reason": "method_unavailable"})
                continue
            limits = vapor_pressure.T_limits.get(method)
            if limits is None:
                rejected.append({"method": method, "reason": "missing_validity_range"})
                continue
            minimum_temperature, maximum_temperature = map(float, limits)
            if required[0] < minimum_temperature or required[1] > maximum_temperature:
                rejected.append(
                    {
                        "method": method,
                        "reason": "required_range_outside_validity",
                        "valid_temperature_range_k": [
                            minimum_temperature,
                            maximum_temperature,
                        ],
                    }
                )
                continue
            valid, reason, minimum_derivative = _curve_diagnostics(
                vapor_pressure, method, required
            )
            if not valid or minimum_derivative is None:
                rejected.append(
                    {
                        "method": method,
                        "reason": reason,
                        "minimum_dpsat_dt_kpa_per_k": minimum_derivative,
                    }
                )
                continue
            accepted = ValidatedExternalVaporPressure(
                smiles=smiles,
                name=name,
                cas=str(chemical.CAS),
                method=method,
                family=METHOD_FAMILY[method],
                minimum_temperature_k=minimum_temperature,
                maximum_temperature_k=maximum_temperature,
                minimum_derivative_kpa_per_k=minimum_derivative,
                _correlation=vapor_pressure,
            )
            break
        if accepted is None:
            records[smiles] = {
                "smiles": smiles,
                "name": name,
                "cas": str(chemical.CAS),
                "status": "unavailable",
                "reason": "no_valid_monotone_external_correlation",
                "required_temperature_range_k": list(required),
                "rejected_methods": rejected,
            }
            continue
        correlations[smiles] = accepted
        records[smiles] = {
            **accepted.audit_record(required),
            "rejected_higher_priority_methods": rejected,
        }
    covered = len(correlations)
    return correlations, {
        "policy": (
            "Select the first external DIPPR/Antoine/Wagner/VDI/HEOS correlation "
            "whose documented temperature range covers every fit and solver temperature; "
            "require finite positive pressure and dPsat/dT > 0 throughout that range."
        ),
        "method_priority": list(METHOD_PRIORITY),
        "native_pressure_unit": "Pa",
        "output_pressure_unit": "kPa",
        "temperature_unit": "K",
        "components": len(component_names),
        "covered_components": covered,
        "component_coverage": covered / len(component_names) if component_names else 0.0,
        "entries": records,
    }
