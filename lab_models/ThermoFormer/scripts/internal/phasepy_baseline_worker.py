"""Standalone Phasepy kernel used by the ThermoFormer baseline adapter.

This file intentionally imports no ThermoFormer package modules and therefore
does not load PyTorch in the Phasepy process.
"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from phasepy import component, mixture, virialgamma
from phasepy.actmodels import nrtl, uniquac, wilson
from phasepy.equilibrium import bubblePy, bubbleTy
from scipy.optimize import least_squares
from thermo import Chemical


class ValidatedPsat:
    def __init__(self, record: dict[str, Any]) -> None:
        self.cas = str(record["cas"])
        self.name = str(record["name"])
        self.method = str(record["method"])
        self.minimum_temperature_k = float(record["minimum_temperature_k"])
        self.maximum_temperature_k = float(record["maximum_temperature_k"])
        self.correlation = Chemical(self.cas).VaporPressure

    def pressure_kpa(self, temperature_k: float) -> float:
        temperature = float(temperature_k)
        if not self.minimum_temperature_k <= temperature <= self.maximum_temperature_k:
            raise ValueError(
                f"{self.name} {self.method} outside audited range at {temperature:.6g} K"
            )
        pressure = float(self.correlation.calculate(temperature, self.method)) / 1000.0
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ValueError(f"{self.name} {self.method} returned nonpositive pressure")
        return pressure


def phase_component(provider: ValidatedPsat, size: list[float] | None) -> component:
    chemical = Chemical(provider.cas)
    required = [chemical.Tc, chemical.Pc, chemical.Zc, chemical.Vc, chemical.omega, chemical.MW]
    if any(value is None for value in required):
        raise ValueError(f"Phasepy pure constants unavailable for {provider.name}")
    values = np.asarray(required, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"Phasepy pure constants are nonphysical for {provider.name}")
    ri, qi = size if size is not None else (0.0, 0.0)
    return component(
        name=provider.name,
        Tc=float(chemical.Tc),
        Pc=float(chemical.Pc) / 1e5,
        Zc=float(chemical.Zc),
        Vc=float(chemical.Vc) * 1e6,
        w=float(chemical.omega),
        Mw=float(chemical.MW),
        ri=float(ri),
        qi=float(qi),
        Ant=[0.0, 0.0, 0.0],
    )


def interaction_matrix(values: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=float)
    cursor = 0
    for left in range(count):
        for right in range(count):
            if left != right:
                matrix[left, right] = float(values[cursor])
                cursor += 1
    return matrix


def log_gamma(
    model_name: str,
    x: np.ndarray,
    temperature_k: float,
    interaction: np.ndarray,
    phase_mixture: mixture,
    alpha_value: float,
    sizes: np.ndarray | None,
) -> np.ndarray:
    if model_name == "nrtl":
        alpha = np.full_like(interaction, alpha_value)
        np.fill_diagonal(alpha, 0.0)
        return np.asarray(
            nrtl(x, temperature_k, alpha, interaction, np.zeros_like(interaction)),
            dtype=float,
        )
    if model_name == "wilson":
        return np.asarray(
            wilson(x, temperature_k, interaction, phase_mixture.vlrackett), dtype=float
        )
    if sizes is None:
        raise ValueError("Phasepy UNIQUAC requires r and q")
    return np.asarray(
        uniquac(
            x,
            temperature_k,
            sizes[:, 0],
            sizes[:, 1],
            interaction,
            np.zeros_like(interaction),
        ),
        dtype=float,
    )


def configured_model(
    model_name: str,
    phase_mixture: mixture,
    interaction: np.ndarray,
    providers: list[ValidatedPsat],
    alpha_value: float,
) -> Any:
    if model_name == "nrtl":
        alpha = np.full_like(interaction, alpha_value)
        np.fill_diagonal(alpha, 0.0)
        phase_mixture.NRTL(alpha, interaction)
    elif model_name == "wilson":
        phase_mixture.wilson(interaction)
    else:
        phase_mixture.uniquac(interaction)
    model = virialgamma(phase_mixture, virialmodel="ideal_gas", actmodel=model_name)

    def psat_bar(temperature_k: float) -> np.ndarray:
        return np.asarray(
            [provider.pressure_kpa(float(temperature_k)) / 100.0 for provider in providers]
        )

    model.psat = psat_bar
    return model


def directions(mode: str) -> tuple[str, ...]:
    if mode == "isothermal":
        return ("isothermal",)
    if mode == "isobaric":
        return ("isobaric",)
    return ("isothermal", "isobaric")


def run(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = str(payload["model"])
    components = tuple(str(value) for value in payload["components"])
    provider_map = {
        key: ValidatedPsat(value) for key, value in payload["pure_properties"].items()
    }
    providers = [provider_map[value] for value in components]
    size_map = payload.get("uniquac_sizes", {})
    size_rows = [size_map.get(value) for value in components]
    if model_name == "uniquac" and any(value is None for value in size_rows):
        raise ValueError("Phasepy UNIQUAC size parameters are unavailable")
    sizes = np.asarray(size_rows, dtype=float) if model_name == "uniquac" else None
    phase_components = [
        phase_component(provider, size_map.get(value))
        for value, provider in zip(components, providers)
    ]
    phase_mixture = mixture(phase_components[0], phase_components[1])
    for value in phase_components[2:]:
        phase_mixture.add_component(value)
    alpha_value = float(payload["nrtl_alpha"])
    fit_rows = payload["fit_rows"]
    parameter_count = len(components) * (len(components) - 1)

    def residual(values: np.ndarray) -> np.ndarray:
        interaction = interaction_matrix(values, len(components))
        parts = []
        try:
            for row in fit_rows:
                x = np.asarray(row["x"], dtype=float)
                target_y = np.asarray(row["y"], dtype=float)
                temperature = float(row["temperature_k"])
                psat = np.asarray(
                    [provider.pressure_kpa(temperature) for provider in providers]
                )
                gamma = np.exp(
                    np.clip(
                        log_gamma(
                            model_name,
                            x,
                            temperature,
                            interaction,
                            phase_mixture,
                            alpha_value,
                            sizes,
                        ),
                        -30.0,
                        30.0,
                    )
                )
                contributions = x * gamma * psat
                pressure = float(contributions.sum())
                vapor = contributions / max(pressure, 1e-15)
                quality = math.sqrt(max(float(row["quality_weight"]), 1e-12))
                parts.append(
                    quality
                    * np.concatenate(
                        [
                            [math.log(max(pressure, 1e-15) / float(row["pressure_kpa"]))],
                            (vapor - target_y) / math.sqrt(len(components)),
                        ]
                    )
                )
        except (ArithmeticError, FloatingPointError, ValueError):
            return np.full(len(fit_rows) * (len(components) + 1), 1e6)
        return np.nan_to_num(np.concatenate(parts), nan=1e6, posinf=1e6, neginf=-1e6)

    initial_values = np.zeros(parameter_count)
    initial_loss = float(np.mean(residual(initial_values) ** 2))
    optimized = least_squares(
        residual,
        initial_values,
        bounds=(-5000.0, 5000.0),
        max_nfev=int(payload["maximum_evaluations"]),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    interaction = interaction_matrix(optimized.x, len(components))
    final_loss = float(np.mean(residual(optimized.x) ** 2))
    success = bool(
        optimized.status > 0
        and math.isfinite(final_loss)
        and np.isfinite(interaction).all()
        and final_loss <= initial_loss + max(1e-12, initial_loss * 1e-8)
    )
    predictions = []
    if success:
        equilibrium_model = configured_model(
            model_name, phase_mixture, interaction, providers, alpha_value
        )
        lower, upper = map(float, payload["temperature_bounds_k"])
        for row in payload["prediction_rows"]:
            x = np.asarray(row["x"], dtype=float)
            for direction in directions(str(row["experiment_mode"])):
                try:
                    if direction == "isothermal":
                        temperature = float(row["temperature_k"])
                        guess = max(float(np.dot(x, equilibrium_model.psat(temperature))), 1e-6)
                        output = bubblePy(
                            x.copy(), guess, x, temperature, equilibrium_model, full_output=True
                        )
                        pressure = float(output.P) * 100.0
                    else:
                        pressure = float(row["pressure_kpa"])
                        guess = min(max(float(row["temperature_k"]), lower), upper)
                        output = bubbleTy(
                            x.copy(),
                            guess,
                            x,
                            pressure / 100.0,
                            equilibrium_model,
                            full_output=True,
                        )
                        temperature = float(output.T)
                        if not lower <= temperature <= upper:
                            raise ValueError("Phasepy bubble temperature outside audited range")
                    vapor = np.asarray(output.Y, dtype=float)
                    gamma = np.exp(
                        np.clip(equilibrium_model.lngama(x, temperature), -30.0, 30.0)
                    )
                    psat = np.asarray(equilibrium_model.psat(temperature)) * 100.0
                    error = abs(float(output.error))
                    converged = bool(
                        np.isfinite([temperature, pressure, error]).all()
                        and np.isfinite(vapor).all()
                        and error <= 1e-6
                    )
                    predictions.append(
                        {
                            "direction": direction,
                            "temperature_k": temperature,
                            "pressure_kpa": pressure,
                            "vapor_composition": vapor.tolist(),
                            "activity_coefficients": np.asarray(gamma).tolist(),
                            "vapor_pressures_kpa": psat.tolist(),
                            "converged": converged,
                            "residual_kpa": error * max(pressure, 1.0),
                            "iterations": int(output.iter),
                            "failure_reason": None if converged else "phasepy_solver_not_converged",
                        }
                    )
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    predictions.append(
                        {
                            "direction": direction,
                            "converged": False,
                            "failure_reason": "phasepy_prediction_error:"
                            + str(error).splitlines()[0][:120],
                        }
                    )
    return {
        "phasepy_version": phasepy_version_value(),
        "success": success,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "optimizer_status": int(optimized.status),
        "optimizer_message": str(optimized.message),
        "interaction_k": interaction.tolist(),
        "predictions": predictions,
    }


def phasepy_version_value() -> str:
    try:
        return version("phasepy")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = run(payload)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output_path)


if __name__ == "__main__":
    main()
