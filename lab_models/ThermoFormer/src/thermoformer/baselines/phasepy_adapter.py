"""Process-isolated adapter for Phasepy activity models and VLE solvers.

On Windows, Phasepy's compiled activity-model extensions and PyTorch can load
different OpenMP runtimes. ThermoFormer therefore executes the Phasepy kernel
in a clean subprocess that never imports PyTorch. Inputs and outputs are plain
JSON, and the validated external vapor-pressure identities are preserved.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import VLESample
from ..data.splitting import canonical_smiles
from .thermodynamic_evaluation import _directions, _record
from .thermodynamic_models import ActivityModelName, BubbleState


@dataclass(frozen=True)
class PhasepyFittedActivityModel:
    """System-specific fit and cached Phasepy prediction states."""

    model: ActivityModelName
    components: tuple[str, ...]
    interaction_k: np.ndarray
    success: bool
    training_samples: int
    initial_training_loss: float
    final_training_loss: float
    optimizer_status: int
    optimizer_message: str
    nrtl_alpha: float
    phasepy_version: str
    prediction_states: tuple[dict[str, Any], ...]


def _ordered_sample(
    sample: VLESample, components: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    positions = {canonical_smiles(value): index for index, value in enumerate(sample.smiles)}
    if set(positions) != set(components):
        raise ValueError("sample components do not match the fitted system")
    order = [positions[value] for value in components]
    return (
        np.asarray([sample.liquid_composition[index] for index in order], dtype=float),
        np.asarray([sample.vapor_composition[index] for index in order], dtype=float),
    )


def _sample_payload(sample: VLESample, components: tuple[str, ...]) -> dict[str, Any]:
    x, y = _ordered_sample(sample, components)
    return {
        "temperature_k": float(sample.temperature_k),
        "pressure_kpa": float(sample.pressure_kpa),
        "x": x.tolist(),
        "y": y.tolist(),
        "quality_weight": float(sample.quality_weight),
        "experiment_mode": str(sample.experiment_mode),
    }


def _worker_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "internal" / "phasepy_baseline_worker.py"
    if not path.is_file():
        raise FileNotFoundError(f"Phasepy worker is missing: {path}")
    return path


def phasepy_version() -> str:
    """Return the installed Phasepy distribution version."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("phasepy")
    except PackageNotFoundError:
        return "unknown"


def fit_phasepy_activity_model(
    model: ActivityModelName,
    samples: Sequence[VLESample],
    vapor_pressure: Mapping[str, Any],
    *,
    prediction_samples: Sequence[VLESample] | None = None,
    temperature_bounds_k: tuple[float, float] = (150.0, 1500.0),
    maximum_evaluations: int = 200,
    nrtl_alpha: float = 0.3,
    uniquac_sizes: Mapping[str, tuple[float, float]] | None = None,
) -> PhasepyFittedActivityModel:
    """Fit and predict with Phasepy in a PyTorch-free subprocess."""

    if not samples:
        raise ValueError("At least one fitting sample is required")
    components = tuple(sorted(canonical_smiles(value) for value in samples[0].smiles))
    if any(
        tuple(sorted(canonical_smiles(value) for value in sample.smiles)) != components
        for sample in samples
    ):
        raise ValueError("All Phasepy fitting rows must belong to one chemical system")
    missing = [value for value in components if value not in vapor_pressure]
    if missing:
        raise ValueError(f"Missing validated vapor pressure for: {', '.join(missing)}")
    prediction_rows = tuple(prediction_samples if prediction_samples is not None else samples)
    pure_properties = {}
    for value in components:
        provider = vapor_pressure[value]
        pure_properties[value] = {
            "cas": str(provider.cas),
            "name": str(provider.name),
            "method": str(provider.method),
            "minimum_temperature_k": float(provider.minimum_temperature_k),
            "maximum_temperature_k": float(provider.maximum_temperature_k),
        }
    payload = {
        "model": model,
        "components": list(components),
        "pure_properties": pure_properties,
        "uniquac_sizes": {
            key: [float(value[0]), float(value[1])]
            for key, value in (uniquac_sizes or {}).items()
            if key in components
        },
        "nrtl_alpha": float(nrtl_alpha),
        "maximum_evaluations": int(maximum_evaluations),
        "temperature_bounds_k": [float(value) for value in temperature_bounds_k],
        "fit_rows": [_sample_payload(sample, components) for sample in samples],
        "prediction_rows": [
            _sample_payload(sample, components) for sample in prediction_rows
        ],
    }
    with tempfile.TemporaryDirectory(prefix="thermoformer_phasepy_") as directory:
        root = Path(directory)
        input_path = root / "input.json"
        output_path = root / "output.json"
        input_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        environment = dict(os.environ)
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        environment["OMP_NUM_THREADS"] = "1"
        completed = subprocess.run(
            [sys.executable, str(_worker_path()), str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1][:240] if detail else "unknown worker error"
            raise RuntimeError(f"Phasepy worker failed: {message}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
    interaction = np.asarray(result["interaction_k"], dtype=float)
    return PhasepyFittedActivityModel(
        model=model,
        components=components,
        interaction_k=interaction,
        success=bool(result["success"]),
        training_samples=len(samples),
        initial_training_loss=float(result["initial_training_loss"]),
        final_training_loss=float(result["final_training_loss"]),
        optimizer_status=int(result["optimizer_status"]),
        optimizer_message=str(result["optimizer_message"]),
        nrtl_alpha=nrtl_alpha,
        phasepy_version=str(result["phasepy_version"]),
        prediction_states=tuple(result["predictions"]),
    )


def predict_phasepy_activity_model(
    samples: Sequence[VLESample],
    fitted: PhasepyFittedActivityModel,
    vapor_pressure: Mapping[str, Any],
    *,
    temperature_bounds_k: tuple[float, float],
) -> list[dict[str, object]]:
    """Convert cached worker states to the repository prediction schema."""

    del vapor_pressure, temperature_bounds_k
    expected = sum(len(_directions(sample)) for sample in samples)
    if len(fitted.prediction_states) != expected:
        raise ValueError("Phasepy worker prediction count does not match requested rows")
    records: list[dict[str, object]] = []
    cursor = 0
    for sample in samples:
        x, _ = _ordered_sample(sample, fitted.components)
        for direction in _directions(sample):
            item = fitted.prediction_states[cursor]
            cursor += 1
            if str(item["direction"]) != direction:
                raise ValueError("Phasepy worker prediction direction changed")
            if not item["converged"]:
                records.append(
                    _record(
                        sample,
                        direction,
                        fitted.model,
                        None,
                        str(item.get("failure_reason") or "phasepy_solver_not_converged"),
                    )
                )
                continue
            state = BubbleState(
                temperature_k=float(item["temperature_k"]),
                pressure_kpa=float(item["pressure_kpa"]),
                liquid_composition=x,
                vapor_composition=np.asarray(item["vapor_composition"], dtype=float),
                activity_coefficients=np.asarray(item["activity_coefficients"], dtype=float),
                vapor_pressures_kpa=np.asarray(item["vapor_pressures_kpa"], dtype=float),
                converged=True,
                residual_kpa=float(item["residual_kpa"]),
                iterations=int(item["iterations"]),
                failure_reason=None,
            )
            records.append(_record(sample, direction, fitted.model, state))
    return records
