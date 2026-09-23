"""Behavioral tests for the ThermoFormer first-class backend.

ThermoFormer requires a trained checkpoint, the ThermoFormer source tree, torch/RDKit, so
these tests exercise the backend contract with mocked prediction internals and
verify structured failures when runtime prerequisites are absent.
"""

from __future__ import annotations

import shutil
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

import pytest

from schemas.domain import (
    ComponentIdentity,
    FailureType,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.thermoformer_backend import (
    ThermoFormerBackend,
    ThermoFormerSettings,
    _select_checkpoint,
    resolve_settings,
)
from thermo_engine.validation import validate_result

ETHANOL = ComponentIdentity(
    component_id="ethanol",
    name="ethanol",
    cas_number="64-17-5",
    smiles="CCO",
    aliases=[],
)
WATER = ComponentIdentity(
    component_id="water",
    name="water",
    cas_number="7732-18-5",
    smiles="O",
    aliases=[],
)
METHANOL = ComponentIdentity(
    component_id="methanol",
    name="methanol",
    cas_number="67-56-1",
    smiles="CO",
    aliases=[],
)


def _vle_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(
            temperature_K=298.15,
            liquid_composition=[0.3, 0.7],
        ),
        model_name="ThermoFormer",
        points=5,
    )


def _gamma_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="infinite_dilution_activity",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        model_name="ThermoFormer",
    )


def _ternary_vle_task() -> TaskManifest:
    return _vle_task().model_copy(
        update={
            "components": [ETHANOL, WATER, METHANOL],
            "conditions": ThermodynamicConditions(
                temperature_K=298.15,
                liquid_composition=[0.2, 0.5, 0.3],
            ),
        }
    )


def _lle_task(component_count: int = 2) -> TaskManifest:
    components = [ETHANOL, WATER] if component_count == 2 else [ETHANOL, WATER, METHANOL]
    return TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=components,
        conditions=ThermodynamicConditions(
            temperature_K=298.15,
            pressure_kPa=101.325,
        ),
        model_name="ThermoFormer",
    )


def _catalog_settings(tmp_path: Path) -> ThermoFormerSettings:
    src = tmp_path / "src"
    src.mkdir()
    models = tmp_path / "models"
    checkpoints = {
        "models/vle/prediction/vle_overall_binary/seed_0/best_model.pt": (
            "vle",
            "vle_overall_binary",
        ),
        "models/vle/prediction/vle_overall_ternary/seed_0/best_model.pt": (
            "vle",
            "vle_overall_ternary",
        ),
        "models/lle/prediction/binary-system/seed_0/best.pt": (
            "lle",
            "binary-system",
        ),
        "models/lle/prediction/ternary-system/seed_0/best.pt": (
            "lle",
            "ternary-system",
        ),
    }
    rows = []
    for relative, (task, protocol) in checkpoints.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        rows.append(
            {
                "task": task,
                "protocol": protocol,
                "seed": 0,
                "path": relative,
                "status": "provided",
            }
        )
    models.mkdir(exist_ok=True)
    (models / "registry.json").write_text(
        '{"checkpoints": ' + __import__("json").dumps(rows) + "}",
        encoding="utf-8",
    )
    return ThermoFormerSettings(
        src_path=src,
        checkpoint_path=None,
        feature_cache_path=tmp_path / "cache",
        use_cuda=False,
    )


def _new_catalog_root() -> Path:
    root = Path.cwd() / f".test_thermoformer_catalog_{uuid4().hex}"
    root.mkdir()
    return root


# ── scope checks ──────────────────────────────────────────────────────────


def test_too_many_components_fails() -> None:
    """ThermoFormer supports up to 3 components."""
    components = [
        ComponentIdentity(component_id=f"c{i}", name=f"c{i}", smiles=f"C{i}H", aliases=[])
        for i in range(4)
    ]
    task = _vle_task().model_copy(update={"components": components})
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL
    assert "3" in captured.value.detail.message or "components" in captured.value.detail.message


def test_too_high_pressure_fails() -> None:
    """ThermoFormer is validated up to 500 kPa."""
    task = _vle_task().model_copy(
        update={"conditions": ThermodynamicConditions(pressure_kPa=600.0, liquid_composition=[0.3, 0.7])}
    )
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.PARAMETER_OUT_OF_DOMAIN


def test_missing_smiles_fails() -> None:
    """SMILES is required for every component."""
    task = _vle_task().model_copy(
        update={
            "components": [
                ETHANOL.model_copy(update={"smiles": None}),
                WATER,
            ]
        }
    )
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.bubble_point(task)
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS
    assert "SMILES" in captured.value.detail.message


def test_missing_src_env_fails_when_default_src_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_settings() raises MISSING_PARAMETERS when no source tree can be resolved."""
    import os
    os.environ.pop("THERMOFORMER_CHECKPOINT", None)
    monkeypatch.setenv("THERMOFORMER_SRC", str(Path("missing_src")))
    with pytest.raises(ThermoEquiError) as captured:
        resolve_settings()
    assert captured.value.detail.failure_type == FailureType.MISSING_PARAMETERS


def test_selects_vle_binary_and_ternary_checkpoints() -> None:
    root = _new_catalog_root()
    settings = _catalog_settings(root)

    try:
        binary = _select_checkpoint(settings, _vle_task())
        ternary = _select_checkpoint(settings, _ternary_vle_task())

        assert "vle_overall_binary" in str(binary)
        assert "vle_overall_ternary" in str(ternary)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_selects_lle_binary_and_ternary_checkpoints() -> None:
    root = _new_catalog_root()
    settings = _catalog_settings(root)

    try:
        binary = _select_checkpoint(settings, _lle_task(2))
        ternary = _select_checkpoint(settings, _lle_task(3))

        assert "binary-system" in str(binary)
        assert "ternary-system" in str(ternary)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── unsupported operations ────────────────────────────────────────────────


def test_unsupported_dew_point_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.dew_point(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_unsupported_tp_flash_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.tp_flash(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_unsupported_azeotrope_fails() -> None:
    settings = ThermoFormerSettings(
        src_path=Path("unused"),
        checkpoint_path=Path("unused.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    with pytest.raises(ThermoEquiError) as captured:
        backend.azeotrope(_vle_task())
    assert captured.value.detail.failure_type == FailureType.UNSUPPORTED_MODEL


def test_lle_uses_lle_predictor_and_returns_phase_endpoints() -> None:
    class FakeLLEPredictor:
        def predict_lle(self, smiles: list[str], temperature_k: float, pressure_kpa: float) -> dict[str, object]:
            return {
                "status": "converged",
                "pairs": [
                    {
                        "endpoints": [[0.1, 0.9], [0.8, 0.2]],
                        "equilibrium_rms": 1e-4,
                    }
                ],
                "attempts": 1,
            }

    root = _new_catalog_root()
    try:
        backend = ThermoFormerBackend(settings=_catalog_settings(root))
        with patch.object(backend, "_get_lle_predictor", return_value=FakeLLEPredictor()):
            result = backend.lle(_lle_task(2))

        assert result.model_name == "ThermoFormer"
        assert result.calculation_type == "lle"
        assert len(result.phases) == 2
        assert result.phases[0].composition == [0.1, 0.9]
        assert validate_result(result).equilibrium_residual.passed is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── composition sweep helper ──────────────────────────────────────────────


def test_build_composition_sweep_binary() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(2, 5)
    assert len(comps) == 3  # n_points=5 → 3 interior points
    for c in comps:
        assert len(c) == 2
        assert abs(sum(c) - 1.0) < 1e-10
        assert all(x > 0.0 for x in c)


def test_build_composition_sweep_ternary() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(3, 10)
    assert len(comps) > 0
    for c in comps:
        assert len(c) == 3
        assert abs(sum(c) - 1.0) < 1e-10
        assert all(x > 0.0 for x in c)


def test_build_composition_sweep_unary_returns_empty() -> None:
    from thermo_engine.thermoformer_backend import _build_composition_sweep

    comps = _build_composition_sweep(1, 5)
    assert comps == []


# ── parameter sources ─────────────────────────────────────────────────────


def test_parameter_sources_are_structured() -> None:
    settings = ThermoFormerSettings(
        src_path=Path("model_src"),
        checkpoint_path=Path("model.pt"),
        feature_cache_path=Path(".cache"),
        use_cuda=False,
    )
    backend = ThermoFormerBackend(settings=settings)
    sources = backend.parameter_sources(_vle_task())
    assert sources
    assert "checkpoint" in sources[0]
    assert sources[0]["source_type"] == "model_prediction"
