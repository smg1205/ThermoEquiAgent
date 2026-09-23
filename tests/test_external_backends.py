"""Behavioral contracts for optional Phasepy and Clapeyron.jl backends."""

from __future__ import annotations

import importlib.util
import os

import pytest

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError

METHANE = ComponentIdentity(component_id="methane", name="methane", cas_number="74-82-8")
ETHANE = ComponentIdentity(component_id="ethane", name="ethane", cas_number="74-84-0")
BENZENE = ComponentIdentity(component_id="benzene", name="benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="toluene", cas_number="108-88-3")
SODIUM_CHLORIDE = ComponentIdentity(
    component_id="sodium-chloride",
    name="sodium chloride",
    cas_number="7647-14-5",
)
WATER = ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5")


def _task(model_name: str, calculation_type: str = "tp_flash") -> TaskManifest:
    conditions_by_calculation = {
        "tp_flash": ThermodynamicConditions(
            temperature_K=150.0,
            pressure_kPa=530.0,
            feed_composition=[0.8, 0.2],
        ),
        "bubble_point": ThermodynamicConditions(
            pressure_kPa=530.0,
            liquid_composition=[0.5, 0.5],
        ),
        "dew_point": ThermodynamicConditions(
            pressure_kPa=530.0,
            vapor_composition=[0.987, 0.013],
        ),
        "isobaric_vle": ThermodynamicConditions(pressure_kPa=530.0),
        "isothermal_vle": ThermodynamicConditions(
            temperature_K=150.0,
        ),
        "azeotrope": ThermodynamicConditions(pressure_kPa=530.0),
    }
    conditions = conditions_by_calculation[calculation_type]
    return TaskManifest(
        equilibrium_type="FLASH" if calculation_type == "tp_flash" else "VLE",
        calculation_type=calculation_type,
        components=[METHANE, ETHANE],
        conditions=conditions,
        model_name=model_name,
        points=5,
    )


@pytest.mark.skipif(importlib.util.find_spec("phasepy") is not None, reason="dependency is installed")
def test_phasepy_selection_reports_the_exact_optional_dependency() -> None:
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(_task("Phasepy/Peng-Robinson"))

    assert captured.value.detail.failure_type == "unsupported_model"
    assert captured.value.detail.details["dependency"] == "phasepy==0.0.56"


@pytest.mark.skipif(importlib.util.find_spec("pyclapeyron") is not None, reason="dependency is installed")
def test_clapeyron_selection_reports_the_exact_optional_dependency() -> None:
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(_task("Clapeyron/Peng-Robinson"))

    assert captured.value.detail.failure_type == "unsupported_model"
    assert captured.value.detail.details["dependency"] == "pyclapeyron==0.1.1"


def test_component_name_and_cas_mismatch_is_rejected_before_backend_execution() -> None:
    task = _task("Clapeyron/Peng-Robinson").model_copy(
        update={
            "components": [
                ComponentIdentity(component_id="wrong", name="ethane", cas_number="74-82-8"),
                ETHANE,
            ]
        }
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "semantic_failure"
    assert captured.value.detail.details["declared_cas"] == "74-82-8"


def test_clapeyron_rejects_missing_reviewed_binary_parameters() -> None:
    task = _task("Clapeyron/Peng-Robinson").model_copy(update={"components": [BENZENE, TOLUENE]})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert captured.value.detail.details["parameter_table"] == "ChemSep PR"
    assert captured.value.detail.details["missing_pairs"] == [["71-43-2", "108-88-3"]]


@pytest.mark.parametrize("include_cas", [True, False])
def test_structured_electrolyte_manifest_is_rejected_at_the_shared_boundary(include_cas: bool) -> None:
    salt = SODIUM_CHLORIDE if include_cas else ComponentIdentity(component_id="sodium-chloride", name="sodium chloride")
    task = _task("Clapeyron/Peng-Robinson").model_copy(update={"components": [salt, WATER], "model_name": None})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
    assert captured.value.detail.details["electrolyte_components"] == ["sodium chloride"]


@pytest.mark.skipif(importlib.util.find_spec("phasepy") is None, reason="Phasepy optional dependency is absent")
def test_phasepy_flash_crosses_the_public_validation_gate() -> None:
    results = {
        calculation: calculate_equilibrium(_task("Phasepy/Peng-Robinson", calculation))
        for calculation in (
            "bubble_point",
            "dew_point",
            "isobaric_vle",
            "isothermal_vle",
            "tp_flash",
            "azeotrope",
        )
    }
    result = results["tp_flash"]
    reports = {calculation: validate_equilibrium_result(value) for calculation, value in results.items()}

    assert result.backend_version.startswith("phasepy/")
    assert result.model_name == "Phasepy/Peng-Robinson"
    assert result.phase_state == "two_phase"
    assert result.vapor_fraction == pytest.approx(0.631, abs=0.02)
    assert reports["tp_flash"].material_balance.passed
    assert all(report.equilibrium_residual.passed for report in reports.values())
    assert len(results["isobaric_vle"].points) == 5
    assert len(results["isothermal_vle"].points) == 5


@pytest.mark.skipif(
    importlib.util.find_spec("pyclapeyron") is None or os.getenv("RUN_CLAPEYRON_INTEGRATION") != "1",
    reason="Clapeyron.jl live integration is opt-in because Julia initialization is expensive",
)
def test_clapeyron_flash_and_bubble_cross_the_public_validation_gate() -> None:
    results = {
        calculation: calculate_equilibrium(_task("Clapeyron/Peng-Robinson", calculation))
        for calculation in (
            "bubble_point",
            "dew_point",
            "isobaric_vle",
            "isothermal_vle",
            "tp_flash",
            "azeotrope",
        )
    }
    flash = results["tp_flash"]
    reports = {calculation: validate_equilibrium_result(value) for calculation, value in results.items()}

    assert flash.backend_version.startswith("pyclapeyron/")
    assert flash.model_name == "Clapeyron/Peng-Robinson"
    assert flash.vapor_fraction == pytest.approx(0.626, abs=0.02)
    assert reports["tp_flash"].material_balance.passed
    assert all(report.equilibrium_residual.passed for report in reports.values())
    assert len(results["isobaric_vle"].points) == 5
    assert len(results["isothermal_vle"].points) == 5
