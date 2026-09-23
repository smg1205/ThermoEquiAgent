"""Regression tests for edge cases in thermo_engine identity, registry, and parameter handling."""

from __future__ import annotations

import pytest

from schemas.domain import ComponentIdentity, FailureType, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium
from thermo_engine.errors import ThermoEquiError
from thermo_engine.identity import resolve_external_component
from thermo_engine.parameters import reverse_binary_parameter_direction
from thermo_engine.registry import DEFAULT_BACKEND_REGISTRY


def test_reverse_binary_parameter_direction_raises_when_pair_is_incomplete() -> None:
    with pytest.raises(ValueError, match=r"Directional pair tau12/tau21 is incomplete"):
        reverse_binary_parameter_direction({"tau12": 0.5}, [("tau12", "tau21")])


def test_resolve_external_component_returns_none_for_unknown_identifier() -> None:
    assert resolve_external_component("unobtainium-123") is None


def test_calculate_equilibrium_rejects_mismatched_component_name_and_cas() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[
            ComponentIdentity(component_id="benzene", name="Toluene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[1.0, 0.0]),
        model_name="Ideal/Raoult",
    )

    with pytest.raises(ThermoEquiError) as exc_info:
        calculate_equilibrium(task)

    assert exc_info.value.detail.failure_type == FailureType.SEMANTIC_FAILURE
    assert "does not match declared CAS" in exc_info.value.detail.message


def test_registry_resolves_nrtl_request_as_missing_parameters() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name="NRTL",
    )

    with pytest.raises(ThermoEquiError) as exc_info:
        DEFAULT_BACKEND_REGISTRY.resolve(task)

    assert exc_info.value.detail.failure_type == FailureType.MISSING_PARAMETERS
    assert "NRTL" in exc_info.value.detail.message


def test_registry_route_task_selects_peng_robinson_for_non_registry_components() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
        ],
        conditions=ThermodynamicConditions(temperature_K=120.0, pressure_kPa=101.325, feed_composition=[0.5, 0.5]),
        model_name=None,
    )

    routed = DEFAULT_BACKEND_REGISTRY.route_task(task)

    assert routed.model_name == "Peng-Robinson"
    assert "Peng-Robinson" in routed.assumptions[0]


def test_registry_route_task_rejects_high_pressure_inapplicable_system() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="water", name="Water", cas_number="7732-18-5"),
            ComponentIdentity(component_id="methanol", name="Methanol", cas_number="67-56-1"),
        ],
        conditions=ThermodynamicConditions(temperature_K=450.0, pressure_kPa=1000.0, feed_composition=[0.5, 0.5]),
        model_name=None,
    )

    with pytest.raises(ThermoEquiError) as exc_info:
        DEFAULT_BACKEND_REGISTRY.route_task(task)

    assert exc_info.value.detail.failure_type == FailureType.PARAMETER_OUT_OF_DOMAIN
    assert "high-pressure" in exc_info.value.detail.message.lower()
