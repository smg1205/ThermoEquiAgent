"""Tests for TaskManifest calculation type normalization and alias handling."""

from __future__ import annotations

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions


def test_task_manifest_normalizes_t_x_y_to_isobaric_vle() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="T-X-Y",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325),
    )
    assert task.calculation_type == "isobaric_vle"


def test_task_manifest_normalizes_p_x_y_to_isothermal_vle() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="p-x-y",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(temperature_K=350.0),
    )
    assert task.calculation_type == "isothermal_vle"


def test_task_manifest_normalizes_flash_aliases_to_tp_flash() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
        ],
        conditions=ThermodynamicConditions(temperature_K=120.0, pressure_kPa=101.325, feed_composition=[0.5, 0.5]),
    )
    assert task.calculation_type == "tp_flash"
