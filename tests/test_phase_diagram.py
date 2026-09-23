"""Behavioral tests for the multi-model phase diagram backend seam.

Covers both the public API endpoint and the ``build_phase_diagram`` builder
that the web frontend's ``POST /api/calculations/phase-diagram`` call depends
on.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import apps.api.main as api_module
from agent.comparison import build_phase_diagram
from database.session import Repository, initialize_database
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions

BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3")


def _isobaric_vle_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(pressure_kPa=101.325),
        points=5,
    )


def _isothermal_vle_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isothermal_vle",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(temperature_K=353.15),
        points=5,
    )


def test_phase_diagram_reports_txy_with_counts_for_isobaric_vle() -> None:
    response = build_phase_diagram(_isobaric_vle_task())

    assert response.diagram_type == "TXY"
    assert response.total_models == len(response.entries)
    assert response.total_models >= 2
    assert response.summary
    assert (
        response.passed_count + response.warning_count + response.failed_count + response.unsupported_count
        == response.total_models
    )
    # Every entry exposes the fields the frontend renders.
    for entry in response.entries:
        assert entry.status in {"passed", "warning", "failed", "unsupported"}
        assert isinstance(entry.executable, bool)
        if entry.status == "unsupported":
            assert not entry.executable
            assert entry.result is None
            assert entry.failure is not None


def test_phase_diagram_maps_isothermal_vle_to_pxy() -> None:
    response = build_phase_diagram(_isothermal_vle_task())
    assert response.diagram_type == "PXY"


def test_phase_diagram_includes_ideal_raoult_and_unsupported_models() -> None:
    response = build_phase_diagram(_isobaric_vle_task())

    assert any(entry.model_name == "Ideal/Raoult" for entry in response.entries)
    assert any(entry.status in {"passed", "warning"} for entry in response.entries)
    # At least one catalog model cannot produce a VLE curve, so it must surface
    # as unsupported rather than silently disappearing.
    assert any(entry.status == "unsupported" for entry in response.entries)


def api_client() -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    api_module.repository = Repository(engine)

    @asynccontextmanager
    async def no_op_lifespan(_):  # type: ignore[no-untyped-def]
        yield

    api_module.app.router.lifespan_context = no_op_lifespan
    return TestClient(api_module.app)


def test_phase_diagram_endpoint_returns_phase_diagram_response() -> None:
    with api_client() as client:
        response = client.post(
            "/api/calculations/phase-diagram",
            json=_isobaric_vle_task().model_dump(mode="json"),
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["diagram_type"] == "TXY"
        assert payload["total_models"] == len(payload["entries"])
        assert any(entry["model_name"] == "Ideal/Raoult" for entry in payload["entries"])
        assert any(entry["status"] == "unsupported" for entry in payload["entries"])
        for entry in payload["entries"]:
            assert entry["status"] in {"passed", "warning", "failed", "unsupported"}
