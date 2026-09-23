"""Behavioral tests for the generic ternary liquid-liquid DWSIM export.

Generalises the binary ``Vessel_LLE`` template to any three named components.
On DWSIM 9.0.5 the ternary split does not resolve through the Automation API, so
the rendered file is structurally correct but its heavy phase is typically empty;
the payload must say so rather than present it as a working extraction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

import agent.extractive_distillation as extractive
from agent.extractive_distillation import (
    _ternary_lle_components,
    is_binary_lle_dwsim_request,
    is_generic_ternary_lle_request,
    is_lle_extraction_request,
)
from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from schemas.domain import Intent
from tests.test_binary_lle_dwsim import FakeAutomation, FakeObjectType

_TERNARY = (
    "\u4e59\u9187 \u4e59\u9178\u4e59\u916f \u6c34 \u4e09\u5143\u6db2\u6db2\u8403\u53d6\uff0c"
    "\u7ec4\u6210 0.129/0.188/0.683\uff0c298.15 K\uff0c\u5bfc\u51fa dwsim"
)
_NON_WHITELISTED = (
    "\u7532\u9187 \u7532\u82ef \u6c34 \u4e09\u5143\u6db2\u6db2\u5e73\u8861\uff0c"
    "\u5bfc\u51fadwsim\uff0c0.3/0.3/0.4"
)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_generic_ternary_hits_for_any_three_named_components() -> None:
    assert is_generic_ternary_lle_request(_TERNARY)
    assert is_generic_ternary_lle_request(_NON_WHITELISTED)


def test_generic_ternary_requires_lle_and_dwsim_markers() -> None:
    # Three components but no LLE marker.
    assert not is_generic_ternary_lle_request(
        "\u4e59\u9187 \u4e59\u9178\u4e59\u916f \u6c34 \u7cbe\u998f\uff0c\u5bfc\u51fa dwsim"
    )
    # LLE but no DWSIM/export marker.
    assert not is_generic_ternary_lle_request(
        "\u4e59\u9187 \u4e59\u9178\u4e59\u916f \u6c34 \u4e09\u5143\u6db2\u6db2\u8403\u53d6"
    )
    # Only two components -> not this router.
    assert not is_generic_ternary_lle_request("\u6b63\u4e01\u9187-\u6c34\u6db2\u6db2\u8403\u53d6\uff0c\u5bfc\u51fa dwsim")


def test_generic_ternary_resolves_three_components() -> None:
    assert len(_ternary_lle_components(_TERNARY)) == 3
    assert _ternary_lle_components(_NON_WHITELISTED) == ["Methanol", "Toluene", "Water"]


def test_binary_router_yields_to_a_three_component_message() -> None:
    """A ternary message must never be claimed by the binary router.

    "methanol toluene water" also contains a resolvable sub-pair; without the
    component-count guard the binary router would win and render the wrong case.
    """
    assert not is_binary_lle_dwsim_request(_NON_WHITELISTED)
    assert is_generic_ternary_lle_request(_NON_WHITELISTED)


def test_a_binary_pair_is_still_binary() -> None:
    binary = "\u6b63\u4e01\u9187-\u6c34\u4e8c\u5143\u6db2\u6db2\u8403\u53d6\uff0c0.3/0.7\uff0c\u5bfc\u51fa dwsim"
    assert is_binary_lle_dwsim_request(binary)
    assert not is_generic_ternary_lle_request(binary)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _export(tmp_path: Path, *, separates: bool = True, **overrides: object) -> tuple[FakeAutomation, Path]:
    from thermo_engine.dwsim_export import export_dwsim_ternary_lle_flowsheet

    automation = FakeAutomation(separates=separates)
    kwargs: dict[str, object] = dict(
        components=["Ethanol", "Ethyl acetate", "Water"],
        feed_composition=[0.129, 0.188, 0.683],
        temperature_K=298.15,
        destination=tmp_path / "ternary_lle.dwxmz",
    )
    kwargs.update(overrides)
    destination = export_dwsim_ternary_lle_flowsheet(
        factory=lambda: automation, object_type=FakeObjectType, **kwargs  # type: ignore[arg-type]
    )
    return automation, destination


def test_ternary_export_builds_the_vessel_template(tmp_path: Path) -> None:
    automation, destination = _export(tmp_path)
    assert destination.is_file()
    fs = automation.flowsheet
    assert fs.compounds == ["Ethanol", "Ethyl acetate", "Water"]
    assert fs.property_package == "NRTL"
    assert "Vessel_LLE" in fs.objects
    for tag in ("Feed", "Vapor", "Light_Liquid", "Heavy_Liquid"):
        assert tag in fs.objects, tag
    assert automation.calculate_calls == 1


def test_ternary_export_connections_match_the_template(tmp_path: Path) -> None:
    automation, _ = _export(tmp_path)
    assert automation.flowsheet.connections == [
        ("Feed", "Vessel_LLE", 0, 0),
        ("Vessel_LLE", "Vapor", 0, 0),
        ("Vessel_LLE", "Light_Liquid", 1, 0),
        ("Vessel_LLE", "Heavy_Liquid", 2, 0),
    ]


def test_ternary_export_sets_the_three_component_feed(tmp_path: Path) -> None:
    automation, _ = _export(tmp_path)
    feed = automation.flowsheet.objects["Feed"]
    assert feed.composition == pytest.approx([0.129, 0.188, 0.683])
    assert feed.temperature == pytest.approx(298.15)
    assert feed.pressure == pytest.approx(101325.0)


def test_ternary_export_rejects_degenerate_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three components"):
        _export(tmp_path, components=["Ethanol", "Water"], feed_composition=[0.5, 0.5])
    with pytest.raises(ValueError, match="sum to one"):
        _export(tmp_path, feed_composition=[0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match="distinct"):
        _export(tmp_path, components=["Water", "Water", "Ethanol"])


# --------------------------------------------------------------------------- #
# Agent runner
# --------------------------------------------------------------------------- #


def test_runner_warns_when_the_heavy_phase_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The known DWSIM ternary limitation must be surfaced, not hidden."""

    def fake_export(**kwargs: object) -> Path:
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-ternary")
        split_out = kwargs.get("split_out")
        if isinstance(split_out, dict):
            split_out.update(
                {
                    "separated": False,
                    "light_flow_mol_s": 1.0,
                    "heavy_flow_mol_s": 0.0,
                    "light_composition": [0.129, 0.188, 0.683],
                    "heavy_composition": [0.129, 0.188, 0.683],
                }
            )
        return destination

    monkeypatch.setattr(extractive, "export_dwsim_ternary_lle_flowsheet", fake_export)
    payload = extractive.run_generic_ternary_lle_export(_TERNARY, export_dir=str(tmp_path))

    assert payload.status == "ready"
    assert payload.lle_kind == "ternary"
    assert payload.ternary_spec is not None
    assert payload.ternary_spec.components == ["Ethanol", "Ethyl Acetate", "Water"]
    # The limitation is stated in the message AND raised as a warning.
    assert "未解出液液分相" in payload.message
    assert payload.warnings


def test_runner_reports_the_split_when_dwsim_separates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_export(**kwargs: object) -> Path:
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-ternary")
        split_out = kwargs.get("split_out")
        if isinstance(split_out, dict):
            split_out.update(
                {
                    "separated": True,
                    "light_flow_mol_s": 0.42,
                    "heavy_flow_mol_s": 0.58,
                    "light_composition": [0.17, 0.35, 0.48],
                    "heavy_composition": [0.086, 0.026, 0.888],
                }
            )
        return destination

    monkeypatch.setattr(extractive, "export_dwsim_ternary_lle_flowsheet", fake_export)
    payload = extractive.run_generic_ternary_lle_export(_TERNARY, export_dir=str(tmp_path))
    assert payload.status == "ready"
    assert not payload.warnings
    assert "Heavy_Liquid F=0.58000" in payload.message


def test_runner_reports_missing_composition(tmp_path: Path) -> None:
    payload = extractive.run_generic_ternary_lle_export(
        "\u4e59\u9187 \u4e59\u9178\u4e59\u916f \u6c34 \u4e09\u5143\u6db2\u6db2\u8403\u53d6\uff0c\u5bfc\u51fa dwsim",
        export_dir=str(tmp_path),
    )
    assert payload.status == "missing_parameters"
    assert "feed_composition" in payload.missing_parameters


# --------------------------------------------------------------------------- #
# Chat routing
# --------------------------------------------------------------------------- #


def test_chat_routes_generic_ternary_to_the_lle_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_export(**kwargs: object) -> Path:
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-ternary")
        split_out = kwargs.get("split_out")
        if isinstance(split_out, dict):
            split_out.update({"separated": False, "light_flow_mol_s": 1.0, "heavy_flow_mol_s": 0.0})
        return destination

    monkeypatch.setattr(extractive, "export_dwsim_ternary_lle_flowsheet", fake_export)
    monkeypatch.setenv("EXTRACTIVE_EXPORT_DIR", str(tmp_path))

    orchestrator = ConversationOrchestrator(DeterministicProvider())
    response = asyncio.run(orchestrator.chat(_TERNARY, "conv-generic-ternary"))

    assert response.intent == Intent.LLE_EXTRACTION
    assert response.lle_extraction is not None
    assert response.lle_extraction.lle_kind == "ternary"
    assert response.lle_extraction.ternary_spec is not None


def test_whitelisted_ternary_still_routes_to_the_tower_path() -> None:
    """The pre-existing whitelisted systems keep their own behaviour."""
    whitelisted = (
        "\u4e59\u9178\u6b63\u4e19\u916f+\u4e59\u9178\u4e59\u916f\u7528DMSO\u6db2\u6db2\u8403\u53d6\uff0c\u5bfc\u51fadwsim"
    )
    assert is_lle_extraction_request(whitelisted)
