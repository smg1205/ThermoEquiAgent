"""Behavioral tests for the binary liquid-liquid (LLE) DWSIM export path.

These cover the ``report/dwsim`` template flow: a two-component partially
miscible pair (canonically water / 1-butanol) that the user asks to render as a
downloadable DWSIM ``.dwxmz``.  The template topology is::

    Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid

with the NRTL property package.

The tests assert at the public seam -- request classification,
``run_binary_lle_export`` and the orchestrator ``chat`` entry -- and mock only
the DWSIM Automation call, which cannot load in a sandboxed process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

import agent.extractive_distillation as extractive
from agent.extractive_distillation import (
    _binary_lle_components,
    is_binary_lle_dwsim_request,
    is_binary_vle_dwsim_request,
    is_lle_extraction_request,
    wants_dwsim_file,
)
from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from schemas.domain import Intent

_BUTANOL_WATER = "\u6b63\u4e01\u9187-\u6c34\u4e8c\u5143\u6db2\u6db2\u8403\u53d6\uff0c0.3/0.7\uff0c\u5bfc\u51fa dwsim"


# --------------------------------------------------------------------------- #
# Request classification
# --------------------------------------------------------------------------- #


def test_binary_lle_request_hits() -> None:
    hits = (
        _BUTANOL_WATER,
        "\u6c34-\u6b63\u4e01\u9187\u6db2\u6db2\u5e73\u8861\uff0c\u5bfc\u51fadwsim\u6587\u4ef6",
        "water and 1-butanol binary LLE, export dwsim",
        "\u4f7f\u7528\u6b63\u4e01\u9187\u548c\u6c34\u505a\u6db2\u6db2\u8403\u53d6\u6a21\u62df\u5e76\u5bfc\u51fadwsim\u6587\u4ef6",
        "water + 1-butanol LLE, download the dwsim file",
    )
    for message in hits:
        assert is_binary_lle_dwsim_request(message), message


def test_binary_lle_requires_lle_and_dwsim_markers() -> None:
    # A VLE marker is *not* an LLE marker: that pair belongs to the binary
    # distillation router.
    assert not is_binary_lle_dwsim_request("2-\u4e19\u9187-\u6c34\u4e8c\u5143VLE\u7cbe\u998f\uff0c\u5bfc\u51fadwsim")
    # LLE pair with no DWSIM/export marker.
    assert not is_binary_lle_dwsim_request(_BUTANOL_WATER.replace("\uff0c\u5bfc\u51fa dwsim", ""))
    # DWSIM marker with no LLE pair.
    assert not is_binary_lle_dwsim_request("\u5bfc\u51fa dwsim \u6587\u4ef6")
    # A plain LLE *calculation* request must not produce a file.
    assert not is_binary_lle_dwsim_request(
        "\u8ba1\u7b97\u6b63\u4e01\u9187-\u6c34\u7684\u6db2\u6db2\u5e73\u8861"
    )


def test_binary_lle_yields_exactly_two_components() -> None:
    assert len(_binary_lle_components(_BUTANOL_WATER)) == 2
    assert not is_binary_lle_dwsim_request(
        "\u6b63\u4e01\u9187-\u6c34-\u4e59\u9187\u4e09\u5143\u6db2\u6db2\u8403\u53d6\uff0c\u5bfc\u51fa dwsim"
    )


def test_ternary_and_extractive_take_precedence_over_binary_lle() -> None:
    # Ternary solvent extraction keeps its own router.
    assert not is_binary_lle_dwsim_request(
        "\u4e59\u9178\u6b63\u4e19\u916f+\u4e59\u9178\u4e59\u916f\u7528DMSO\u6db2\u6db2\u8403\u53d6\uff0c\u5bfc\u51fadwsim"
    )
    # Extractive distillation (entrainer + column) keeps its own router.
    assert not is_binary_lle_dwsim_request("\u4e59\u9187-\u6c34\u8403\u53d6\u7cbe\u998f\uff0c\u5bfc\u51fa dwsim")


def test_binary_lle_and_binary_vle_are_disjoint() -> None:
    lle_message = _BUTANOL_WATER
    vle_message = "2-\u4e19\u9187-\u6c34\u4e8c\u5143VLE\u7cbe\u998f\u5854\uff0cx=0.3/0.7\uff0c\u5bfc\u51fa dwsim"
    assert is_binary_lle_dwsim_request(lle_message)
    assert not is_binary_vle_dwsim_request(lle_message)
    assert is_binary_vle_dwsim_request(vle_message)
    assert not is_binary_lle_dwsim_request(vle_message)
    assert not is_lle_extraction_request(lle_message)


def test_wants_dwsim_file_still_gates_on_deliverable_markers() -> None:
    assert wants_dwsim_file(_BUTANOL_WATER)
    assert not wants_dwsim_file("\u6b63\u4e01\u9187-\u6c34\u6db2\u6db2\u5e73\u8861")


# --------------------------------------------------------------------------- #
# The DWSIM export itself
# --------------------------------------------------------------------------- #


class FakeObjectType:
    MaterialStream = "material-stream"
    Vessel = "vessel"


class FakeStream:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.GraphicObject = tag
        self.temperature: float | None = None
        self.pressure: float | None = None
        self.flow: float | None = None
        self.composition: list[float] | None = None
        self.FlashTemperature: float | None = None
        self.FlashPressure: float | None = None

    def SetTemperature(self, value: float) -> None:
        self.temperature = value

    def SetPressure(self, value: float) -> None:
        self.pressure = value

    def SetMolarFlow(self, value: float) -> None:
        self.flow = value

    def SetOverallComposition(self, value: list[float]) -> None:
        self.composition = list(value)

    # --- DWSIM reflection surface used by ``_binary_lle_split`` -------------
    def GetType(self) -> "FakeStream":
        return self

    def GetMethod(self, name: str) -> "FakeMethod":
        return FakeMethod(self, name)

    def Invoke(self, _target: object, _args: object) -> object:
        raise AssertionError("Invoke must be routed through FakeMethod")


class FakeMethod:
    """Minimal stand-in for a reflected .NET method."""

    def __init__(self, stream: "FakeStream", name: str) -> None:
        self.stream = stream
        self.name = name

    def Invoke(self, _target: object, _args: object) -> object:
        if self.name == "GetMolarFlow":
            return self.stream.flow
        if self.name == "GetOverallComposition":
            return list(self.stream.composition or [])
        raise AssertionError(f"unexpected reflected method {self.name}")


class FakeVessel(FakeStream):
    """Mirrors DWSIM's ``Vessel``, whose flash defaults to 298.15 K."""

    def __init__(self, tag: str) -> None:
        super().__init__(tag)
        self.FlashTemperature: float = 298.15
        self.FlashPressure: float = 101325.0


class FakeFlowsheet:
    def __init__(self) -> None:
        self.compounds: list[str] = []
        self.property_package: str | None = None
        self.objects: dict[str, FakeStream] = {}
        self.connections: list[tuple[str, str, int, int]] = []

    def AddCompound(self, name: str) -> None:
        self.compounds.append(name)

    def CreateAndAddPropertyPackage(self, name: str) -> None:
        self.property_package = name

    def AddObject(self, object_type: str, _x: int, _y: int, tag: str) -> FakeStream:
        stream = FakeVessel(tag) if object_type == FakeObjectType.Vessel else FakeStream(tag)
        self.objects[tag] = stream
        return stream

    def ConnectObjects(self, source: str, target: str, source_port: int, target_port: int) -> None:
        self.connections.append((source, target, source_port, target_port))


class FakeAutomation:
    """Fake DWSIM automation that mimics a real solve.

    ``CalculateFlowsheet4`` performs the two-liquid split a real DWSIM ``Vessel``
    would, so the tests exercise the same post-solve read-back the production
    code relies on (rather than asserting against a stream that never changes).
    """

    #: Splits used for the canonical water/1-butanol case at 298.15 K.
    LIGHT_FRACTION = 0.7481074939668431
    LIGHT_X = [0.39915063483146, 0.600849365168541]
    HEAVY_X = [0.00552782963999133, 0.994472170360009]

    def __init__(self, *, separates: bool = True) -> None:
        self.flowsheet = FakeFlowsheet()
        self.calculate_calls = 0
        self.separates = separates

    def CreateFlowsheet(self) -> FakeFlowsheet:
        return self.flowsheet

    def CalculateFlowsheet4(self, _: FakeFlowsheet) -> object:
        self.calculate_calls += 1
        feed = self.flowsheet.objects["Feed"]
        total = feed.flow or 0.0
        if self.separates:
            light_flow = total * self.LIGHT_FRACTION
            self.flowsheet.objects["Light_Liquid"].flow = light_flow
            self.flowsheet.objects["Light_Liquid"].composition = list(self.LIGHT_X)
            self.flowsheet.objects["Heavy_Liquid"].flow = total - light_flow
            self.flowsheet.objects["Heavy_Liquid"].composition = list(self.HEAVY_X)
        else:
            # A single liquid phase: everything leaves as the light product and
            # the heavy phase stays empty -- the exact failure seen in the field.
            self.flowsheet.objects["Light_Liquid"].flow = total
            self.flowsheet.objects["Light_Liquid"].composition = list(feed.composition or [])
            self.flowsheet.objects["Heavy_Liquid"].flow = 0.0
            self.flowsheet.objects["Heavy_Liquid"].composition = list(feed.composition or [])
        return None

    def SaveFlowsheet(self, _: FakeFlowsheet, destination: str) -> None:
        Path(destination).write_bytes(b"fake-binary-lle-file")


def _export(tmp_path: Path, **overrides: object) -> tuple[FakeAutomation, Path]:
    from thermo_engine.dwsim_export import export_dwsim_binary_lle_flowsheet

    automation = FakeAutomation()
    kwargs: dict[str, object] = dict(
        components=["1-butanol", "Water"],
        feed_composition=[0.3, 0.7],
        temperature_K=298.15,
        pressure_kPa=101.325,
        feed_flow_mol_s=1.0,
        property_package="NRTL",
        destination=tmp_path / "binary_lle.dwxmz",
    )
    kwargs.update(overrides)
    destination = export_dwsim_binary_lle_flowsheet(
        factory=lambda: automation, object_type=FakeObjectType, **kwargs  # type: ignore[arg-type]
    )
    return automation, destination


def test_export_builds_the_report_template_topology(tmp_path: Path) -> None:
    """Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid, NRTL."""
    automation, destination = _export(tmp_path)
    assert destination.read_bytes() == b"fake-binary-lle-file"
    fs = automation.flowsheet
    assert fs.compounds == ["1-butanol", "Water"]
    assert fs.property_package == "NRTL"
    assert isinstance(fs.objects["Vessel_LLE"], FakeVessel)
    # No column and no plain TP-flash vessel.
    assert "Equilibrium Flash" not in fs.objects
    assert "Binary Distillation Column" not in fs.objects
    for tag in ("Feed", "Vapor", "Light_Liquid", "Heavy_Liquid"):
        assert tag in fs.objects, tag


def test_export_connections_match_the_template(tmp_path: Path) -> None:
    automation, _ = _export(tmp_path)
    assert automation.flowsheet.connections == [
        ("Feed", "Vessel_LLE", 0, 0),
        ("Vessel_LLE", "Vapor", 0, 0),
        ("Vessel_LLE", "Light_Liquid", 1, 0),
        ("Vessel_LLE", "Heavy_Liquid", 2, 0),
    ]


def test_export_pins_the_flash_to_the_feed_temperature(tmp_path: Path) -> None:
    """The vessel must not keep DWSIM's 298.15 K default silently."""
    automation, _ = _export(tmp_path, temperature_K=313.15)
    fs = automation.flowsheet
    assert fs.objects["Feed"].temperature == pytest.approx(313.15)
    assert fs.objects["Feed"].composition == pytest.approx([0.3, 0.7])
    assert fs.objects["Feed"].flow == pytest.approx(1.0)
    # Pressure is written in Pa, matching the rest of the DWSIM export module.
    assert fs.objects["Feed"].pressure == pytest.approx(101325.0)
    assert fs.objects["Vessel_LLE"].FlashTemperature == pytest.approx(313.15)
    assert fs.objects["Vessel_LLE"].FlashPressure == pytest.approx(101325.0)


def test_export_solves_the_flowsheet_before_saving(tmp_path: Path) -> None:
    """The saved file must carry solved phase results, not an unsolved skeleton.

    The ``report/dwsim/water_butanol_*.dwxmz`` templates are produced by scripts
    that call ``CalculateFlowsheet4`` before saving, so every object in them is
    ``<Calculated>true</Calculated>`` / ``<Status>Calculated</Status>``.  A file
    saved without solving opens in the DWSIM GUI with empty phase fractions and
    compositions, which is a silent, user-visible regression.
    """
    automation, _ = _export(tmp_path)
    assert automation.calculate_calls == 1


def test_export_survives_a_failed_solve(tmp_path: Path) -> None:
    """A non-converging solve must still yield the (correct) structure."""

    class FailingAutomation(FakeAutomation):
        def CalculateFlowsheet4(self, _: FakeFlowsheet) -> object:
            raise RuntimeError("solver did not converge")

    automation = FailingAutomation()
    from thermo_engine.dwsim_export import export_dwsim_binary_lle_flowsheet

    destination = export_dwsim_binary_lle_flowsheet(
        components=["1-butanol", "Water"],
        feed_composition=[0.3, 0.7],
        temperature_K=298.15,
        destination=tmp_path / "binary_lle.dwxmz",
        factory=lambda: automation,  # type: ignore[arg-type]
        object_type=FakeObjectType,  # type: ignore[arg-type]
    )
    assert destination.is_file()
    assert automation.flowsheet.connections


def test_export_reports_the_two_liquid_split(tmp_path: Path) -> None:
    """``split_out`` carries DWSIM's own phase result, including the heavy phase."""
    from thermo_engine.dwsim_export import export_dwsim_binary_lle_flowsheet

    automation = FakeAutomation()
    split: dict[str, object] = {}
    export_dwsim_binary_lle_flowsheet(
        components=["1-butanol", "Water"],
        feed_composition=[0.3, 0.7],
        temperature_K=298.15,
        destination=tmp_path / "binary_lle.dwxmz",
        split_out=split,
        factory=lambda: automation,  # type: ignore[arg-type]
        object_type=FakeObjectType,  # type: ignore[arg-type]
    )
    assert split["separated"] is True
    assert split["light_flow_mol_s"] == pytest.approx(0.748107, abs=1e-5)
    # The heavy phase must carry the remainder -- never zero.
    assert split["heavy_flow_mol_s"] == pytest.approx(0.251893, abs=1e-5)
    assert split["light_composition"][0] == pytest.approx(0.399151, abs=1e-5)
    assert split["heavy_composition"][1] == pytest.approx(0.994472, abs=1e-5)


def test_export_flags_a_single_phase_result(tmp_path: Path) -> None:
    """A non-separating solve must be reported as ``separated == False``."""
    from thermo_engine.dwsim_export import export_dwsim_binary_lle_flowsheet

    automation = FakeAutomation(separates=False)
    split: dict[str, object] = {}
    export_dwsim_binary_lle_flowsheet(
        components=["1-butanol", "Water"],
        feed_composition=[0.5, 0.5],
        temperature_K=298.15,
        destination=tmp_path / "binary_lle.dwxmz",
        split_out=split,
        factory=lambda: automation,  # type: ignore[arg-type]
        object_type=FakeObjectType,  # type: ignore[arg-type]
    )
    assert split["separated"] is False
    assert split["heavy_flow_mol_s"] == pytest.approx(0.0)


def test_runner_reports_the_heavy_phase_in_its_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat message must state the heavy-phase flow, not just claim success."""
    payload = extractive.run_binary_lle_export(
        f"{_BUTANOL_WATER}\uff0c298.15 K", export_dir=str(tmp_path)
    )
    assert payload.status == "ready"
    assert "Heavy_Liquid" in payload.message
    # DWSIM's real numbers must appear (this runs against the local DWSIM).
    assert "0.25189" in payload.message or "0.74811" in payload.message


def test_runner_warns_when_dwsim_predicts_a_single_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-phase render must be surfaced, not presented as a valid split."""

    def fake_export(**kwargs: object) -> Path:
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake")
        split_out = kwargs.get("split_out")
        if isinstance(split_out, dict):
            split_out.update(
                {
                    "separated": False,
                    "light_flow_mol_s": 1.0,
                    "heavy_flow_mol_s": 0.0,
                    "light_composition": [0.5, 0.5],
                    "heavy_composition": [0.5, 0.5],
                }
            )
        return destination

    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", fake_export)
    payload = extractive.run_binary_lle_export(f"{_BUTANOL_WATER}\uff0c298.15 K", export_dir=str(tmp_path))
    assert payload.status == "ready"
    assert "未解出液液分相" in payload.message


def test_export_rejects_a_ternary_or_degenerate_case(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two components"):
        _export(tmp_path, components=["1-butanol", "Water", "Ethanol"], feed_composition=[0.3, 0.6, 0.1])
    with pytest.raises(ValueError, match="sum to one"):
        _export(tmp_path, feed_composition=[0.3, 0.3])
    with pytest.raises(ValueError, match="distinct"):
        _export(tmp_path, components=["Water", "water"])
    with pytest.raises(ValueError, match=r"\.dwxmz"):
        _export(tmp_path, destination=tmp_path / "binary_lle.json")


def test_dwsim_compound_mapping_covers_the_binary_lle_pairs() -> None:
    """DWSIM keys 1-butanol in lower case; the resolver returns title case."""
    from thermo_engine.dwsim_export import _dwsim_compound_name, missing_dwsim_compound_mappings

    assert _dwsim_compound_name("1-Butanol") == "1-butanol"
    assert _dwsim_compound_name("n-butanol") == "1-butanol"
    assert _dwsim_compound_name("\u6b63\u4e01\u9187") == "1-butanol"
    assert _dwsim_compound_name("4-Methyl-2-Pentanone") == "Methyl isobutyl ketone"
    assert _dwsim_compound_name("Water") == "Water"
    assert missing_dwsim_compound_mappings(["1-Butanol", "Water"]) == []


# --------------------------------------------------------------------------- #
# The agent runner
# --------------------------------------------------------------------------- #


def _fake_export(**kwargs: object) -> Path:
    destination = Path(kwargs["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"fake-binary-lle-file")
    return destination


def test_run_binary_lle_export_writes_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", _fake_export)
    payload = extractive.run_binary_lle_export(
        f"{_BUTANOL_WATER}\uff0c298.15 K", export_dir=str(tmp_path)
    )
    assert payload.status == "ready"
    assert payload.lle_kind == "binary"
    assert payload.binary_spec is not None
    assert payload.binary_spec.kind == "lle"
    assert payload.binary_spec.mode == "two_liquid_vessel"
    assert payload.binary_spec.feed_composition == pytest.approx([0.3, 0.7])
    assert payload.binary_spec.temperature_K == pytest.approx(298.15)
    assert payload.file_id is not None
    assert payload.dwsim_file_uri == f"/api/export/extractive/{payload.file_id}.dwxmz"
    assert (tmp_path / f"{payload.file_id}.dwxmz").is_file()


def test_run_binary_lle_export_defaults_the_temperature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing temperature falls back to the template default and says so."""
    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", _fake_export)
    payload = extractive.run_binary_lle_export(_BUTANOL_WATER, export_dir=str(tmp_path))
    assert payload.status == "ready"
    assert payload.binary_spec is not None
    assert payload.binary_spec.temperature_K == pytest.approx(298.15)
    assert "298.15" in payload.message


def test_run_binary_lle_export_honours_an_explicit_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", _fake_export)
    payload = extractive.run_binary_lle_export(
        f"{_BUTANOL_WATER}\uff0c25\u2103", export_dir=str(tmp_path)
    )
    assert payload.status == "ready"
    assert payload.binary_spec is not None
    assert payload.binary_spec.temperature_K == pytest.approx(298.15)


def test_run_binary_lle_export_reports_missing_mapping(tmp_path: Path) -> None:
    payload = extractive.run_binary_lle_export(
        "\u73af\u5df1\u70f7-\u6c34\u4e8c\u5143\u6db2\u6db2\u8403\u53d6\uff0c0.3/0.7\uff0c\u5bfc\u51fa dwsim",
        export_dir=str(tmp_path),
    )
    assert payload.status == "missing_parameters"
    assert "dwsim_compound_mapping" in payload.missing_parameters
    assert payload.dwsim_file_uri is None


def test_run_binary_lle_export_handles_dwsim_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from schemas.domain import FailureType
    from thermo_engine.errors import ThermoEquiError

    def raise_missing(**_: object) -> Path:
        raise ThermoEquiError(FailureType.MISSING_DATA, "DWSIM_HOME is not configured.", "Set DWSIM_HOME.")

    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", raise_missing)
    payload = extractive.run_binary_lle_export(_BUTANOL_WATER, export_dir=str(tmp_path))
    assert payload.status == "dwsim_unavailable"
    assert payload.dwsim_file_uri is None
    # The resolved inputs still reach the caller.
    assert payload.binary_spec is not None
    assert payload.binary_spec.components == ["1-Butanol", "Water"]


def test_evidence_names_the_template_and_the_dwsim_boundary() -> None:
    payload = extractive.LLEExportPayload(
        status="ready",
        message="",
        lle_kind="binary",
    )
    statements = extractive.evidence_for_lle_extraction(payload)
    text = " ".join(s.text for s in statements)
    assert "Vessel_LLE" in text
    assert any(s.category == "Warning" for s in statements)


# --------------------------------------------------------------------------- #
# Chat routing
# --------------------------------------------------------------------------- #


async def _chat(message: str, conversation_id: str):  # pragma: no cover - helper
    orchestrator = ConversationOrchestrator(DeterministicProvider())
    return await orchestrator.chat(message, conversation_id)


def test_chat_routes_binary_lle_to_the_lle_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extractive, "export_dwsim_binary_lle_flowsheet", _fake_export)
    monkeypatch.setattr(extractive, "export_directory", lambda *_a, **_k: tmp_path)
    response = asyncio.run(_chat(_BUTANOL_WATER, "conv-binary-lle"))
    assert response.intent == Intent.LLE_EXTRACTION
    assert response.lle_extraction is not None
    assert response.lle_extraction.lle_kind == "binary"
    assert response.lle_extraction.status == "ready"
    assert response.lle_extraction.dwsim_file_uri is not None
    assert response.statements


def test_chat_keeps_binary_vle_on_the_distillation_channel() -> None:
    response = asyncio.run(
        _chat(
            "2-\u4e19\u9187-\u6c34\u4e8c\u5143VLE\u7cbe\u998f\u5854\uff0cx=0.3/0.7\uff0c\u5bfc\u51fa dwsim",
            "conv-binary-vle-still",
        )
    )
    assert response.intent == Intent.DISTILLATION_DESIGN
    assert response.distillation is not None
    assert response.lle_extraction is None
