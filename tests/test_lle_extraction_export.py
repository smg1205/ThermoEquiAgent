"""Behavioral tests for the liquid-liquid extraction (decanter) DWSIM export.

These verify that ``export_dwsim_lle_extraction`` is a distinct code path from
the TP-flash and extractive-distillation exports, that it resolves the ternary
n-propyl acetate + ethyl acetate + DMSO system to DWSIM-canonical compound keys,
and that it produces a feed -> solvent -> decanter -> raffinate/extract topology
using NRTL (DWSIM supplies its own LLE binary parameters).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from agent.extractive_distillation import (
    is_lle_extraction_request,
    run_lle_extraction_export,
)
from thermo_engine.dwsim_export import _dwsim_compound_name, export_dwsim_lle_extraction


class FakeObjectType(Enum):
    MaterialStream = "material-stream"
    Vessel = "vessel"
    LiquidLiquidExtractor = "liquid-liquid-extractor"
    DistillationColumn = "distillation-column"
    ShortcutColumn = "shortcut-column"


class FakeStream:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.GraphicObject = tag
        self.temperature: float | None = None
        self.pressure: float | None = None
        self.flow: float | None = None
        self.composition: list[float] | None = None

    def SetTemperature(self, value: float) -> None:
        self.temperature = value

    def SetPressure(self, value: float) -> None:
        self.pressure = value

    def SetMolarFlow(self, value: float) -> None:
        self.flow = value

    def SetOverallComposition(self, value: list[float]) -> None:
        self.composition = value


class FakeFlowsheet:
    def __init__(self) -> None:
        self.compounds: list[str] = []
        self.property_package: str | None = None
        self.objects: dict[str, FakeStream] = {}
        self.connections: list[tuple[str, str, int, int]] = []
        self.create_and_add_calls: list[str] = []

    def AddCompound(self, name: str) -> None:
        self.compounds.append(name)

    def AddPropertyPackage(self, name: str) -> None:
        self.property_package = name

    def CreateAndAddPropertyPackage(self, name: str) -> None:
        self.property_package = name
        self.create_and_add_calls.append(name)

    def AddObject(self, _: FakeObjectType, _x: int, _y: int, tag: str) -> FakeStream:
        stream = FakeStream(tag)
        self.objects[tag] = stream
        return stream

    def ConnectObjects(self, source: str, target: str, source_port: int, target_port: int) -> None:
        self.connections.append((source, target, source_port, target_port))


class FakeAutomation:
    def __init__(self) -> None:
        self.flowsheet = FakeFlowsheet()

    def CreateFlowsheet(self) -> FakeFlowsheet:
        return self.flowsheet

    def SaveFlowsheet(self, _: FakeFlowsheet, destination: str) -> None:
        Path(destination).write_bytes(b"fake-lle-extraction-file")


def _lle_export_kwargs(tmp_path: Path) -> dict[str, object]:
    return dict(
        components=["n-propyl acetate", "ethyl acetate", "Dimethyl sulfoxide"],
        feed_composition=[0.4, 0.6],
        feed_flow_mol_s=1.0,
        feed_temperature_K=298.15,
        feed_pressure_kPa=101.325,
        solvent="Dimethyl sulfoxide",
        solvent_ratio=1.5,
        property_package="NRTL",
        destination=tmp_path / "lle.dwxmz",
    )


def test_lle_export_builds_a_decanter_not_a_column_or_flash(tmp_path: Path) -> None:
    automation = FakeAutomation()
    destination = export_dwsim_lle_extraction(
        factory=lambda: automation, object_type=FakeObjectType, **_lle_export_kwargs(tmp_path)  # type: ignore[arg-type]
    )

    assert destination.read_bytes() == b"fake-lle-extraction-file"
    fs = automation.flowsheet
    # Model-side (underscored/Chinese) names are canonicalised to DWSIM keys.
    # DWSIM stores propyl acetate as "N-propyl acetate" (uppercase N, lowercase p).
    assert fs.compounds == ["N-propyl acetate", "Ethyl acetate", "Dimethyl sulfoxide"]
    assert fs.property_package == "NRTL"
    # A liquid-liquid extractor decanter object must exist.
    assert "Liquid-Liquid Extractor" in fs.objects
    # No column, no TP-flash vessel.
    assert "Equilibrium Flash" not in fs.objects
    # Four material streams: feed, solvent, raffinate, extract.
    assert "Feed" in fs.objects
    assert "Solvent" in fs.objects
    assert "Raffinate" in fs.objects
    assert "Extract" in fs.objects


def test_lle_export_sets_feed_and_solvent_flows(tmp_path: Path) -> None:
    automation = FakeAutomation()
    kwargs = _lle_export_kwargs(tmp_path)
    export_dwsim_lle_extraction(
        factory=lambda: automation, object_type=FakeObjectType, **kwargs  # type: ignore[arg-type]
    )
    fs = automation.flowsheet
    feed = fs.objects["Feed"]
    assert feed.flow == pytest.approx(1.0)
    # Feed is the two solutes only (solvent fraction zero).
    assert feed.composition == pytest.approx([0.4, 0.6, 0.0])
    solvent = fs.objects["Solvent"]
    # Solvent molar flow = ratio × feed flow; pure DMSO at the solvent index (2).
    assert solvent.flow == pytest.approx(1.5)
    assert solvent.composition == pytest.approx([0.0, 0.0, 1.0])


def test_lle_export_connections_match_extraction_topology(tmp_path: Path) -> None:
    automation = FakeAutomation()
    kwargs = _lle_export_kwargs(tmp_path)
    export_dwsim_lle_extraction(
        factory=lambda: automation, object_type=FakeObjectType, **kwargs  # type: ignore[arg-type]
    )
    fs = automation.flowsheet
    extractor = "Liquid-Liquid Extractor"
    # feed/solvent in; raffinate (port 0) + extract (port 1) out.
    assert fs.connections == [
        ("Feed", extractor, 0, 0),
        ("Solvent", extractor, 0, 1),
        (extractor, "Raffinate", 0, 0),
        (extractor, "Extract", 1, 0),
    ]


def test_lle_export_requires_dwxmz_extension(tmp_path: Path) -> None:
    automation = FakeAutomation()
    kwargs = _lle_export_kwargs(tmp_path)
    kwargs["destination"] = tmp_path / "lle.json"
    with pytest.raises(ValueError, match=".dwxmz"):
        export_dwsim_lle_extraction(  # type: ignore[arg-type]
            factory=lambda: automation, object_type=FakeObjectType, **kwargs
        )


def test_lle_export_validates_arguments(tmp_path: Path) -> None:
    automation = FakeAutomation()
    base = _lle_export_kwargs(tmp_path)
    # Too few components.
    bad = dict(base)
    bad["components"] = ["n-propyl acetate", "ethyl acetate"]
    with pytest.raises(ValueError, match="three components"):
        export_dwsim_lle_extraction(  # type: ignore[arg-type]
            factory=lambda: automation, object_type=FakeObjectType, **bad
        )
    # Feed composition not summing to one.
    bad2 = dict(base)
    bad2["feed_composition"] = [0.4, 0.4]
    with pytest.raises(ValueError, match="sum to one"):
        export_dwsim_lle_extraction(  # type: ignore[arg-type]
            factory=lambda: automation, object_type=FakeObjectType, **bad2
        )
    # Solvent not among the components.
    bad3 = dict(base)
    bad3["solvent"] = "water"
    with pytest.raises(ValueError, match="extraction solvent"):
        export_dwsim_lle_extraction(  # type: ignore[arg-type]
            factory=lambda: automation, object_type=FakeObjectType, **bad3
        )


def test_dwsim_compound_name_maps_lle_components() -> None:
    assert _dwsim_compound_name("n-propyl acetate") == "N-propyl acetate"
    assert _dwsim_compound_name("propyl acetate") == "N-propyl acetate"
    assert _dwsim_compound_name("乙酸正丙酯") == "N-propyl acetate"
    assert _dwsim_compound_name("ethyl acetate") == "Ethyl acetate"
    assert _dwsim_compound_name("dimethyl sulfoxide") == "Dimethyl sulfoxide"
    assert _dwsim_compound_name("dmso") == "Dimethyl sulfoxide"
    assert _dwsim_compound_name("二甲基亚砜") == "Dimethyl sulfoxide"


def test_lle_request_detection() -> None:
    assert not is_lle_extraction_request("乙酸正丙酯+乙酸乙酯+二甲基亚砜，做萃取的dwsim导出")
    assert is_lle_extraction_request("帮我做 n-propyl acetate、ethyl acetate 的 DMSO 液液萃取模拟")
    # Solvent or solutes missing -> not an LLE request.
    assert not is_lle_extraction_request("乙酸正丙酯+乙酸乙酯 萃取精馏")
    assert not is_lle_extraction_request("乙醇水萃取精馏，用乙二醇")


def test_lle_equilibrium_calculation_is_not_extraction_export() -> None:
    assert not is_lle_extraction_request(
        "use ThermoFormer to calculate ethyl acetate n-propyl acetate DMSO LLE at 298.15 K and 101.325 kPa"
    )
    assert is_lle_extraction_request(
        "build a DWSIM liquid-liquid extraction export for ethyl acetate n-propyl acetate DMSO"
    )


def test_run_lle_extraction_export_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import extractive_distillation as mod

    def fake_export(**kwargs: object) -> Path:
        Path(kwargs["destination"]).write_bytes(b"fake-lle-lle-file")
        return Path(kwargs["destination"])

    monkeypatch.setattr(mod, "export_dwsim_lle_extraction", fake_export)
    payload = run_lle_extraction_export(
        "乙酸正丙酯 40%、乙酸乙酯 60%，用二甲基亚砜做溶剂，溶剂比 1.5，做萃取的dwsim导出",
        export_dir=str(tmp_path),
    )
    assert payload.status == "ready"
    assert payload.dwsim_file_uri is not None
    assert payload.spec is not None
    assert payload.spec.feed_components == ["n-propyl acetate", "ethyl acetate"]
    assert payload.spec.solvent == "Dimethyl sulfoxide"
    assert payload.spec.solvent_ratio == pytest.approx(1.5)
    assert payload.spec.feed_composition == pytest.approx([0.4, 0.6])


def test_run_lle_extraction_export_reports_missing_parameters(tmp_path: Path) -> None:
    payload = run_lle_extraction_export("乙酸正丙酯和乙酸乙酯用DMSO萃取", export_dir=str(tmp_path))
    # No solute fractions parsed -> missing_parameters.
    assert payload.status == "missing_parameters"
    assert "feed_composition" in payload.missing_parameters


def test_run_lle_extraction_export_handles_dwsim_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import extractive_distillation as mod

    def raise_missing(**kwargs: object) -> Path:
        from schemas.domain import FailureType

        raise ThermoEquiError(FailureType.MISSING_DATA, "DWSIM_HOME is not configured.", "Set DWSIM_HOME.")

    monkeypatch.setattr(mod, "export_dwsim_lle_extraction", raise_missing)
    payload = run_lle_extraction_export(
        "乙酸正丙酯 40%、乙酸乙酯 60%，二甲基亚砜 萃取",
        export_dir=str(tmp_path),
    )
    assert payload.status == "dwsim_unavailable"
    assert payload.dwsim_file_uri is None
