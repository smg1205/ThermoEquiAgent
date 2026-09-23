"""Behavioral tests for the DWSIM extractive-distillation column export.

These verify that the extractive column export is a distinct code path from the
TP-flash ``export_dwsim_flowsheet`` and that the computed design numbers are
applied to a DWSIM column object.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from schemas.column_design import ExtractiveColumnDesign, ExtractiveColumnSpec
from thermo_engine.column_design import design_extractive_distillation_column
from thermo_engine.dwsim_export import export_dwsim_extractive_column


class FakeObjectType(Enum):
    MaterialStream = "material-stream"
    Vessel = "vessel"
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

    def AddCompound(self, name: str) -> None:
        self.compounds.append(name)

    def AddPropertyPackage(self, name: str) -> None:
        self.property_package = name

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
        Path(destination).write_bytes(b"fake-extractive-column-file")


def _design() -> ExtractiveColumnDesign:
    spec = ExtractiveColumnSpec(
        feed_components=["ethanol", "water"],
        feed_composition=[0.40, 0.60],
        feed_flow_mol_s=1.0,
        feed_temperature_K=298.15,
        feed_pressure_kPa=101.325,
        entrainer="ethylene glycol",
        entrainer_ratio=2.0,
        operating_pressure_kPa=101.325,
    )
    return design_extractive_distillation_column(spec)


def test_extractive_export_builds_a_column_not_a_flash_vessel(tmp_path: Path) -> None:
    automation = FakeAutomation()
    design = _design()
    destination = export_dwsim_extractive_column(
        design, tmp_path / "extractive.dwxmz", factory=lambda: automation, object_type=FakeObjectType
    )

    assert destination.read_bytes() == b"fake-extractive-column-file"
    fs = automation.flowsheet
    # Model-side names are canonicalised to DWSIM's case-sensitive compound keys
    # before AddCompound, so the recorded names are DWSIM-canonical.
    assert fs.compounds == ["Ethanol", "Water", "Ethylene glycol"]
    assert fs.property_package == "NRTL"
    # A column object must exist; no "Equilibrium Flash" vessel may exist.
    assert "Extractive Distillation Column" in fs.objects
    assert "Equilibrium Flash" not in fs.objects
    # Four material streams: feed, entrainer, distillate, bottoms.
    assert "Ethanol Water Feed" in fs.objects
    assert "Entrainer" in fs.objects
    assert "Ethanol Product" in fs.objects
    assert "Water Entrainer Bottoms" in fs.objects


def test_extractive_export_applies_computed_design_numbers(tmp_path: Path) -> None:
    automation = FakeAutomation()
    design = _design()
    export_dwsim_extractive_column(
        design, tmp_path / "extractive.dwxmz", factory=lambda: automation, object_type=FakeObjectType
    )
    fs = automation.flowsheet
    feed = fs.objects["Ethanol Water Feed"]
    assert feed.flow == 1.0
    assert feed.composition == [0.4, 0.6, 0.0]
    entrainer = fs.objects["Entrainer"]
    assert entrainer.flow == pytest.approx(2.0)
    assert entrainer.composition == [0.0, 0.0, 1.0]


def test_extractive_export_connections_match_process_topology(tmp_path: Path) -> None:
    automation = FakeAutomation()
    export_dwsim_extractive_column(
        _design(), tmp_path / "extractive.dwxmz", factory=lambda: automation, object_type=FakeObjectType
    )
    fs = automation.flowsheet
    column = "Extractive Distillation Column"
    # A DistillationColumn (rigorous) is preferred because it accepts two feeds:
    # feed enters on inlet port 0 and the entrainer on inlet port 1, matching the
    # extractive topology.  Column outlets: 0 = distillate, 1 = bottoms.
    assert fs.connections == [
        ("Ethanol Water Feed", column, 0, 0),
        ("Entrainer", column, 0, 1),
        (column, "Ethanol Product", 0, 0),
        (column, "Water Entrainer Bottoms", 1, 0),
    ]


def test_extractive_export_requires_dwxmz_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".dwxmz"):
        export_dwsim_extractive_column(
            _design(), tmp_path / "extractive.json", factory=FakeAutomation, object_type=FakeObjectType
        )


def test_extractive_export_tolerates_a_rejected_connection(tmp_path: Path) -> None:
    """A rejected DWSIM graphic connection must not abort the whole export.

    On some DWSIM versions a ShortcutColumn rejects certain inlet connections
    (``ConnectObject`` raises ``System.Exception``).  The exporter should record a
    warning and still write the flowsheet instead of failing with a misleading
    "DWSIM unavailable" error.
    """

    class FlakyFlowsheet(FakeFlowsheet):
        def ConnectObjects(self, source: str, target: str, source_port: int, target_port: int) -> None:
            if target.startswith("Extractive Distillation Column") and source.startswith("Entrainer"):
                raise RuntimeError("DWSIM rejected the connection")
            super().ConnectObjects(source, target, source_port, target_port)

    class FlakyAutomation(FakeAutomation):
        def __init__(self) -> None:
            self.flowsheet = FlakyFlowsheet()

    automation = FlakyAutomation()
    design = _design()
    destination = export_dwsim_extractive_column(
        design, tmp_path / "extractive.dwxmz", factory=lambda: automation, object_type=FakeObjectType
    )
    # The file is still written even though one connection was rejected.
    assert destination.read_bytes() == b"fake-extractive-column-file"
    # The connection that was attempted and accepted is still recorded.
    assert ("Ethanol Water Feed", "Extractive Distillation Column", 0, 0) in automation.flowsheet.connections
    # The rejected connection must not have been recorded (DWSIM refused it).
    assert not any(
        c[:2] == ("Entrainer", "Extractive Distillation Column") for c in automation.flowsheet.connections
    )


def test_dwsim_compound_name_maps_model_spellings_to_canonical_keys() -> None:
    """Model-side (often lowercase) names must resolve to DWSIM's dictionary keys.

    DWSIM's ``AddCompound`` is case-sensitive and raises KeyNotFoundException for
    unknown keys; this guard ensures common spellings are canonicalised before the
    call so real DWSIM exports no longer fail on the default extractive components.
    """
    from thermo_engine.dwsim_export import _dwsim_compound_name

    assert _dwsim_compound_name("ethanol") == "Ethanol"
    assert _dwsim_compound_name("water") == "Water"
    assert _dwsim_compound_name("ethylene glycol") == "Ethylene glycol"
    # Case-insensitive for already-correct spellings.
    assert _dwsim_compound_name("Ethanol") == "Ethanol"
    assert _dwsim_compound_name("ETHYLENE GLYCOL") == "Ethylene glycol"
    # Unknown names pass through unchanged rather than being silently rewritten.
    assert _dwsim_compound_name("acetonitrile") == "acetonitrile"


def test_export_dwsim_binary_column() -> None:
    """``export_dwsim_binary_column`` is a standalone, non-extractive code path:
    two compounds, one feed, one distillate + one bottoms stream, and the design
    numbers are applied to the column object."""
    import os
    import tempfile
    import shutil

    export_dir = tempfile.mkdtemp(prefix="bin_dwsim_", dir=os.getcwd())
    from thermo_engine.dwsim_export import export_dwsim_binary_column

    automation = FakeAutomation()
    try:
        destination = export_dwsim_binary_column(
            components=["methanol", "water"],
            feed_composition=[0.5, 0.5],
            feed_flow_mol_s=1.0,
            feed_temperature_K=337.0,
            operating_pressure_kPa=101.325,
            stages=16,
            minimum_stages=7.0,
            reflux_ratio=1.0,
            minimum_reflux_ratio=0.7,
            feed_stage=7,
            condenser_temperature_K=337.98,
            reboiler_temperature_K=369.77,
            destination=Path(export_dir) / "binary.dwxmz",
            factory=lambda: automation,
            object_type=FakeObjectType,
        )
        assert destination.read_bytes() == b"fake-extractive-column-file"
        # Two compounds, no entrainer/extractant.
        assert automation.flowsheet.compounds == ["Methanol", "Water"]
        # Column object present (distillation column) with 3 streams.
        assert "Binary Distillation Column" in automation.flowsheet.objects
        # Three connections: feed->column, column->distillate, column->bottoms.
        assert len(automation.flowsheet.connections) == 3
        # No entrainer stream was created.
        assert "Entrainer" not in automation.flowsheet.objects
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)
    """Model-side (often lowercase) names must resolve to DWSIM's dictionary keys.

    DWSIM's ``AddCompound`` is case-sensitive and raises KeyNotFoundException for
    unknown keys; this guard ensures common spellings are canonicalised before the
    call so real DWSIM exports no longer fail on the default extractive components.
    """
    from thermo_engine.dwsim_export import _dwsim_compound_name

    assert _dwsim_compound_name("ethanol") == "Ethanol"
    assert _dwsim_compound_name("water") == "Water"
    assert _dwsim_compound_name("ethylene glycol") == "Ethylene glycol"
    # Case-insensitive for already-correct spellings.
    assert _dwsim_compound_name("Ethanol") == "Ethanol"
    assert _dwsim_compound_name("ETHYLENE GLYCOL") == "Ethylene glycol"
    # Unknown names pass through unchanged rather than being silently rewritten.
    assert _dwsim_compound_name("acetonitrile") == "acetonitrile"
