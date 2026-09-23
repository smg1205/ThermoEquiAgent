"""Inspect the exported flowsheet's column specifications and stream wiring.

The design balance closes (verified), yet DWSIM reports "Failed to fulfill mass
balance for Water: Relative Error = 0.99924" -- i.e. it sees essentially none of the
water leaving.  That points at what the *flowsheet* specifies, not what we computed:
the column's own specs/settings as written into the file.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402,F401

from agent import generic_ternary_export as gte  # noqa: E402

MSG = "用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim，纯度 0.99"
outdir = Path(tempfile.mkdtemp(prefix="inspect-tf-"))
p = gte.run_generic_ternary_extractive_export(MSG, export_dir=str(outdir))
f = outdir / f"{p.file_id}.dwxmz"
print("file:", f)

from thermo_engine import dwsim_export as ded  # noqa: E402

factory, object_type = ded._automation_factory()
a = factory()
fs = a.LoadFlowsheet2(str(f))

col = None
for item in fs.SimulationObjects.Values:
    if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
        col = item.GetAsObject() if hasattr(item, "GetAsObject") else item
print(f"\ncolumn: {col.GetType().Name}")
for attr in ("NumberOfStages", "RefluxRatio", "CondenserPressure", "ReboilerPressure",
             "MaxIterations", "ColumnPressureDrop", "CondenserSpec", "ReboilerSpec",
             "CondenserTemperature", "ReboilerTemperature"):
    try:
        print(f"  {attr} = {getattr(col, attr)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {attr} <{type(exc).__name__}>")

print("\n=== attached specifications (what the solver must satisfy) ===")
try:
    for spec in col.Specs.Values:
        print(f"  name={spec.Name!r} type={spec.Type} value={getattr(spec,'Value','?')} "
              f"stage={getattr(spec,'Stage','?')}")
except Exception as exc:  # noqa: BLE001
    print("  Specs read failed:", type(exc).__name__, str(exc)[:120])

print("\n=== stream -> stage wiring ===")
for item in fs.SimulationObjects.Values:
    if item.GetType().Name != "MaterialStream":
        continue
    s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
    tag = str(item.GraphicObject.Tag)
    try:
        idx = col.GetStreamFeedStageIndex(str(item.GraphicObject.Tag))
        print(f"  {tag:<12} feed stage = {idx}")
    except Exception:  # noqa: BLE001
        try:
            idx = col.GetStreamFeedStageIndex(s)
            print(f"  {tag:<12} feed stage = {idx}")
        except Exception as exc2:  # noqa: BLE001
            print(f"  {tag:<12} not a column feed ({type(exc2).__name__})")

print("\n=== product stream connections ===")
for item in fs.SimulationObjects.Values:
    if item.GetType().Name != "MaterialStream":
        continue
    go = item.GraphicObject
    tag = str(go.Tag)
    if tag in ("Feed", "Entrainer"):
        continue
    print(f"  {tag:<12} InputConnectors={go.InputConnectors.Count} "
          f"OutputConnectors={go.OutputConnectors.Count}")

errs = a.CalculateFlowsheet2(fs)
print(f"\ncalculate error: {str(errs[0])[:220] if errs and errs.Count else ''!r}")
print("\noutdir:", outdir)
