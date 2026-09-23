"""Verify the product-flow spec pair against DWSIM's own defaults.

A fresh rigorous column has Specs = {"C": Stream_Ratio (reflux), "R":
Product_Molar_Flow_Rate (0.0)}.  The "C" slot is already driven by RefluxRatio, so
overwriting it with a flow spec would remove the reflux constraint and over-specify
the column.  This probe checks what a correct extractive configuration looks like by
setting only the reboiler flow and leaving the condenser on reflux.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
a = factory()
fs = a.CreateFlowsheet()
for c in ("Water", "Toluene", "1-butanol"):
    fs.AddCompound(c)
ded._add_property_package(fs, "NRTL")
col = ded._simulation_object(fs.AddObject(object_type.DistillationColumn, 250, 0, "C"))
col.SetNumberOfStages(28)
col.RefluxRatio = 7.5

print("=== default specs of a fresh column ===")
for k in col.Specs.Keys:
    s = col.Specs[k]
    print(f"  {k!r}: SType={s.SType} value={s.SpecValue} unit={s.SpecUnit!r}")

print("\n=== after SetRefluxRatio(7.5) ===")
for k in col.Specs.Keys:
    s = col.Specs[k]
    print(f"  {k!r}: SType={s.SType} value={s.SpecValue} unit={s.SpecUnit!r}")

print("\n=== what does the working reference file use? ===")
ref = ROOT / "report" / "dwsim" / "water_butanol_LLE_298p15K.dwxmz"
try:
    fs2 = a.LoadFlowsheet2(str(ref))
    for item in fs2.SimulationObjects.Values:
        if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
            c2 = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            for k in c2.Specs.Keys:
                s = c2.Specs[k]
                print(f"  {k!r}: SType={s.SType} value={s.SpecValue} unit={s.SpecUnit!r}")
except Exception as exc:  # noqa: BLE001
    print("  reference read failed:", type(exc).__name__)

print("\n=== check an existing extractive column file (if any) ===")
for cand in sorted((ROOT / "data" / "exports" / "flow_examples").glob("*extractive*.dwxmz")):
    try:
        fs3 = a.LoadFlowsheet2(str(cand))
        for item in fs3.SimulationObjects.Values:
            if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
                c3 = item.GetAsObject() if hasattr(item, "GetAsObject") else item
                specs = []
                for k in c3.Specs.Keys:
                    s = c3.Specs[k]
                    specs.append(f"{k}:{s.SType}={s.SpecValue}")
                print(f"  {cand.name}: R={getattr(c3,'RefluxRatio','?')} specs={specs}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {cand.name}: read failed {type(exc).__name__}")
