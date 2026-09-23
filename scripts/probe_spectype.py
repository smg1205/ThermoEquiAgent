"""Find valid spectype strings and apply a working condenser/reboiler specification."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
a = factory()

print("=== SpecType enum values ===")
try:
    import System  # type: ignore
    asm = System.Reflection.Assembly.Load("DWSIM.UnitOperations")
    for t in asm.GetTypes():
        if t.Name == "SpecType" and t.IsEnum:
            print("  enum:", t.FullName)
            for n in System.Enum.GetNames(t):
                print("    ", n)
except Exception as exc:  # noqa: BLE001
    print("  enum read failed:", type(exc).__name__, str(exc)[:100])


def fresh():
    fs = a.CreateFlowsheet()
    for c in ("Water", "Toluene", "1-butanol"):
        fs.AddCompound(c)
    ded._add_property_package(fs, "NRTL")
    col = ded._simulation_object(fs.AddObject(object_type.DistillationColumn, 250, 0, "C"))
    col.SetNumberOfStages(28)
    col.RefluxRatio = 7.5
    return fs, col


def show(col, label):
    try:
        items = []
        for k in col.Specs.Keys:
            s = col.Specs[k]
            items.append(f"{k}: SType={s.SType} val={s.SpecValue} unit={s.SpecUnit} "
                         f"comp={s.ComponentID!r}")
        print(f"  [{label}] {items}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] failed: {type(exc).__name__} {str(exc)[:60]}")


print("\n=== try SetCondenserSpec with several spectype spellings ===")
for spectype in ("Distillate Flow Rate", "DistillateFlowRate", "Product Flow Rate",
                 "Molar Flow", "Reflux Ratio", "Distillate Rate"):
    fs, col = fresh()
    try:
        col.SetCondenserSpec(spectype, 0.4545, "mol/s", "")
        show(col, spectype)
    except Exception as exc:  # noqa: BLE001
        print(f"  {spectype!r} failed: {type(exc).__name__}: {str(exc)[:70]}")

print("\n=== try directly writing the existing ColumnSpec entries ===")
fs, col = fresh()
show(col, "before")
for k in col.Specs.Keys:
    s = col.Specs[k]
    print(f"  key={k!r} initial SType={s.SType} value={s.SpecValue} unit={s.SpecUnit!r}")
