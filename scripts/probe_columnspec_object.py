"""Inspect the existing ColumnSpec object and the real SetCondenserSpec signature."""
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

print("=== Specs collection ===")
specs = col.Specs
print("  type:", specs.GetType().FullName, " count:", specs.Count)
for k in specs.Keys:
    s = specs[k]
    print(f"  key={k!r} spec type={s.GetType().Name}")
    print("    properties:")
    for p in s.GetType().GetProperties():
        try:
            print(f"      {p.Name} : {p.PropertyType.Name} (canWrite={p.CanWrite})")
        except Exception:  # noqa: BLE001
            pass
    break

print("\n=== SetCondenserSpec / SetReboilerSpec signatures via .NET reflection ===")
import System  # type: ignore  # noqa: E402

t = col.GetType()
for name in ("SetCondenserSpec", "SetReboilerSpec"):
    print(f"\n{name}:")
    for meth in t.GetMethods():
        if meth.Name == name:
            parts = []
            for p in meth.GetParameters():
                parts.append(f"{p.ParameterType.FullName} {p.Name}")
            print(f"    ({', '.join(parts)})")
