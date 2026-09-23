"""Probe DWSIM's column specification API so a product spec can be attached.

The export sets NumberOfStages and RefluxRatio but never attaches a condenser or
reboiler specification.  A rigorous DistillationColumn with no specification has two
degrees of freedom unconstrained, so the solver cannot satisfy a component balance --
which is exactly the "Failed to fulfill mass balance for Water" error.

This finds the correct call for setting a distillate-flow (or reflux-ratio) spec.
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

print("=== condenser / reboiler spec members ===")
for m in sorted(dir(col)):
    if any(k in m.lower() for k in ("spec", "condenser", "reboiler", "distillate",
                                    "bottoms", "vaporflow")):
        print("  ", m)

print("\n=== overloads of SetCondenserSpec / SetReboilerSpec ===")
for name in ("SetCondenserSpec", "SetReboilerSpec"):
    fn = getattr(col, name, None)
    print(f"\n{name}: callable={callable(fn)}")
    if not callable(fn):
        continue
    try:
        t = fn.GetType()
        for meth in t.GetMethods():
            if meth.Name == name:
                params = ", ".join(f"{p.ParameterType.Name} {p.Name}"
                                   for p in meth.GetParameters())
                print(f"    {name}({params})")
    except Exception as exc:  # noqa: BLE001
        print("    overload read failed:", type(exc).__name__)

print("\n=== SpecType / CondenserSpecType enums ===")
try:
    import clr
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import SolvingMethods  # type: ignore
    print("  SolvingMethods imported")
except Exception as exc:  # noqa: BLE001
    print("  SolvingMethods import:", type(exc).__name__)

print("\n=== Specs collection before ===")
try:
    print("  Specs.Count =", col.Specs.Count)
except Exception as exc:  # noqa: BLE001
    print("  Specs read failed:", type(exc).__name__)
