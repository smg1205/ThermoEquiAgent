"""Determine how to attach a condenser/reboiler specification to the column.

A rigorous column needs two specifications.  Setting DistillateFlowRate and
RefluxRatio is the standard pair for an extractive column; this probe checks which
assignment actually creates entries in Specs / sets IsSpecAttached.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
a = factory()


def fresh():
    fs = a.CreateFlowsheet()
    for c in ("Water", "Toluene", "1-butanol"):
        fs.AddCompound(c)
    ded._add_property_package(fs, "NRTL")
    col = ded._simulation_object(fs.AddObject(object_type.DistillationColumn, 250, 0, "C"))
    col.SetNumberOfStages(28)
    return fs, col


def state(col, label):
    try:
        n = col.Specs.Count
        ids = []
        for s in col.Specs.Values:
            ids.append(f"{s.Name}:{s.Type}={getattr(s,'Value','?')}")
        print(f"  [{label}] Specs.Count={n} IsSpecAttached={col.IsSpecAttached} {ids}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] read failed: {type(exc).__name__} {str(exc)[:60]}")


fs, col = fresh()
state(col, "initial")

print("\n-- try DistillateFlowRate = 0.4545 --")
try:
    col.DistillateFlowRate = 0.4545
    state(col, "after D")
except Exception as exc:  # noqa: BLE001
    print("  failed:", type(exc).__name__, str(exc)[:80])

print("\n-- try SetCondenserSpec(DistillateFlowRate, 0.4545, 1) --")
fs2, col2 = fresh()
for args in (
    ("DistillateFlowRate", 0.4545, 1),
    (0, 0.4545, 1),
    ("Distillate Flow Rate", 0.4545, 1),
):
    fs3, col3 = fresh()
    try:
        col3.SetCondenserSpec(*args)
        state(col3, f"SetCondenserSpec{args}")
    except Exception as exc:  # noqa: BLE001
        print(f"  SetCondenserSpec{args} failed: {type(exc).__name__}: {str(exc)[:70]}")

print("\n-- try SetReboilerSpec variants --")
for args in (("BottomsFlowRate", 2.5455, 1), (0, 2.5455, 1)):
    fs4, col4 = fresh()
    try:
        col4.SetReboilerSpec(*args)
        state(col4, f"SetReboilerSpec{args}")
    except Exception as exc:  # noqa: BLE001
        print(f"  SetReboilerSpec{args} failed: {type(exc).__name__}: {str(exc)[:70]}")

print("\n-- overload signatures --")
for name in ("SetCondenserSpec", "SetReboilerSpec"):
    fn = getattr(col, name)
    try:
        for meth in fn.GetType().GetMethods():
            if meth.Name == name:
                params = ", ".join(f"{p.ParameterType.Name} {p.Name}"
                                   for p in meth.GetParameters())
                print(f"  {name}({params})")
    except Exception as exc:  # noqa: BLE001
        print("  ", name, "overload read failed:", type(exc).__name__)
