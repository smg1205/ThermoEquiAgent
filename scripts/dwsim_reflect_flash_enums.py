"""Reflect Vessel's CalculationModes enum + flash algorithm enums + any two-liquid switch."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
automation = factory()
fs = automation.CreateFlowsheet()
for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
    try:
        fs.AddCompound(cand); break
    except Exception:
        continue
fs.AddCompound("Water")
ded._add_property_package(fs, "UNIQUAC")
vessel_go = fs.AddObject(object_type.Vessel, 300, 0, "Vessel")
vessel = ded._simulation_object(vessel_go)

import System

def dump_enum(propName):
    pr = vessel.GetType().GetProperty(propName)
    if pr is None:
        print(f"{propName}: NO PROPERTY")
        return
    t = pr.PropertyType
    print(f"\n{propName} -> {t.FullName}")
    if t.IsEnum:
        for name in System.Enum.GetNames(t):
            print(f"    {name}")
    elif t.FullName and "List" in t.FullName:
        # a list of strings (calculation modes)
        try:
            vals = pr.GetValue(vessel, None)
            for v in vals:
                print("    ", repr(v))
        except Exception as e:
            print("    list read err", type(e).__name__)

dump_enum("CalculationMode")
dump_enum("PressureCalculation")

# search ALL assemblies for enum types whose names mention Flash/Phase/LL/TwoLiquid
print("\n=== enums with 'flash'/'phase'/'liquid'/'ll' in name (across loaded DWSIM asms) ===")
seen = set()
for asm in System.AppDomain.CurrentDomain.GetAssemblies():
    name = str(asm.GetName().Name)
    if "DWSIM" not in name:
        continue
    try:
        types = asm.GetTypes()
    except Exception:
        continue
    for t in types:
        if t.IsEnum:
            tn = t.Name
            if any(k in tn.lower() for k in ("flash", "phase", "liquid", "ll", "twoliquid", "stability")):
                key = (name, t.FullName)
                if key not in seen:
                    seen.add(key)
                    print(f"  [{name}] {t.FullName}")
                    for ev in System.Enum.GetNames(t):
                        print(f"        {ev}")
