"""Inspect Vessel's connection ports to see if it natively exposes two liquid outlets."""
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
        fs.AddCompound(cand)
        break
    except Exception:
        continue
fs.AddCompound("Water")
ded._add_property_package(fs, "UNIQUAC")

vessel_go = fs.AddObject(object_type.Vessel, 300, 0, "Vessel")
vessel = ded._simulation_object(vessel_go)

print("=== GetConnectionPortsList() ===")
try:
    ports = vessel.GetConnectionPortsList()
    for p in ports:
        print("  ", type(p).__name__, "->", p)
except Exception as e:
    print("  err:", type(e).__name__, e)

print("\n=== GetConnectionPortsInfo() ===")
try:
    info = vessel.GetConnectionPortsInfo()
    print("  type:", type(info).__name__)
    # could be a list of tuples/dicts; dump with reflection
    if hasattr(info, "__iter__") and not isinstance(info, str):
        for item in info:
            print("  item:", repr(item))
            if hasattr(item, "__iter__") and not isinstance(item, str):
                for sub in item:
                    print("     ", repr(sub))
    else:
        print("  value:", repr(info))
except Exception as e:
    print("  err:", type(e).__name__, e)

print("\n=== does Vessel expose a 'Phase'/'Separator' spec through CalcSegments? check CalculationMode enum ===")
try:
    modes = vessel.GetCalculationModes()
    for m in modes:
        print("  mode:", repr(m))
except Exception as e:
    print("  GetCalculationModes err:", type(e).__name__, e)
try:
    cm = vessel.GetType().GetProperty("CalculationMode")
    import System
    print("  CalculationMode enum:", System.Enum.GetNames(cm.PropertyType))
except Exception as e:
    print("  CalculationMode err:", type(e).__name__, e)
