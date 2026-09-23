"""Dump the Vessel GraphicObject's own connection ports (the real ConnectObjects indices)."""
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
go = vessel_go.GraphicObject
print("GraphicObject type:", go.GetType().FullName)

# properties mentioning port / connection / input / output
print("\n=== GraphicObject properties with 'port'/'connection'/'input'/'output' ===")
for pr in go.GetType().GetProperties():
    n = pr.Name.lower()
    if any(k in n for k in ("port", "connection", "input", "output", "connect", "segment")):
        try:
            print(f"  {pr.Name} : {pr.PropertyType.Name}")
        except Exception:
            pass

print("\n=== methods with 'connect'/'port'/'attach' ===")
for m in go.GetType().GetMethods():
    n = m.Name.lower()
    if any(k in n for k in ("connect", "port", "attach", "input", "output")):
        print(f"  {m.Name}({', '.join(p.Name for p in m.GetParameters())})")
