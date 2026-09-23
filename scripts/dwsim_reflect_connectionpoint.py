"""Reflect ConnectionPoint members to learn how to attach streams to Vessel LLE outlets."""
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

out = go.OutputConnectors[0]  # first output connector
print("ConnectionPoint type:", out.GetType().FullName)
print("\n=== ConnectionPoint properties ===")
for pr in out.GetType().GetProperties():
    try:
        print(f"  {pr.Name} : {pr.PropertyType.Name}")
    except Exception:
        pass
print("\n=== ConnectionPoint methods ===")
for m in out.GetType().GetMethods():
    print(f"  {m.Name}({', '.join(p.Name for p in m.GetParameters())})")

# also check IConnectionPoint interface member names
print("\n=== IConnectionPoint-like attrs on the object ===")
for attr in dir(out):
    if not attr.startswith("__"):
        if any(k in attr.lower() for k in ("attach", "connect", "stream", "object", "name", "type", "link")):
            try:
                print(f"  {attr} = {getattr(out, attr)!r}")
            except Exception:
                pass
