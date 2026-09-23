"""Probe whether DWSIM's native Vessel can perform and expose a two-liquid (LLE) split."""
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

feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
vessel_go = fs.AddObject(object_type.Vessel, 300, 0, "Vessel")
vessel = ded._simulation_object(vessel_go)

print("=== Vessel full property list (name : type : writable) ===")
for pr in vessel.GetType().GetProperties():
    try:
        canwrite = pr.CanWrite
    except Exception:
        canwrite = "?"
    print(f"  {pr.Name} : {pr.PropertyType.Name} : w={canwrite}")

print("\n=== Vessel methods (filtered) ===")
for m in vessel.GetType().GetMethods():
    n = m.Name
    if any(k in n for k in ("Flash", "Phase", "Liquid", "Spec", "Connect", "Outlet", "Inlet", "Mode", "Split", "Type")):
        print(f"  {n}({', '.join(p.Name for p in m.GetParameters())})")
