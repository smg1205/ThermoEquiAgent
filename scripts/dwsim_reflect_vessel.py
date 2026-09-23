"""Inspect the DWSIM Vessel object members to find how to configure a flash."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from thermo_engine import dwsim_export as ded  # noqa: E402

factory, object_type = ded._automation_factory()
automation = factory()

fs = automation.CreateFlowsheet()
fs.AddCompound("Isopropanol")
fs.AddCompound("Water")
ded._add_property_package(fs, "NRTL")

feed = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
sep = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
vapor = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
lig = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")

# Inspect the Vessel object type + members
sep_obj = sep.GetAsObject() if hasattr(sep, "GetAsObject") else sep
print("=== Vessel object type:", type(sep_obj).__name__, "FullName:", sep_obj.GetType().FullName)

# list properties, looking for flash-relevant ones
print("\n=== Vessel properties (name : type) ===")
for pr in sep_obj.GetType().GetProperties():
    nm = pr.Name
    low = nm.lower()
    if any(k in low for k in ("flash", "spec", "vapor", "phase", "split", "mode", "type", "pressure", "temp")):
        try:
            val = pr.GetValue(sep_obj, None)
            print(f"  {nm}  ({pr.PropertyType.Name}) = {val}")
        except Exception as e:
            print(f"  {nm}  ({pr.PropertyType.Name}) = <err {type(e).__name__}>")

print("\n=== Vessel methods (flash-related) ===")
for m in sep_obj.GetType().GetMethods():
    low = m.Name.lower()
    if any(k in low for k in ("flash", "spec", "split", "vapor", "phase", "calculate")):
        print(f"  {m.Name}({', '.join(p.Name for p in m.GetParameters())})")
