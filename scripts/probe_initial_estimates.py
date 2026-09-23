"""Inspect the column's InitialEstimates object to set composition estimates."""
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
ded._add_property_package(fs, "UNIQUAC")
col = ded._simulation_object(fs.AddObject(object_type.DistillationColumn, 250, 0, "C"))
ie = col.InitialEstimates
print("InitialEstimates type:", ie.GetType().Name)
print("\n=== properties ===")
for p in ie.GetType().GetProperties():
    try:
        print(f"  {p.Name} : {p.PropertyType.Name}")
    except Exception:
        pass
print("\n=== methods ===")
for m in ie.GetType().GetMethods():
    if m.Name.startswith(("get_", "set_", "Add", "Clear", "Set")):
        params = ", ".join(pp.ParameterType.Name for pp in m.GetParameters())
        print(f"  {m.Name}({params})")
