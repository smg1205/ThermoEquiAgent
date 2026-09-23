"""Find the correct signature for SetInitialMolarCompositionEstimates."""
import inspect
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

print("=== overloads of the estimate setters ===")
for name in ("SetInitialMolarCompositionEstimates", "SetInitialTemperatureEstimates",
             "SetInitialLiquidMolarFlowEstimates", "SetInitialVaporMolarFlowEstimates"):
    m = getattr(col, name, None)
    print(f"\n{name}: callable={callable(m)}")
    if not callable(m):
        continue
    for meth in m.GetType().GetMethods() if hasattr(m, "GetType") else []:
        if meth.Name == name:
            params = ", ".join(f"{p.ParameterType.Name} {p.Name}"
                               for p in meth.GetParameters())
            print(f"    {name}({params})")

# Also check what .NET type the column is and whether it exposes public props
print("\n=== Stage-related public properties ===")
t = col.GetType()
for p in t.GetProperties():
    if any(k in p.Name.lower() for k in ("estimat", "initial")):
        print(f"    {p.Name} : {p.PropertyType.Name}")
