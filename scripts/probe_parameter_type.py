"""Determine how to populate DWSIM's InitialEstimates lists.

StageTemps / LiqMolarFlows / VapMolarFlows are List<Parameter>, while
LiqCompositions / VapCompositions are List<Dictionary<String, Parameter>>
(one dictionary per stage, keyed by compound name).  Inspect the Parameter type and
confirm the working assignment pattern before rewriting the exporter.
"""
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

print("=== Parameter type members ===")
lst = ie.StageTemps
item = lst[0]
print("current item type:", item.GetType().FullName)
for p in item.GetType().GetProperties():
    print(f"  {p.Name} : {p.PropertyType.Name}  (canWrite={p.CanWrite})")

print("\n=== try writing StageTemps[0] via .Value ===")
try:
    item.Value = 300.0
    print("  item.Value = 300.0 OK; read back:", float(ie.StageTemps[0].Value))
except Exception as e:
    print("  failed:", type(e).__name__, str(e)[:80])

print("\n=== composition dictionary keys ===")
comp = ie.LiqCompositions[0]
print("  type:", comp.GetType().FullName)
try:
    print("  keys:", [str(k) for k in comp.Keys])
    for k in comp.Keys:
        v = comp[k]
        print(f"    {k} -> {v.GetType().Name} value={v.Value}")
except Exception as e:
    print("  key read failed:", type(e).__name__, str(e)[:80])

print("\n=== try writing a composition entry ===")
try:
    k0 = list(comp.Keys)[0]
    comp[k0].Value = 0.7
    print(f"  set {k0}.Value = 0.7 OK")
except Exception as e:
    print("  failed:", type(e).__name__, str(e)[:80])
