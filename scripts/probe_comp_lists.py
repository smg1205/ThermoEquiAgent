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
for attr in ("StageTemps", "LiqMolarFlows", "LiqCompositions", "VapCompositions"):
    lst = getattr(ie, attr)
    t = lst.GetType()
    print(f"{attr}: type={t.Name} count={lst.Count}")
    # element type
    try:
        for iface in t.GetInterfaces():
            if "Generic" in iface.Name:
                for ga in iface.GetGenericArguments():
                    print(f"    element type: {ga.Name}")
    except Exception as e:
        print(f"    iface read failed: {e}")
    # try Add with different payloads on LiqCompositions
    if attr == "LiqCompositions":
        for payload, label in ((0.3, "double"), ([0.3, 0.3, 0.4], "list")):
            tmp = getattr(ie, attr)
            try:
                tmp.Add(payload)
                print(f"    Add({label}) OK")
            except Exception as e:
                print(f"    Add({label}) failed: {type(e).__name__}")
