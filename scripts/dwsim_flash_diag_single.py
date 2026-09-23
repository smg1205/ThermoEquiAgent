"""Diagnose a single DWSIM isobaric flash to see why bubble bisection returns nan."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from thermo_engine import dwsim_export as ded  # noqa: E402

factory, object_type = ded._automation_factory()
automation = factory()

fs = automation.CreateFlowsheet()
try:
    fs.AddCompound("Isopropanol")
    print("added Isopropanol")
except Exception as e:
    print("Isopropanol FAILED:", type(e).__name__, e)
    for c in ["2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]:
        try:
            fs.AddCompound(c)
            print("added", c)
            break
        except Exception as e2:
            print(c, "FAILED:", type(e2).__name__)

try:
    fs.AddCompound("Water")
    print("added Water")
except Exception as e:
    print("Water FAILED:", type(e).__name__, e)

ded._add_property_package(fs, "NRTL")
print("added NRTL")

feed = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
sep = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
vapor = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
lig = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
f0 = ded._simulation_object(feed)
f0.SetTemperature(90.0 + 273.15)
f0.SetPressure(760.0 * 133.322)
f0.SetMolarFlow(1.0)
f0.SetOverallComposition(ded._composition_argument([0.5, 0.5]))
fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)

print("flowsheet built; calculating...")
try:
    res = automation.CalculateFlowsheet2(fs)
    print("CalculateFlowsheet2 returned:", res, type(res).__name__)
except Exception as e:
    print("CalculateFlowsheet2 RAISED:", type(e).__name__, e)

v = ded._simulation_object(vapor)
try:
    vf = float(v.GetMolarFlow())
    print("vapor molar flow =", vf)
except Exception as e:
    print("GetMolarFlow RAISED:", type(e).__name__, e)

try:
    comp = [float(c) for c in list(v.GetOverallComposition())]
    print("vapor comp =", comp)
except Exception as e:
    print("GetOverallComposition RAISED:", type(e).__name__, e)

# also check liquid
l = ded._simulation_object(lig)
try:
    print("liquid molar flow =", float(l.GetMolarFlow()))
    print("liquid comp =", [float(c) for c in list(l.GetOverallComposition())])
except Exception as e:
    print("liquid read RAISED:", type(e).__name__, e)
