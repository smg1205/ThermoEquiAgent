"""Test Vessel flash with explicit FlashTemperature/FlashPressure settings."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from thermo_engine import dwsim_export as ded  # noqa: E402

factory, object_type = ded._automation_factory()
automation = factory()


def make_and_flash(x, t_c):
    fs = automation.CreateFlowsheet()
    fs.AddCompound("Isopropanol")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    feed = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    sep = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
    vapor = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
    lig = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
    f0 = ded._simulation_object(feed)
    f0.SetTemperature(t_c + 273.15)
    f0.SetPressure(760.0 * 133.322)
    f0.SetMolarFlow(1.0)
    f0.SetOverallComposition(ded._composition_argument([x, 1 - x]))

    sep_obj = sep.GetAsObject() if hasattr(sep, "GetAsObject") else sep
    # Explicitly drive the vessel at the feed's T/P (isothermal-isobaric PT flash).
    try:
        sep_obj.FlashTemperature = t_c + 273.15
        sep_obj.FlashPressure = 760.0 * 133.322
        print(f"  set FlashTemperature={t_c + 273.15:.2f}, FlashPressure=101325")
    except Exception as e:
        print(f"  set flash T/P FAILED: {type(e).__name__}: {e}")

    fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)

    automation.CalculateFlowsheet2(fs)
    v = ded._simulation_object(vapor)
    l = ded._simulation_object(lig)
    return float(v.GetMolarFlow()), [float(c) for c in list(v.GetOverallComposition())], float(l.GetMolarFlow())


for t in [70.0, 80.0, 83.0, 85.0, 90.0]:
    try:
        vf, vcomp, lf = make_and_flash(0.5, t)
        print(f"T={t} C: vapor={vf:.5f} (x_ipa={vcomp[0]:.4f}) liquid={lf:.5f}")
    except Exception as e:
        print(f"T={t} C: ERROR {type(e).__name__}: {e}")
