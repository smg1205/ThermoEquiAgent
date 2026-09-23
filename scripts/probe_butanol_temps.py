"""Probe candidate replacement temperatures for the water/1-butanol DWSIM series."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded  # noqa: E402

CANDIDATES = [323.15, 328.15, 333.15, 353.15, 318.15, 308.15]


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    for T in CANDIDATES:
        fs = automation.CreateFlowsheet()
        fs.AddCompound("1-butanol")
        fs.AddCompound("Water")
        ded._add_property_package(fs, "NRTL")
        feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        v_go = fs.AddObject(object_type.Vessel, 350, 0, "V")
        vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vap")
        light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "L")
        heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "H")
        feed = ded._simulation_object(feed_go)
        feed.SetTemperature(T); feed.SetPressure(101325.0); feed.SetMolarFlow(1.0)
        feed.SetOverallComposition(ded._composition_argument([0.3, 0.7]))
        v = ded._simulation_object(v_go)
        try:
            v.FlashTemperature = T; v.FlashPressure = 101325.0
        except Exception:
            pass
        for f, t, fi, ti in ((feed_go, v_go, 0, 0), (v_go, vap_go, 0, 0),
                             (v_go, light_go, 1, 0), (v_go, heavy_go, 2, 0)):
            try:
                fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            except Exception:
                pass
        for go in (light_go, heavy_go, vap_go):
            try:
                ded._simulation_object(go).SetTemperature(T)
            except Exception:
                pass
        errs = automation.CalculateFlowsheet4(fs)
        err = str(errs[0])[:50] if errs and errs.Count else ""

        def vals(go):
            s = ded._simulation_object(go)
            return (float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
                    [float(x) for x in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)])
        lf, lz = vals(light_go)
        hf, hz = vals(heavy_go)
        ok = lf > 1e-6 and hf > 1e-6
        print(f"T={T}: {'SPLIT' if ok else 'FAIL '} light={lf:.5f} x={lz[0]:.4f} | "
              f"heavy={hf:.5f} x={hz[0]:.4f}" + (f"  ERR {err}" if err else ""))


if __name__ == "__main__":
    main()
