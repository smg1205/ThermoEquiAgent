"""Probe how to detect the bubble point on the installed DWSIM.

Strategy: read Vapor/Liquid product MOLAR FLOWS and the feed stream's own
vapor fraction, across a temperature scan, to find where Vapor flow becomes > 0
(the bubble point). Also dumps the feed stream's available getters/properties.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_diag_bubble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

IPA_CANDIDATES = ["Isopropanol", "2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]
WATER = "Water"
P_PA = 760.0 * 133.322
X_IPA = 0.5
T_SCAN = [70.0, 74.0, 78.0, 80.5, 82.0, 86.0, 90.0, 100.0]


def main():
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    # dump feed stream member names containing fraction/flow/phase
    fs_probe = automation.CreateFlowsheet()
    for cand in IPA_CANDIDATES:
        try: fs_probe.AddCompound(cand); break
        except Exception: pass
    fs_probe.AddCompound(ded._dwsim_compound_name(WATER))
    ded._add_property_package(fs_probe, "NRTL")
    feed_obj = fs_probe.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed_inner = ded._simulation_object(feed_obj)
    print("=== Feed stream (GetAsObject) fraction/flow/phase members ===")
    for n in sorted(dir(feed_inner)):
        ln = n.lower()
        if any(k in ln for k in ("fraction", "flow", "phase", "flash", "mole", "enthal", "temperature", "pressure")):
            print("  ", n)

    print("\n=== Temperature scan: Vapor/Liquid product molar flows ===")
    for t_c in T_SCAN:
        fs = automation.CreateFlowsheet()
        for cand in IPA_CANDIDATES:
            try: fs.AddCompound(cand); break
            except Exception: pass
        fs.AddCompound(ded._dwsim_compound_name(WATER))
        ded._add_property_package(fs, "NRTL")
        feed = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        sep = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
        vapor = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
        lig = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
        f0 = ded._simulation_object(feed)
        f0.SetTemperature(t_c + 273.15)
        f0.SetPressure(P_PA)
        f0.SetMolarFlow(1.0)
        f0.SetOverallComposition(ded._composition_argument([X_IPA, 1 - X_IPA]))
        fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
        fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
        fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)
        # calculate
        try:
            automation.CalculateFlowsheet2(fs)
        except Exception as e:
            print(f"  T={t_c}: CalculateFlowsheet2 err {e}")
            continue
        v, l = ded._simulation_object(vapor), ded._simulation_object(lig)
        def rd(obj, meth):
            m = getattr(obj, meth, None)
            if not callable(m): return None
            try: return m()
            except Exception as e: return f"ERR:{type(e).__name__}"
        Vf, Lf = rd(v, "GetMolarFlow"), rd(l, "GetMolarFlow")
        Vcomp, Lcomp = rd(v, "GetOverallComposition"), rd(l, "GetOverallComposition")
        def fmt(c):
            if c is None: return None
            if isinstance(c, str): return c
            try: return [round(float(x),4) for x in list(c)]
            except Exception: return c
        print(f"  T={t_c:5.1f} C | VaporFlow={Vf}  LiquidFlow={Lf} | Vcomp={fmt(Vcomp)}  Lcomp={fmt(Lcomp)}")


if __name__ == "__main__":
    main()
