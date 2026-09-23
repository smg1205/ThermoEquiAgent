"""Diagnose how to run a DWSIM flash and read bubble results on this install.

For one composition (x_IPA=0.5) at a few flash temperatures, it:
  - triggers DWSIM calculation via Automation.CalculateFlowsheet,
  - tries multiple ways to read the vapor fraction / bubble temperature,
  - prints the raw returns so we can fix the parsing for the bubble script.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_diag_flash.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

IPA_CANDIDATES = ["Isopropanol", "2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]
WATER = "Water"
P_PA = 760.0 * 133.322
TRIALS_C = [70.0, 80.0, 90.0, 100.0]
X_IPA = 0.5


def _make(automation, object_type, x, t_c):
    from thermo_engine import dwsim_export as ded
    fs = automation.CreateFlowsheet()
    for cand in IPA_CANDIDATES:
        try:
            fs.AddCompound(cand); break
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
    f0.SetOverallComposition(ded._composition_argument([x, 1 - x]))
    fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)
    return automation, fs, ded._simulation_object(vapor), ded._simulation_object(lig)


def main():
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    for t_c in TRIALS_C:
        print(f"\n=== Flash at T={t_c} C, x_IPA={X_IPA} ===")
        automation, fs, v, l = _make(automation, object_type, X_IPA, t_c)
        # trigger calculate (Automation.CalculateFlowsheet variants; flowsheet also)
        done = False
        for fname in ("CalculateFlowsheet", "CalculateFlowsheet2", "CalculateFlowsheet3", "CalculateFlowsheet4"):
            m = getattr(automation, fname, None)
            if callable(m):
                try:
                    m(fs)
                    done = True
                    print(f"  calculated via Automation.{fname}")
                    break
                except Exception as e:
                    print(f"  Automation.{fname} err: {type(e).__name__}: {e}")
        if not done:
            r = ded._first_call(fs, ("RequestCalculationAndWait", "RequestCalculation"))
            print("  flowsheet calc:", r)
        # read temperature & vapor fraction
        for obj, tag in ((v, "Vapor"), (l, "Liquid")):
            T = None
            try: T = obj.GetTemperature()
            except Exception as e: print(f"  {tag} GetTemperature err: {e}")
            print(f"  {tag} GetTemperature = {T}")
            for meth in ("GetOverallTPFraction", "GetTPFraction", "GetOverallVaporFraction"):
                m = getattr(obj, meth, None)
                if callable(m):
                    try:
                        raw = m()
                        print(f"  {tag} {meth}() -> {raw}  (type={type(raw).__name__})")
                    except Exception as e:
                        print(f"  {tag} {meth}() err: {type(e).__name__}: {e}")
        # composition read
        try:
            comp = v.GetOverallComposition()
            print(f"  Vapor GetOverallComposition() -> {list(comp)} (type={type(comp).__name__})")
        except Exception as e:
            print(f"  Vapor GetOverallComposition err: {e}")


if __name__ == "__main__":
    main()
