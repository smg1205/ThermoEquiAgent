"""Compute isobaric bubble points for 2-propanol-water via DWSIM (corrected).

Uses the real DWSIM 8 Automation API discovered via introspection:
  - trigger math:  automation.CalculateFlowsheet2(flowsheet)
  - vapor marker:  vapor_stream.GetMolarFlow()   (0 = all liquid; >0 = vapor present)
  - vapor comp:    vapor_stream.GetOverallComposition()

Bubble point T for a fixed x is where the vapor product molar flow first becomes
> 0 (i.e. the feed stops being all-liquid and the first bubble forms).  The
vapor composition at that T is read from the Vapor product stream.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_ipa_bubble.py
Output: docs/dwsim_ipa_bubble.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

IPA_CANDIDATES = ["Isopropanol", "2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]
WATER = "Water"
P_PA = 760.0 * 133.322
XS = [0.1, 0.3, 0.5, 0.7, 0.9]   # liquid mole fraction of 2-propanol
T_MIN_C, T_MAX_C = 60.0, 105.0
OUT = _REPO / "docs" / "dwsim_ipa_bubble.csv"


def _build(automation, object_type, x, t_c):
    from thermo_engine import dwsim_export as ded
    fs = automation.CreateFlowsheet()
    for cand in IPA_CANDIDATES:
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    else:
        raise RuntimeError("2-propanol compound not resolved")
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
    return automation, fs, ded._simulation_object(vapor)


def _vapor_flow(automation, fs, vapor):
    # Suppress DWSIM/Ipopt console spam that otherwise floods each flash.
    _quiet(automation)
    automation.CalculateFlowsheet2(fs)
    return float(vapor.GetMolarFlow())


def _quiet(automation):
    try:
        pkglist = list(automation.AvailablePropertyPackages)
        del pkglist
    except Exception:
        pass
    # Best-effort: lower DWSIM logging; if unavailable, just proceed.
    try:
        getattr(automation, "SetLogLevel")(3)
    except Exception:
        pass


def _read_vapor_comp(vapor):
    return [float(c) for c in list(vapor.GetOverallComposition())]


def bubble(x):
    """Return (T_bubble_C, y_ipa) for composition x via bisection on vapor flow."""
    lo, hi = T_MIN_C, T_MAX_C
    Tb, y = float("nan"), None
    # Ensure hi is above bubble (vapor present)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        automation, fs, vapor = _build(automation_global, object_type_global, x, mid)
        vf = _vapor_flow(automation, fs, vapor)
        if vf > 0:
            # vapor present -> we are at/above bubble; capture comp, move hi down
            Tb, y = mid, _read_vapor_comp(vapor)
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return Tb, (y[0] if y else float("nan"))


automation_global = None
object_type_global = None


def main():
    global automation_global, object_type_global
    from thermo_engine import dwsim_export as ded
    factory, object_type_global = ded._automation_factory()
    automation_global = factory()  # Automation3 instance, not the class

    rows = []
    for x in XS:
        Tb, y_ipa = bubble(x)
        rows.append({"x_ipa": round(x, 4), "T_bubble_C": round(Tb, 2), "y_ipa": round(y_ipa, 4)})
        print(f"x_IPA={x}: T_bubble={Tb:.2f} C   y_IPA={y_ipa:.4f}")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote: {OUT}")


if __name__ == "__main__":
    main()
