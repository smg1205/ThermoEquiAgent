"""Generate .dwxmz files positioned exactly AT each composition's bubble point.

For each chosen liquid composition x_IPA, this script:
  1. finds the bubble temperature Tb via DWSIM (bisection on vapor flow, same as
     dwsim_ipa_bubble.py),
  2. saves a flowsheet whose Feed temperature is set exactly to Tb as
     docs/BubbleData_ipa_water_<x>.dwxmz.

Because the feed is set right at the bubble temperature, opening the file in the
DWSIM GUI shows the bubble state directly: the Vapor product molar flow is ~0
and just turning positive, and the Flash / Vapor stream reports the bubble-phase
composition and temperature. No phase-envelope menu has to be found.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_ipa_bubble_files.py
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
XS = [0.1, 0.3, 0.5, 0.7, 0.9]
T_MIN_C, T_MAX_C = 60.0, 105.0
OUT_DIR = _REPO / "docs"

automation = None
object_type = None


def _make(x, t_c):
    """Build feed -> flash flowsheet at (x, t_c). Return (fs, vapor sim object)."""
    from thermo_engine import dwsim_export as ded
    fs = automation.CreateFlowsheet()
    ok = False
    for cand in IPA_CANDIDATES:
        try:
            fs.AddCompound(cand); ok = True; break
        except Exception:
            continue
    if not ok:
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
    return fs, ded._simulation_object(vapor)


def _vapor_flow(fs, vapor):
    automation.CalculateFlowsheet2(fs)
    return float(vapor.GetMolarFlow())


def _bubble_temperature(x):
    lo, hi = T_MIN_C, T_MAX_C
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fs, vapor = _make(x, mid)
        if _vapor_flow(fs, vapor) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return hi


def main():
    global automation, object_type
    from thermo_engine import dwsim_export as ded

    factory, object_type = ded._automation_factory()
    automation = factory()

    out = []
    for x in XS:
        Tb = _bubble_temperature(x)
        # Rebuild a file whose feed is at the bubble temperature.
        fs, vapor = _make(x, Tb)
        automation.CalculateFlowsheet2(fs)
        vf_b = float(vapor.GetMolarFlow())
        y_ipa = float(list(vapor.GetOverallComposition())[0])
        fname = f"BubbleData_ipa_x{str(x).replace('.', 'p')}.dwxmz"
        dest = OUT_DIR / fname
        ded._save_flowsheet(automation, fs, dest)
        out.append({"x_ipa": round(x, 4), "T_bubble_C": round(Tb, 2),
                    "vapor_frac_at_bubble": round(vf_b, 6), "y_ipa": round(y_ipa, 4), "file": fname})
        print(f"x_IPA={x}: T_bubble={Tb:.2f} C  vaporFrac={vf_b:.2e}  y_IPA={y_ipa:.4f}  -> {fname}")

    index = OUT_DIR / "dwsim_ipa_bubble_files.csv"
    with open(index, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote index: {index}")
    print("Open any .dwxmz in the DWSIM GUI: the Feed temperature is set AT the")
    print("bubble point, so the Flash shows vapor fraction ~0 (start of boiling).")


if __name__ == "__main__":
    main()
