"""DWSIM side of the water / 1-butanol three-source comparison.

One native-Vessel LLE flowsheet per temperature (298.15 / 313.15 / 343.15 / 365.85 K),
NRTL with DWSIM's built-in water/1-butanol pair.  Run as its own process so the
ThermoFormer src path never pollutes the imports.

Writes report/dwsim/water_butanol_LLE_<T>K.dwxmz and prints the two phase
compositions for each temperature.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402

P_KPA = 101.325
# 365.85 K sits too close to the UCST for DWSIM to split, so the high end of the
# series uses 353.15 K instead (both have experimental tie-lines).
TEMPERATURES = [298.15, 313.15, 343.15, 353.15]
OUTDIR = ROOT / "report" / "dwsim"
OUT_CSV = OUTDIR / "water_butanol_dwsim.csv"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    rows = []

    for T in TEMPERATURES:
        fs = automation.CreateFlowsheet()
        fs.AddCompound("1-butanol")
        fs.AddCompound("Water")
        ded._add_property_package(fs, "NRTL")

        feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
        v_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
        vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
        light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
        heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

        feed = ded._simulation_object(feed_go)
        feed.SetTemperature(T)
        feed.SetPressure(P_KPA * 1000.0)
        feed.SetMolarFlow(1.0)
        feed.SetOverallComposition(ded._composition_argument([0.3, 0.7]))

        v = ded._simulation_object(v_go)
        try:
            v.FlashTemperature = T
            v.FlashPressure = P_KPA * 1000.0
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
        err = str(errs[0])[:70] if errs and errs.Count else ""

        def vals(go):
            s = ded._simulation_object(go)
            return (float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
                    [float(x) for x in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)])

        lf, lz = vals(light_go)
        hf, hz = vals(heavy_go)
        name = f"water_butanol_LLE_{str(T).replace('.', 'p')}K.dwxmz"
        dest = OUTDIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        ded._save_flowsheet_via_temp(automation, fs, dest)

        rows.append({
            "T_K": T,
            "light_flow_mol_s": round(lf, 5),
            "dwsim_x_butanol_organic": round(lz[0], 4),
            "dwsim_x_water_organic": round(lz[1], 4),
            "heavy_flow_mol_s": round(hf, 5),
            "dwsim_x_butanol_aqueous": round(hz[0], 4),
            "dwsim_x_water_aqueous": round(hz[1], 4),
            "file": name,
            "error": err,
        })
        print(f"T={T}: light flow={lf:.5f} x_BuOH={lz[0]:.4f} | heavy flow={hf:.5f} x_BuOH={hz[0]:.4f}"
              f"  -> {name}" + (f"  ERR {err}" if err else ""))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
