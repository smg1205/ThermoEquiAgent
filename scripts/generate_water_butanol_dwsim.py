"""Water / 1-butanol DWSIM flowsheet generation for report v5 (§1.4 / §2.3).

Water / 1-butanol is a **binary LLE** system (partially miscible, lower
critical solution temperature), not a VLE system.  The v5 report's third
validation system is therefore backed by native-``Vessel`` two-liquid-phase
flowsheets, one per experimental temperature:

    298.15 K / 313.15 K / 343.15 K / 353.15 K

Each flowsheet is ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid``
with the NRTL property package and DWSIM's built-in water/1-butanol interaction
parameters.  ``Light_Liquid`` is the 1-butanol-rich (organic) phase and
``Heavy_Liquid`` is the water-rich (aqueous) phase, matching the phase naming
used in the report.

This follows the established construction of ``scripts/run_water_butanol_dwsim.py``
and is regenerated here as a single self-contained command for the v5 assets.

All equilibrium numbers are computed by DWSIM itself; this script only builds
the flowsheet structure, drives the flash, and reads DWSIM's results back.

Run with a DWSIM + pythonnet capable interpreter (pythonnet needs full process
permission to initialise, so this cannot run under a constrained sandbox):

    conda activate thermo
    python scripts/generate_water_butanol_dwsim.py
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
P_PA = P_KPA * 1000.0

#: LLE temperatures with experimental tie-lines.  365.85 K sits too close to the
#: UCST for DWSIM to split, so the high end uses 353.15 K instead.
TEMPERATURES = [298.15, 313.15, 343.15, 353.15]
#: Overall feed composition inside the two-phase region (butanol / water).
FEED_Z = [0.3, 0.7]

OUTDIR = ROOT / "report" / "dwsim"
OUT_CSV = OUTDIR / "water_butanol_dwsim.csv"


def _sim_vals(go) -> tuple[float, list[float]]:
    """Return (molar flow, overall composition) of a material stream."""
    stream = ded._simulation_object(go)
    stype = stream.GetType()
    flow = float(stype.GetMethod("GetMolarFlow").Invoke(stream, None))
    comp = [float(x) for x in stype.GetMethod("GetOverallComposition").Invoke(stream, None)]
    return flow, comp


def _build_lle_flowsheet(automation, object_type, T_K: float):
    """Build ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid``."""
    fs = automation.CreateFlowsheet()
    # Each flowsheet needs its own AddCompound calls; compounds are not cached
    # across flowsheets (otherwise every flash collapses to a single phase).
    fs.AddCompound("1-butanol")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    v_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T_K)
    feed.SetPressure(P_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))

    # The Vessel defaults to a 298.15 K flash; it must be pinned to the feed
    # conditions or the two-phase split is evaluated at the wrong temperature.
    vessel = ded._simulation_object(v_go)
    try:
        vessel.FlashTemperature = T_K
        vessel.FlashPressure = P_PA
    except Exception:
        pass

    for src, dst, si, di in ((feed_go, v_go, 0, 0), (v_go, vap_go, 0, 0),
                             (v_go, light_go, 1, 0), (v_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(src.GraphicObject, dst.GraphicObject, si, di)
        except Exception:
            pass
    for go in (light_go, heavy_go, vap_go):
        try:
            ded._simulation_object(go).SetTemperature(T_K)
        except Exception:
            pass

    return fs, light_go, heavy_go


def run_lle(automation, object_type) -> list[dict[str, object]]:
    print("===== water / 1-butanol binary LLE (native Vessel, NRTL) =====")
    rows: list[dict[str, object]] = []

    for T in TEMPERATURES:
        fs, light_go, heavy_go = _build_lle_flowsheet(automation, object_type, T)

        errs = automation.CalculateFlowsheet4(fs)
        err = str(errs[0])[:70] if errs and errs.Count else ""

        lf, lz = _sim_vals(light_go)
        hf, hz = _sim_vals(heavy_go)
        name = f"water_butanol_LLE_{str(T).replace('.', 'p')}K.dwxmz"
        dest = OUTDIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        ded._save_flowsheet_via_temp(automation, fs, dest)

        print(f"  T={T}: light flow={lf:.5f} x_BuOH={lz[0]:.4f} | "
              f"heavy flow={hf:.5f} x_BuOH={hz[0]:.4f}  -> {name}"
              + (f"  ERR {err}" if err else ""))
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
    return rows


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = run_lle(automation, object_type)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
