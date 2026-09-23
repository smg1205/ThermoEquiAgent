"""DWSIM for ethyl acetate - n-propyl acetate - DMSO (ternary): bubble points + .dwxmz.

For each representative ternary composition (taken from the ThermoFormer/
experiment CSV), this script:
  1. finds the isobaric bubble temperature at 760 mmHg (bisection on Vapor-flow),
  2. saves a .dwxmz flowsheet whose Feed is set exactly at the bubble point,
  3. writes a comparison CSV (experiment / ThermoFormer / DWSIM).

Compounds: Ethyl acetate, N-propyl acetate, Dimethyl sulfoxide (all in DWSIM).

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_dmso_bubble.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

# (logical, dwsim-key-candidates)
COMPS = [
    ("ethyl acetate", ["Ethyl acetate"]),
    ("n-propyl acetate", ["N-propyl acetate", "n-Propyl acetate", "Propyl acetate"]),
    ("DMSO", ["Dimethyl sulfoxide", "DMSO"]),
]
P_PA = 760.0 * 133.322
T_MIN_C, T_MAX_C = 60.0, 130.0
INPUT_CSV = _REPO / "docs" / "prediction_ternary_dms.csv"
OUT_CSV = _REPO / "docs" / "dwsim_dmso_bubble.csv"
OUT_DIR = _REPO / "docs"

automation = None
object_type = None
RESOLVED = {}


def _resolve(fs, name, candidates):
    for c in candidates:
        try:
            fs.AddCompound(c)
            RESOLVED[name] = c
            return c
        except Exception:
            continue
    return None


def _make(x, t_c):
    from thermo_engine import dwsim_export as ded
    fs = automation.CreateFlowsheet()
    for name, cands in COMPS:
        if _resolve(fs, name, cands) is None:
            raise RuntimeError(f"compound not resolved: {name} ({cands})")
    ded._add_property_package(fs, "NRTL")
    feed = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    sep = fs.AddObject(object_type.Vessel, 450, 0, "Flash")
    vapor = fs.AddObject(object_type.MaterialStream, 750, -80, "Vapor")
    lig = fs.AddObject(object_type.MaterialStream, 750, 80, "Liquid")
    f0 = ded._simulation_object(feed)
    f0.SetTemperature(t_c + 273.15)
    f0.SetPressure(P_PA)
    f0.SetMolarFlow(1.0)
    f0.SetOverallComposition(ded._composition_argument([x[0], x[1], x[2]]))
    fs.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    fs.ConnectObjects(sep.GraphicObject, lig.GraphicObject, 1, 0)
    return fs, ded._simulation_object(vapor)


def _bubble(x):
    lo, hi = T_MIN_C, T_MAX_C
    y_orig = [float("nan")] * 3
    Tb = float("nan")
    last_vapor = 1.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        fs, vapor = _make(x, mid)
        automation.CalculateFlowsheet2(fs)
        vf = float(vapor.GetMolarFlow())
        if vf > 0:
            if vf < last_vapor:
                last_vapor = vf
                Tb = mid
                try:
                    y_orig = [float(c) for c in list(vapor.GetOverallComposition())]
                except Exception:
                    pass
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return Tb, y_orig


def main():
    global automation, object_type
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    idxs = sorted(set(int(round(i * (n - 1) / 5.0)) for i in range(6)))
    reps = [rows[i] for i in idxs]

    out = []
    for e in reps:
        x = [float(e["x_etac"]), float(e["x_npac"]), float(e["x_dmso"])]
        Tb, y_orig = _bubble(x)
        y_etac, y_npac = y_orig[0], y_orig[1]
        if len(y_orig) >= 3:
            y_dmso = y_orig[2]
        else:
            y_npac, y_dmso = float("nan"), float("nan")
        fname = f"DmsoBubble_x_etac{str(round(x[0],3)).replace('.', 'p')}.dwxmz"
        fs, _vapor = _make(x, Tb)
        automation.CalculateFlowsheet2(fs)
        ded._save_flowsheet(automation, fs, OUT_DIR / fname)
        out.append({"x_etac": round(x[0],4), "x_npac": round(x[1],4), "x_dmso": round(x[2],4),
                    "T_exp_C": e["T_exp_C"], "T_tf_C": e["T_tf_C"], "T_dwsim_C": round(Tb,2),
                    "y_etac_dwsim": round(y_etac,4), "y_npac_dwsim": round(y_npac,4),
                    "y_dmso_dwsim": round(y_dmso,4), "file": fname})
        print(f"x=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f}]  T_exp={e['T_exp_C']}  T_tf={e['T_tf_C']}  "
              f"T_dwsim={Tb:.2f}  y_dwsim=[{y_etac:.4f},{y_npac:.4f},{y_dmso:.4f}]  -> {fname}")
    print("resolved compounds:", RESOLVED)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\nwrote:", OUT_CSV)


if __name__ == "__main__":
    main()
