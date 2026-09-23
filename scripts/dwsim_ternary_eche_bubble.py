"""DWSIM isobaric bubble points for ethanol-cyclohexane-ethyl propionate (ternary).

For a set of ternary liquid compositions (x1,x2,x3) at 760 mmHg, bisection on the
Vapor product molar flow finds each composition's bubble temperature (same method
as scripts/dwsim_ipa_bubble.py). DWSIM supplies its own built-in Antoine /
property data for all three compounds, so no external Antoine is needed here.

The compositions come from the ThermoFormer/experimental ternary CSV so the DWSIM
row can be compared with experiment and ThermoFormer in one table.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_ternary_eche_bubble.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

# DWSIM compound key candidates (first resolvable wins).
COMP_CANDIDATES = {
    "ethanol": ["Ethanol"],
    "cyclohexane": ["Cyclohexane"],
    "ethyl propionate": ["Ethyl propionate", "Propyl acetate", "Ethylpropionate"],
}
P_PA = 760.0 * 133.322
T_MIN_C, T_MAX_C = 60.0, 110.0
INPUT_CSV = _REPO / "docs" / "prediction_ternary_eche.csv"
OUT_CSV = _REPO / "docs" / "dwsim_ternary_eche_bubble.csv"

automation = None
object_type = None


def _add_compound(fs, key, name):
    import sys as _s
    from thermo_engine import dwsim_export as ded
    for cand in COMP_CANDIDATES[name]:
        try:
            fs.AddCompound(cand)
            return cand
        except Exception:
            continue
    return None


def _make(x, t_c):
    from thermo_engine import dwsim_export as ded
    fs = automation.CreateFlowsheet()
    names = ["ethanol", "cyclohexane", "ethyl propionate"]
    resolved = [_add_compound(fs, None, nm) for nm in names]
    if any(r is None for r in resolved):
        raise RuntimeError("one compound could not be resolved: " + str(resolved))
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
    y = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fs, vapor = _make(x, mid)
        automation.CalculateFlowsheet2(fs)
        vf = float(vapor.GetMolarFlow())
        if vf > 0:
            hi = mid
            try:
                y = [float(c) for c in list(vapor.GetOverallComposition())]
            except Exception:
                pass
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return hi, (y[0] if y else float("nan"))


def main():
    global automation, object_type
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    # read representative points from the ternary prediction CSV
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # pick ~6 representative compositions spread across the set
    n = len(rows)
    idxs = sorted(set(int(round(i * (n - 1) / 5)) for i in range(6)))
    reps = [rows[i] for i in idxs]

    out = []
    for e in reps:
        x = [float(e["x_etoh"]), float(e["x_chx"]), float(e["x_ester"])]
        Tb, y_etoh = _bubble(x)
        out.append({"x_etoh": round(x[0], 4), "x_chx": round(x[1], 4), "x_ester": round(x[2], 4),
                    "T_exp_C": e["T_exp_C"], "T_tf_C": e["T_tf_C"], "T_dwsim_C": round(Tb, 2),
                    "y_etoh_dwsim": round(y_etoh, 4)})
        print(f"x=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f}]  T_exp={e['T_exp_C']}  "
              f"T_tf={e['T_tf_C']}  T_dwsim={Tb:.2f}  y_etoh={y_etoh:.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
