"""Confirm DWSIM availability of extra components and list Excel terineries
whose three components all exist in DWSIM AND are atmospheric + alcohol/ester-ish.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_check_more.py
"""
from __future__ import annotations

import sys, os
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

EXTRA = [
    "Dimethyl sulfoxide", "DMSO", "n-Hexane", "diisopropyl ether", "Diisopropyl ether",
    "2-butanone", "Acetone", "Methyl propionate", "Ethyl acetate", "N-propyl acetate",
    "1-propanol", "2-propanol", "n-propyl acetate", "Propyl acetate",
    "dimethyl carbonate", "Dimethyl carbonate", "Acetonitrile", "Benzene", "Toluene",
    "1,4-dioxane", "4-methyl-2-pentanone", "Methyl isobutyl ketone", "3-methyl-2-butanone",
]


def dwsim_available(automation, names):
    out = {}
    for n in names:
        fs = automation.CreateFlowsheet()
        try:
            fs.AddCompound(n); out[n] = "ADD-OK"
        except Exception:
            out[n] = "fail"
    return out


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()
    print("== extended AddCompound probe ==")
    avail = dwsim_available(automation, EXTRA)
    for k, v in avail.items():
        print(f"  {k:28s} {v}")

    # Build map of which DWSIM-ok labels map from SMILES-like need: we'll instead
    # print Excel ternary systems and their component names, then the user picks.
    import openpyxl
    from collections import defaultdict
    wb = openpyxl.load_workbook(_REPO / "docs" / "ternary_vle_english.xlsx", read_only=True, data_only=True)
    ws = wb["ternary_vle_data"]
    ri = ws.iter_rows(values_only=True)
    h = next(ri); ix = {n: i for i, n in enumerate(h)}
    agg = defaultdict(lambda: {"n": 0, "Pmax": 0.0, "Pmin": 1e9, "names": ()})
    for r in ri:
        names = tuple(str(r[ix['component_%d_original_name' % k]] or "").strip() for k in (1, 2, 3))
        sm = tuple(str(r[ix['smiles%d' % k]] or "").strip() for k in (1, 2, 3))
        if not all(names) or not all(sm):
            continue
        key = tuple(sorted(names))
        a = agg[key]
        a["n"] += 1
        P = r[ix["pressure_mmhg"]]
        if P is not None:
            P = float(P); a["Pmax"] = max(a["Pmax"], P); a["Pmin"] = min(a["Pmin"], P)
        a["names"] = names
    wb.close()
    print("\n== atmospheric ternary systems (n>=15, near 760) ==")
    cand = [(k, v) for k, v in agg.items() if v["n"] >= 15 and v["Pmax"] <= 800 and v["Pmin"] >= 600]
    cand.sort(key=lambda kv: -kv[1]["n"])
    for k, v in cand[:25]:
        print(f"  n={v['n']:4d} P={v['Pmin']:.0f}-{v['Pmax']:.0f}  {v['names']}")


if __name__ == "__main__":
    main()
