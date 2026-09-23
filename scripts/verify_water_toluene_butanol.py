"""Verify the water/toluene + 1-butanol extractive exports, including mass balance.

Checks per file:
  * compounds, property package, column stage count / reflux / feed stages;
  * CalculateFlowsheet2 runs;
  * the distillate/bottoms streams resolve (F > 0) and the component balance closes
    against feed + entrainer -- the failure mode that broke the previous system.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILES = [
    "water_toluene_butanol_extractive_unifac.dwxmz",
    "water_toluene_butanol_extractive_thermoformer.dwxmz",
]
# feed = 1.0 (water .5 / toluene .5), entrainer = 2.0 1-butanol
EXPECTED_TOTAL = 3.0


def main() -> int:
    factory, object_type = ded._automation_factory()
    a = factory()
    for name in FILES:
        p = (ROOT / "data" / "exports" / "flow_examples" / name).resolve()
        fs = a.LoadFlowsheet2(str(p))
        print(f"== {name}")

        comps = [str(c).split(",")[0].strip().lstrip("[") for c in fs.SelectedCompounds]
        pkgs = [str(pp.GetType().Name) for pp in fs.PropertyPackages.Values]
        print(f"   compounds: {comps}")
        print(f"   property package: {pkgs}")

        col = None
        for item in fs.SimulationObjects.Values:
            if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
                col = item
        if col is not None:
            c = col.GetAsObject() if hasattr(col, "GetAsObject") else col
            try:
                print(f"   column: {c.GetType().Name}  stages={c.NumberOfStages}  "
                      f"R={c.RefluxRatio}")
            except Exception as exc:  # noqa: BLE001
                print(f"   column read failed: {type(exc).__name__}")

        errs = a.CalculateFlowsheet2(fs)
        err = str(errs[0])[:110] if errs and errs.Count else ""
        print(f"   calculate error: {err!r}")

        flows = {}
        for item in fs.SimulationObjects.Values:
            if item.GetType().Name != "MaterialStream":
                continue
            tag = str(item.GraphicObject.Tag)
            s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            try:
                F = float(s.GetMolarFlow())
                T = float(s.GetTemperature())
                z = [float(x) for x in s.GetOverallComposition()]
                flows[tag] = (F, z)
                print(f"   {tag:<12} F={F:8.4f} mol/s  T={T-273.15:7.2f} C  "
                      f"z=[{z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f}]")
            except Exception as exc:  # noqa: BLE001
                print(f"   {tag:<12} read failed: {type(exc).__name__}")

        # Component balance across the column.
        if {"Feed", "Entrainer", "Distillate", "Bottoms"} <= set(flows):
            f_in, z_in = flows["Feed"]
            f_en, z_en = flows["Entrainer"]
            f_d, z_d = flows["Distillate"]
            f_b, z_b = flows["Bottoms"]
            print("   --- 物料衡算 ---")
            for i, nm in enumerate(comps):
                cin = f_in * z_in[i] + f_en * z_en[i]
                cout = f_d * z_d[i] + f_b * z_b[i]
                rel = abs(cout - cin) / cin if cin else 0.0
                flag = "OK" if rel < 0.01 else "** 不平衡 **"
                print(f"     {nm:<12} in={cin:7.4f}  out={cout:7.4f}  "
                      f"rel.err={rel:8.2e}  {flag}")
            tot_in = f_in + f_en
            tot_out = f_d + f_b
            print(f"     {'total':<12} in={tot_in:7.4f}  out={tot_out:7.4f}  "
                  f"rel.err={abs(tot_out-tot_in)/tot_in:8.2e}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
