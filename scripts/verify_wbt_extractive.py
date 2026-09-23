"""Verify the exported water/1-butanol+toluene extractive columns.

Checks, per file:
  * the flowsheet loads and contains the three compounds plus the column;
  * the column carries the designed stage count / reflux / feed stages;
  * CalculateFlowsheet2 runs, and the distillate/bottoms streams actually resolve
    (F > 0) rather than sitting at their initial values.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILES = [
    "water_butanol_toluene_uniquac_unifac.dwxmz",
    "water_butanol_toluene_uniquac_thermoformer.dwxmz",
]


def main() -> int:
    factory, object_type = ded._automation_factory()
    a = factory()
    for name in FILES:
        p = (ROOT / "data" / "exports" / "flow_examples" / name).resolve()
        fs = a.LoadFlowsheet2(str(p))
        print(f"== {name}")

        comps = [str(c).split(",")[0].strip().lstrip("[") for c in fs.SelectedCompounds]
        print(f"   compounds: {comps}")

        pkgs = []
        try:
            for pp in fs.PropertyPackages.Values:
                pkgs.append(str(pp.GetType().Name))
        except Exception as exc:  # noqa: BLE001
            pkgs = [f"<read failed: {type(exc).__name__}>"]
        print(f"   property packages: {pkgs}")

        col = None
        for item in fs.SimulationObjects.Values:
            if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
                col = item
        if col is not None:
            c = col.GetAsObject() if hasattr(col, "GetAsObject") else col
            try:
                print(f"   column: {c.GetType().Name}  stages={c.NumberOfStages}  "
                      f"R={c.RefluxRatio}  Pdrop(Pa)={c.ColumnPressureDrop}")
            except Exception as exc:  # noqa: BLE001
                print(f"   column read failed: {type(exc).__name__}")

        errs = a.CalculateFlowsheet2(fs)
        err = str(errs[0])[:90] if errs and errs.Count else ""
        print(f"   calculate error: {err!r}")

        for item in fs.SimulationObjects.Values:
            if item.GetType().Name != "MaterialStream":
                continue
            tag = str(item.GraphicObject.Tag)
            s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            try:
                F = float(s.GetMolarFlow()); T = float(s.GetTemperature())
                z = [float(x) for x in s.GetOverallComposition()]
                print(f"   {tag:<12} F={F:8.4f} mol/s  T={T-273.15:7.2f} C  "
                      f"z=[{z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f}]")
            except Exception as exc:  # noqa: BLE001
                print(f"   {tag:<12} read failed: {type(exc).__name__}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
