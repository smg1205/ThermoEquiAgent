"""Verify the generic-export .dwxmz files: package, components, estimates, balance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded  # noqa: E402

files = sorted(Path(r"C:\Temp").glob("generic-ternary-*/*.dwxmz"))
if not files:
    print("no files found")
    raise SystemExit(1)

factory, object_type = ded._automation_factory()
a = factory()
for f in files:
    fs = a.LoadFlowsheet2(str(f))
    print(f"== {f.name}  ({f.stat().st_size} bytes)")
    comps = [str(c).split(",")[0].strip().lstrip("[") for c in fs.SelectedCompounds]
    pkgs = [str(p.GetType().Name) for p in fs.PropertyPackages.Values]
    print(f"   compounds: {comps}")
    print(f"   package  : {pkgs}")

    col = None
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
            col = item.GetAsObject() if hasattr(item, "GetAsObject") else item
    if col is not None:
        print(f"   stages={col.NumberOfStages}  R={col.RefluxRatio}  "
              f"maxiter={getattr(col, 'MaxIterations', '?')}")
        try:
            ie = col.InitialEstimates
            t0 = float(ie.StageTemps[0].Value) - 273.15
            tn = float(ie.StageTemps[ie.StageTemps.Count - 1].Value) - 273.15
            print(f"   T profile persisted: {t0:.1f} -> {tn:.1f} C")
            print(f"   D={float(ie.DistillateFlowRate):.4f}  B={float(ie.BottomsFlowRate):.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"   estimate readback failed: {type(exc).__name__}")

    errs = a.CalculateFlowsheet2(fs)
    print(f"   calculate error: {str(errs[0])[:90] if errs and errs.Count else ''!r}")
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name != "MaterialStream":
            continue
        s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        try:
            F = float(s.GetMolarFlow()); T = float(s.GetTemperature())
            z = [float(x) for x in s.GetOverallComposition()]
            print(f"   {str(item.GraphicObject.Tag):<12} F={F:7.4f} T={T-273.15:7.2f}C "
                  f"z=[{z[0]:.3f},{z[1]:.3f},{z[2]:.3f}]")
        except Exception:  # noqa: BLE001
            pass
    print()
