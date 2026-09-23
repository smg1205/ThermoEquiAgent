"""Verify the saved files carry the initial estimates, then attempt a calculation.

A file that merely accepts the estimate setters in-process is not enough: the values
must survive SaveFlowsheet and LoadFlowsheet2.  This checks the reloaded values and
then reports the solver outcome (converged composition or the exact error string).
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


def main() -> int:
    factory, object_type = ded._automation_factory()
    a = factory()
    for name in FILES:
        p = (ROOT / "data" / "exports" / "flow_examples" / name).resolve()
        fs = a.LoadFlowsheet2(str(p))
        print(f"== {name}")

        col = None
        for item in fs.SimulationObjects.Values:
            if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
                col = item
        if col is None:
            print("   no column found")
            continue
        c = col.GetAsObject() if hasattr(col, "GetAsObject") else col
        try:
            print(f"   stages={c.NumberOfStages}  R={c.RefluxRatio}  "
                  f"maxiter={getattr(c, 'MaxIterations', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"   column read failed: {type(exc).__name__}")

        # Re-read the persisted estimates.
        try:
            ie = c.InitialEstimates
            temps = [round(float(ie.StageTemps[i].Value) - 273.15, 1)
                     for i in range(min(3, ie.StageTemps.Count))]
            temps_end = [round(float(ie.StageTemps[i].Value) - 273.15, 1)
                         for i in range(max(0, ie.StageTemps.Count - 3), ie.StageTemps.Count)]
            print(f"   StageTemps persisted: first={temps} last={temps_end} C")
            print(f"   DistillateFlowRate={ie.DistillateFlowRate}  "
                  f"BottomsFlowRate={ie.BottomsFlowRate}")
            stage0 = ie.LiqCompositions[0]
            z0 = {str(k): round(float(stage0[k].Value), 4) for k in stage0.Keys}
            print(f"   LiqCompositions[0] persisted: {z0}")
        except Exception as exc:  # noqa: BLE001
            print(f"   estimate readback failed: {type(exc).__name__}: {str(exc)[:60]}")

        errs = a.CalculateFlowsheet2(fs)
        err = str(errs[0])[:160] if errs and errs.Count else ""
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
