"""Read back the exported extractive columns' convergence state and stream results.

Uses the same reflection-based access the repo's other DWSIM readers rely on
(``GetType().GetMethod(...).Invoke(...)``), which works for both the pythonnet
dynamic and interface-typed handles.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FLOWS = [
    "acn_toluene_thf_extractive_unifac.dwxmz",
    "acn_toluene_thf_extractive_thermoformer.dwxmz",
]


def call(obj, name, *args):
    m = obj.GetType().GetMethod(name)
    return m.Invoke(obj, list(args) if args else None)


def main() -> int:
    factory, object_type = ded._automation_factory()
    a = factory()
    for name in FLOWS:
        p = (ROOT / "data" / "exports" / "flow_examples" / name).resolve()
        fs = a.LoadFlowsheet(str(p))
        errs = a.CalculateFlowsheet2(fs)
        err = str(errs[0])[:100] if errs and errs.Count else ""
        print(f"== {name}")
        print(f"   calculate error: {err!r}")
        for key, obj in fs.SimulationObjects.items():
            go = obj.GraphicObject
            if go.ObjectType != object_type.MaterialStream:
                continue
            s = ded._simulation_object(go)
            try:
                F = float(call(s, "GetMolarFlow"))
                T = float(call(s, "GetTemperature"))
                z = [float(x) for x in call(s, "GetOverallComposition")]
                print(f"   {go.Tag:<12} F={F:8.4f} mol/s  T={T-273.15:7.2f} C  "
                      f"z=[{z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f}] mf")
            except Exception as exc:  # noqa: BLE001
                print(f"   {go.Tag:<12} read failed: {type(exc).__name__}: {str(exc)[:70]}")
        # column status
        for key, obj in fs.SimulationObjects.items():
            if obj.GraphicObject.ObjectType == object_type.DistillationColumn:
                c = ded._simulation_object(obj.GraphicObject)
                for attr in ("ColumnType", "NumberOfStages", "RefluxRatio", "CondenserPressure",
                             "ReboilerPressure", "CondenserTemperature", "ReboilerTemperature"):
                    try:
                        print(f"   [col] {attr} = {getattr(c, attr)}")
                    except Exception:
                        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
