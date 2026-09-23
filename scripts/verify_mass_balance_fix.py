"""Confirm the mass-balance failure is resolved and check solver behaviour.

Before the fix every export carried `R: Product_Molar_Flow_Rate = 0.0`, so the
rigorous column had no bottoms specification and DWSIM aborted with
"Failed to fulfill mass balance for Water: Relative Error = 0.99924".

This test loads the regenerated file, reports the attached specifications, and runs
the solver, reporting the outcome either way.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402,F401

from agent import generic_ternary_export as gte  # noqa: E402
from thermo_engine import dwsim_export as ded  # noqa: E402

CASES = [
    ("用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim，纯度 0.99", "ThermoFormer 0.99"),
    ("导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件", "UNIFAC 默认 0.95"),
]

outdir = Path(tempfile.mkdtemp(prefix="mb-verify-"))
for msg, label in CASES:
    print("=" * 78)
    print(f"{label}: {msg}")
    p = gte.run_generic_ternary_extractive_export(msg, export_dir=str(outdir))
    print(f"  export status: {p.status}")
    if not p.file_id:
        print("  no file produced:", p.message.replace("\n", " | ")[:200])
        continue
    f = outdir / f"{p.file_id}.dwxmz"
    print(f"  file: {f.name} ({f.stat().st_size} bytes)")

    factory, object_type = ded._automation_factory()
    a = factory()
    fs = a.LoadFlowsheet2(str(f))
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name == "DistillationColumn":
            c = item.GetAsObject() if hasattr(item, "GetAsObject") else item
            specs = {k: (str(c.Specs[k].SType), float(c.Specs[k].SpecValue))
                     for k in c.Specs.Keys}
            print(f"  specs: {specs}")
            print(f"  R={c.RefluxRatio}  stages={c.NumberOfStages}  "
                  f"maxiter={getattr(c, 'MaxIterations', '?')}")

    errs = a.CalculateFlowsheet2(fs)
    err = str(errs[0])[:200] if errs and errs.Count else ""
    print(f"  calculate error: {err!r}")
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name != "MaterialStream":
            continue
        s = item.GetAsObject() if hasattr(item, "GetAsObject") else item
        try:
            F = float(s.GetMolarFlow()); T = float(s.GetTemperature())
            z = [float(x) for x in s.GetOverallComposition()]
            print(f"    {str(item.GraphicObject.Tag):<12} F={F:7.4f} "
                  f"T={T-273.15:7.2f}C z={[round(v,4) for v in z]}")
        except Exception:  # noqa: BLE001
            pass
    print()
print("outdir:", outdir)
