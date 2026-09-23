"""End-to-end test of the generic ternary extractive export.

Exercises several systems and property packages through the real DWSIM exporter and
reports the produced file plus the design it was built from.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from agent import generic_ternary_export as gte  # noqa: E402

CASES = [
    "导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
    "用 UNIQUAC 导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
    "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件",
]

outdir = Path(tempfile.mkdtemp(prefix="generic-ternary-"))
print("export dir:", outdir, "\n")

for msg in CASES:
    print("=" * 78)
    print("message:", msg)
    req = gte.parse_request(msg)
    if req is None:
        print("  parse -> None (skipped)")
        continue
    print(f"  parsed: {req.light} / {req.heavy} + {req.entrainer}  "
          f"pkg={req.property_package} purity={req.purity} rec={req.recovery}")
    try:
        payload = gte.run_generic_ternary_extractive_export(msg, export_dir=str(outdir))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {str(exc)[:200]}")
        continue
    print(f"  status: {payload.status}")
    print(f"  file_id: {payload.file_id}")
    print(f"  message: {payload.message}")
    if payload.file_id:
        f = outdir / f"{payload.file_id}.dwxmz"
        print(f"  produced: {f.exists()}  {f.stat().st_size if f.exists() else 0} bytes")
    print()
