"""Test the Agent's routing end-to-end via run_extractive_export."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from agent import extractive_distillation as ed  # noqa: E402

outdir = Path(tempfile.mkdtemp(prefix="agent-route-"))
CASES = [
    "导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
    "用 UNIQUAC 导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
    "导出乙腈/甲苯/四氢呋喃三元VLE的DWSIM萃取精馏文件",
    "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件",
    "用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim",
]

for msg in CASES:
    print("=" * 78)
    print("message:", msg)
    generic = ed.is_generic_ternary_extractive_request(msg)
    fixed = ed.is_eac_npac_dmso_vle_export_request(msg)
    print(f"  router: generic={generic}  fixed={fixed}")
    try:
        p = ed.run_extractive_export(msg, export_dir=str(outdir))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {str(exc)[:200]}")
        continue
    print(f"  status  : {p.status}")
    print(f"  file_id : {p.file_id}")
    if p.file_id:
        f = outdir / f"{p.file_id}.dwxmz"
        print(f"  produced: {f.exists()} ({f.stat().st_size if f.exists() else 0} bytes)")
    print("  message :", p.message.replace("\n", "\n            "))
    print()
print("outdir:", outdir)
