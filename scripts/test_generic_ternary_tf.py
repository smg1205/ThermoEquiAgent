"""Verify the ThermoFormer branch of the generic ternary export in a clean process.

``clr`` (pythonnet) and ``torch`` cannot coexist once the CLR is initialised: after
that, importing torch raises ``OSError [WinError 127] ... torch\\lib\\shm.dll``.
The ThermoFormer design needs torch, so it must be imported before any DWSIM
Automation call happens.  This test imports torch first (the realistic server
ordering, where the ThermoFormer backend is loaded lazily on first use) and checks
that the whole path completes.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

# Import torch up front: this is what the running service does when the
# ThermoFormer backend is initialised before DWSIM is touched.
import torch  # noqa: E402,F401

from agent import extractive_distillation as ed  # noqa: E402

outdir = Path(tempfile.mkdtemp(prefix="tf-ternary-"))
CASES = [
    "用 ThermoFormer 导出 水/甲苯/1-丁醇 萃取精馏 dwsim",
    "用 ThermoFormer 导出 乙腈/甲苯/四氢呋喃 三元 VLE 萃取精馏 dwsim",
    "用 ThermoFormer 导出 乙醇/水/乙二醇 萃取精馏 dwsim",  # 不在 SMILES 表内
]

for msg in CASES:
    print("=" * 78)
    print("message:", msg)
    try:
        p = ed.run_extractive_export(msg, export_dir=str(outdir))
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}: {str(exc)[:180]}")
        continue
    print(f"  status  : {p.status}")
    print(f"  alpha   : {p.alpha_source}")
    print(f"  file_id : {p.file_id}")
    if p.file_id:
        f = outdir / f"{p.file_id}.dwxmz"
        print(f"  produced: {f.exists()} ({f.stat().st_size if f.exists() else 0} bytes)")
    print("  message :", p.message.replace("\n", "\n            ")[:400])
    print()
print("outdir:", outdir)
