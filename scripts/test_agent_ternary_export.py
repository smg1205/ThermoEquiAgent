"""Test whether the Agent's router can export a ternary VLE extractive DWSIM file.

Exercises the public routing entry points with several phrasings and reports which
handler fires, so the real capability (not just the presence of code) is measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from agent import extractive_distillation as ed  # noqa: E402

MESSAGES = [
    "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件",
    "ethyl acetate n-propyl acetate DMSO extractive distillation dwsim export",
    "导出 1-丁醇/水/甲苯 三元VLE 的 DWSIM 萃取精馏文件",
    "导出 水/甲苯 + 1-丁醇 萃取精馏 dwsim 文件",
    "导出乙腈/甲苯/四氢呋喃 三元 VLE dwsim 文件",
    "导出乙醇/水/乙二醇萃取精馏的DWSIM文件",
    "用ThermoFormer导出乙酸乙酯/乙酸正丙酯/DMSO的萃取精馏塔",
]

ROUTERS = [
    ("is_eac_npac_dmso_vle_export_request", ed.is_eac_npac_dmso_vle_export_request),
    ("is_extractive_request", getattr(ed, "is_extractive_request", None)),
    ("is_ipa_extractive_request", getattr(ed, "is_ipa_extractive_request", None)),
    ("is_lle_extraction_request", getattr(ed, "is_lle_extraction_request", None)),
    ("is_distillation_request", ed.is_distillation_request),
    ("wants_dwsim_file", ed.wants_dwsim_file),
]

print("=== router matrix (True = that handler claims the message) ===")
hdr = f"{'message':<58} " + " ".join(f"{n[:14]:>15}" for n, _ in ROUTERS)
print(hdr)
print("-" * len(hdr))
for msg in MESSAGES:
    cells = []
    for _, fn in ROUTERS:
        if fn is None:
            cells.append(f"{'n/a':>15}")
            continue
        try:
            cells.append(f"{str(bool(fn(msg))):>15}")
        except Exception as exc:  # noqa: BLE001
            cells.append(f"{type(exc).__name__[:14]:>15}")
    print(f"{msg[:57]:<58} " + " ".join(cells))

print("\n=== handler module surface ===")
for n in dir(ed):
    if n.startswith("run_") and "export" in n:
        print("  ", n)
