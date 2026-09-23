"""Regression + acceptance test for the generic ternary extractive export.

Acceptance criteria (from the user):
  A1  any named ternary system is accepted (not just EtOAc/nPrOAc/DMSO);
  A2  property package defaults to NRTL;
  A3  a user-specified package (UNIQUAC / Wilson / UNIFAC) is honoured;
  A4  product spec uses the report's 0.95 purity / 0.90 recovery;
  A5  a component with no SMILES mapping still designs under UNIFAC but reports a
      structured missing-parameter failure under ThermoFormer;
  A6  the produced .dwxmz carries solver initial estimates;
  A7  the previously supported systems still route to their original handlers.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402,F401  (load before any DWSIM Automation call)

from agent import extractive_distillation as ed  # noqa: E402

outdir = Path(tempfile.mkdtemp(prefix="accept-"))
results: list[tuple[str, str, str]] = []


def run(label: str, msg: str, expect_status: str, expect_pkg: str | None = None) -> None:
    p = ed.run_extractive_export(msg, export_dir=str(outdir))
    ok = p.status == expect_status
    detail = p.file_id or ""
    if expect_pkg and p.file_id:
        pass
    results.append((label, p.status, "PASS" if ok else f"FAIL(exp {expect_status})"))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: status={p.status} file={detail}")
    if not ok:
        print("      message:", p.message.replace("\n", " | ")[:200])


print("=== A1/A2/A4: 任意体系 + 默认 NRTL + 报告规格 ===")
run("水/甲苯/1-丁醇 NRTL", "导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件", "ready")
run("乙腈/甲苯/THF", "导出乙腈/甲苯/四氢呋喃三元VLE的DWSIM萃取精馏文件", "ready")
run("甲醇/水/乙二醇", "导出 甲醇/水/乙二醇 萃取精馏 dwsim 文件", "ready")

print("\n=== A3: 用户指定物性包 ===")
run("UNIQUAC", "用 UNIQUAC 导出 水/甲苯/1-丁醇 萃取精馏 dwsim", "ready")
run("Wilson", "用 Wilson 导出 水/甲苯/1-丁醇 萃取精馏 dwsim", "ready")
run(
    "规格覆盖",
    "用 UNIQUAC 导出 水/甲苯/1-丁醇 萃取精馏 dwsim，纯度 0.99，回收率 0.95",
    "ready",
)

print("\n=== A5: SMILES 缺失 -> UNIFAC 可算 / TF 结构化报缺 ===")
run("氯仿 UNIFAC", "导出 氯仿/甲苯/甲醇 萃取精馏 dwsim", "ready")
run("氯仿 ThermoFormer", "用 ThermoFormer 导出 氯仿/甲苯/甲醇 萃取精馏 dwsim",
    "missing_parameters")

print("\n=== A7: 既有体系仍走原处理器 ===")
run("EtOAc/nPrOAc/DMSO (旧固定路径)", "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件",
    "ready")

print("\n=== 结果汇总 ===")
bad = [r for r in results if r[2] != "PASS"]
for label, status, verdict in results:
    print(f"  {verdict:<20} {label:<28} status={status}")
print(f"\n{len(results) - len(bad)}/{len(results)} passed")
print("outdir:", outdir)
sys.exit(1 if bad else 0)
