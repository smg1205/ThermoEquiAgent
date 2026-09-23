"""Test the generic ternary extractive parser and end-to-end export."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent")
sys.path.insert(0, str(ROOT))

from agent import generic_ternary_export as gte  # noqa: E402

print("=== 组分识别 ===")
for msg in [
    "导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
    "导出乙酸乙酯/乙酸正丙酯/DMSO三元VLE的DWSIM萃取精馏文件",
    "导出乙腈/甲苯/四氢呋喃 三元 VLE dwsim 文件",
    "水 甲苯 1-丁醇",                       # 只有两个/三个?
    "导出甲醇/水 精馏",                      # 只有两个 -> None
    "导出 乙醇/水/乙二醇 萃取精馏 UNIQUAC",
]:
    comps = gte.find_components(msg)
    print(f"  {msg[:44]:<46} -> {comps}")

print("\n=== 物性包解析 ===")
from agent.schema_helpers import resolve_property_package  # noqa: E402
for msg in ["导出水/甲苯/1-丁醇 萃取精馏 dwsim",
            "用 UNIQUAC 导出水/甲苯/1-丁醇 萃取精馏",
            "use uniquac for water/toluene/1-butanol extractive dwsim",
            "用 Wilson 物性包导出",
            "用 UNIFAC 导出",
            "导出（未指定物性包）"]:
    print(f"  {msg[:44]:<46} -> {resolve_property_package(msg)}")

print("\n=== 完整参数解析 ===")
for msg in ["导出 水/甲苯 + 1-丁醇 萃取精馏的 DWSIM 文件",
            "用 UNIQUAC 导出 水/甲苯 + 1-丁醇 萃取精馏 dwsim，纯度 0.99，回收率 0.95，剂料比 3"]:
    r = gte.parse_request(msg)
    if r is None:
        print(f"  {msg[:40]} -> None")
        continue
    print(f"  msg: {msg[:50]}")
    for f in ("light", "heavy", "entrainer", "property_package", "purity",
              "recovery", "entrainer_ratio", "feed_flow_mol_s", "pressure_kpa",
              "alpha_source"):
        print(f"      {f}: {getattr(r, f)}")
