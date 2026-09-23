"""找出让 NumberOfStages 与 Stages 列表同步的正确调用。

已确认的根因：
  设 col.NumberOfStages = 18 后，col.Stages.Count 仍恒为 12（不同步）。
  DWSIM 内部按 NumberOfStages 索引 Stages 列表时越界，抛出
    ArgumentOutOfRangeException: 索引超出范围。必须为非负值并小于集合大小。
  这解释了为何乙醇/水、水/甲苯、三元体系全都以同一异常失败。

本脚本枚举所有可能与级数相关的成员，找出能正确同步的调用方式。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from thermo_engine.dwsim_export import _add_property_package, _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    flowsheet = automation.CreateFlowsheet()
    for name in ("Ethanol", "Water"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")
    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()

    print("=== 与级数相关的成员 ===")
    for name in sorted(dir(col)):
        if any(t in name.lower() for t in ("stage", "tray")):
            print(f"  {name}")

    print("\n=== AddStages / SetNumberOfStages 的真实签名 ===")
    col_type = col.GetType()
    for method in col_type.GetMethods():
        if method.Name in ("AddStages", "SetNumberOfStages", "set_NumberOfStages"):
            params = [
                f"{p.ParameterType.Name} {p.Name}" for p in method.GetParameters()
            ]
            print(f"  {method.Name}({', '.join(params)}) -> {method.ReturnType.Name}")

    print("\n=== 逐一尝试：哪种调用能让 Stages.Count 跟上 ===")

    def fresh():
        fs = automation.CreateFlowsheet()
        for name in ("Ethanol", "Water"):
            fs.AddCompound(name)
        _add_property_package(fs, "UNIQUAC")
        return (
            fs,
            fs.AddObject(object_type.DistillationColumn, 250, 0, "C").GetAsObject(),
        )

    target = 18

    _, c1 = fresh()
    c1.NumberOfStages = target
    print(f"  仅设 NumberOfStages={target}          -> Stages.Count={c1.Stages.Count}")

    _, c2 = fresh()
    try:
        c2.SetNumberOfStages(target)
        print(f"  SetNumberOfStages({target})             -> Stages.Count={c2.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  SetNumberOfStages({target})             -> 异常 {type(exc).__name__}")

    _, c3 = fresh()
    try:
        c3.AddStages(target)
        print(f"  AddStages({target})                     -> Stages.Count={c3.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  AddStages({target})                     -> 异常 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")

    _, c4 = fresh()
    try:
        c4.AddStages(target, False)
        print(f"  AddStages({target}, False)              -> Stages.Count={c4.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  AddStages({target}, False)              -> 异常 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")

    _, c5 = fresh()
    try:
        c5.AddStages(None, False)
        print(f"  AddStages(None, False)                -> Stages.Count={c5.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  AddStages(None, False)                -> 异常 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")

    _, c6 = fresh()
    try:
        while c6.Stages.Count < target:
            c6.AddStages(1)
        print(f"  循环 AddStages(1) 至 {target}            -> Stages.Count={c6.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  循环 AddStages(1)                     -> 异常 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")

    # 先设 NumberOfStages 再补级数
    _, c7 = fresh()
    try:
        c7.NumberOfStages = target
        while c7.Stages.Count < target:
            c7.AddStages(1)
        print(f"  设 Number 后循环 AddStages(1)          -> Stages.Count={c7.Stages.Count}")
    except Exception as exc:  # noqa: BLE001
        print(f"  设 Number 后循环 AddStages(1)          -> 异常 {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
