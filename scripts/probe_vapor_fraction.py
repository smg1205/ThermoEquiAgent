"""读取闪蒸的相分率（vapor fraction）——「计算气相量失败」的直接对象。

线索：DWSIM 报错原文是
  PT Flash: Error calculating amount of the vapor phase in the mixture
即「计算混合物中气相的量」失败。此前我只读了相组成，从未读相分率。

若某股进料在塔内条件下相分率求解失败（例如全液相/全气相的边界判定），
逐板闪蒸就会中断，塔的迭代随之失败。

本脚本尝试多种途径读取相分率：
  - Phase.MoleFraction / MassFraction（相自身的分率属性）
  - Phase.Properties 中的相关键
  - 直接查询 DWSIM 的闪蒸接口
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dump_phase_members(automation, object_type):
    """先看 Phase 对象到底有哪些可用成员。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in ("Water", "Toluene"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    feed.SetTemperature(360.0)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double]([0.5, 0.5]))
    automation.CalculateFlowsheet4(flowsheet)

    phase = feed.Phases[2]      # Vapor 槽位
    print(f"=== Phase 对象成员（{phase.Name}）===")
    interesting = [
        m for m in dir(phase)
        if any(t in m.lower() for t in ("fraction", "amount", "mole", "mass", "phase", "ratio"))
        and not m.startswith("__")
    ]
    for name in sorted(interesting):
        try:
            value = getattr(phase, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:28} ! {type(exc).__name__}")
            continue
        if callable(value):
            print(f"  {name:28} <method>")
        elif hasattr(value, "Values"):
            try:
                print(f"  {name:28} = {[round(float(v), 4) for v in value.Values]}")
            except Exception:  # noqa: BLE001
                print(f"  {name:28} = <collection>")
        else:
            try:
                print(f"  {name:28} = {round(float(value), 6)}")
            except Exception:  # noqa: BLE001
                print(f"  {name:28} = {value!r}")

    print("\n=== 所有 Phase 槽位的分率 ===")
    for i in range(feed.Phases.Count):
        p = feed.Phases[i]
        try:
            pname = str(p.Name)
        except Exception:  # noqa: BLE001
            pname = f"slot{i}"
        row = []
        for attr in ("MoleFraction", "MassFraction", "PhaseMoleFraction"):
            try:
                v = getattr(p, attr)
                row.append(f"{attr}={round(float(v), 5)}")
            except Exception:  # noqa: BLE001
                row.append(f"{attr}=n/a")
        print(f"  [{i}] {pname:15} " + "  ".join(row))


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()
    dump_phase_members(automation, object_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
