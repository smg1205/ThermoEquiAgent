"""回答：为何理想模型与活度系数模型的误差相差如此之大。

现象对比（同一设计、同一塔结构、同一规格）：
  Raoult's Law            误差 -0.019 ~ 0.54，且【随规格变化】
  所有活度系数模型          误差 0.9999，且【与规格完全无关】

诊断含义：
  「随规格变化」= 求解器在迭代、在响应规格变化
  「恒定 0.9999」= 求解器没在正常工作，一进门就失败

要找的是：活度系数模型下 γ 或 K 值是否有异常（极大/极小/NaN/突变），
这类异常会让逐板迭代第一步就崩掉。

做法：构造一个接近单级的简单情形，直接读取 γ 与 K 值。
DWSIM 的 MaterialStream 有 Phase 对象，可尝试读取各相的属性。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]


def probe_pp(automation, object_type, pp, z, T):
    """对给定物性包做一次 flash，尽可能读出 γ / K 值。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in COMPOUNDS:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, pp)

    from System import Array, Double

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    feed.SetTemperature(T)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double](z))

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return {"error": str(errors[0]).splitlines()[0][:70]}

    out: dict = {}
    phases = feed.Phases
    for i in range(phases.Count):
        phase = phases[i]
        try:
            name = str(phase.Name)
        except Exception:  # noqa: BLE001
            continue
        if name not in ("Vapor", "Liquid1", "OverallLiquid", "Mixture"):
            continue
        info: dict = {}
        for attr in ("MoleFraction", "MassFraction", "Fugacity", "FugacityCoefficient",
                     "ActivityCoefficient", "Kvalue", "KValue", "Compounds"):
            try:
                value = getattr(phase, attr)
            except Exception:  # noqa: BLE001
                continue
            if attr == "Compounds":
                try:
                    info["x"] = [
                        None if c.MoleFraction is None else round(float(c.MoleFraction), 4)
                        for c in value.Values
                    ]
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                if hasattr(value, "Values"):
                    info[attr] = [
                        None if v is None else round(float(v), 5) for v in value.Values
                    ]
                else:
                    info[attr] = round(float(value), 5) if value is not None else None
            except Exception:  # noqa: BLE001
                pass
        out[name] = info
    return out


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    z = [0.2, 0.3, 0.5]
    T = 365.0

    print(f"条件：水/甲苯/丁醇 = {z}，T={T} K，P=101.325 kPa")
    print("（这是塔中部附近的典型条件）\n")

    for pp in ("Raoult's Law", "NRTL", "UNIQUAC", "UNIFAC", "Wilson"):
        try:
            result = probe_pp(automation, object_type, pp, z, T)
        except Exception as exc:  # noqa: BLE001
            print(f"=== {pp} === 抛异常 {type(exc).__name__}")
            continue
        print(f"=== {pp} ===")
        if "error" in result:
            print(f"  失败: {result['error']}")
            continue
        for phase_name, info in result.items():
            print(f"  {phase_name}: {info}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
