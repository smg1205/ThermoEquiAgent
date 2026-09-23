"""用设计原始意图（塔顶水纯度 / 水回收率）作为规格，测试哪种可收敛。

背景：当前文件用「回流比 + 塔釜流量」两条规格，DWSIM 报水物料平衡失配 0.99，
且该误差不随迭代次数下降（1000/5000/20000 次均为 0.99），提示规格对本身不适定，
而非收敛慢。

本脚本对同一体系（1-丁醇/水/甲苯，18 级，R=2.095，进料板 8，萃取剂板 2）
逐一试验不同的第二条规格，并用 CalculateFlowsheet4 读取真实错误
（CalculateFlowsheet2 会吞掉错误，不可用于判定）。

需要完整进程权限（pythonnet 需 OpenProcess）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = 18
REFLEX = 2.095
FEED_STAGE_0 = 7
ENT_STAGE_0 = 1


def build(automation, object_type, second_spec: str):
    """按 second_spec 指定第二条规格，返回 (flowsheet, dist, bot)。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in ("Water", "Toluene", "1-butanol"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()
    ent = flowsheet.AddObject(object_type.MaterialStream, 0, 80, "Ent").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()

    col.NumberOfStages = STAGES
    col.RefluxRatio = REFLEX
    col.MaxIterations = 5000

    feed.SetTemperature(327.09)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double]([0.5, 0.5, 0.0]))

    ent.SetTemperature(325.63)
    ent.SetPressure(101325.0)
    ent.SetMolarFlow(2.0)
    ent.SetOverallComposition(Array[Double]([0.0, 0.0, 1.0]))

    # 先 ConnectFeed 再 SetStreamFeedStage，两者都必需且顺序不可颠倒。
    col.ConnectFeed(feed, FEED_STAGE_0)
    col.SetStreamFeedStage(feed, FEED_STAGE_0)
    col.ConnectFeed(ent, ENT_STAGE_0)
    col.SetStreamFeedStage(ent, ENT_STAGE_0)

    condenser = col.Specs["C"]
    reboiler = col.Specs["R"]

    # 冷凝器统一用回流比。
    condenser.SType = ColumnSpec.SpecType.Stream_Ratio
    condenser.SpecValue = REFLEX
    condenser.SpecUnit = ""

    if second_spec == "bottoms_flow":
        reboiler.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
        reboiler.SpecValue = 2.526316
        reboiler.SpecUnit = "mol/s"
        reboiler.ComponentID = ""
    elif second_spec == "water_recovery":
        reboiler.SType = ColumnSpec.SpecType.Component_Recovery
        reboiler.SpecValue = 0.90
        reboiler.SpecUnit = ""
        reboiler.ComponentID = "Water"
    elif second_spec == "distillate_water_fraction":
        reboiler.SType = ColumnSpec.SpecType.Component_Fraction
        reboiler.SpecValue = 0.95
        reboiler.SpecUnit = ""
        reboiler.ComponentID = "Water"
    else:
        raise ValueError(second_spec)

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)
    return flowsheet, dist, bot


def describe(stream) -> str:
    phases = stream.Phases
    x = (
        [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]
        if phases.Count > 0
        else []
    )
    return f"T={float(stream.GetTemperature()):7.2f} F={float(stream.GetMolarFlow()):7.4f} x={x}"


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    for mode in ("bottoms_flow", "water_recovery", "distillate_water_fraction"):
        print(f"\n=== 第二条规格: {mode} ===")
        try:
            flowsheet, dist, bot = build(automation, object_type, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"  构建失败: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            continue

        try:
            errors = automation.CalculateFlowsheet4(flowsheet)
        except Exception as exc:  # noqa: BLE001
            print(f"  求解抛异常: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            continue

        if errors.Count == 0:
            print("  收敛 ✅")
        else:
            print(f"  未收敛 (错误数 {errors.Count})")
            for i in range(min(errors.Count, 2)):
                print(f"     {str(errors[i]).splitlines()[0][:120]}")
        print(f"  塔顶: {describe(dist)}")
        print(f"  塔釜: {describe(bot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
