"""追问：为何理想模型与活度系数模型误差相差巨大 —— 定位到「液相组成被改写」。

最新发现（水/甲苯/丁醇 = 0.2/0.3/0.5，T=365 K）：
  Raoult's Law   Liquid1 = [0.2,   0.3,   0.5  ]  等于进料 -> 单液相
  NRTL           Liquid1 = [0.2,   0.3,   0.5  ]  等于进料 -> 单液相
  Wilson         Liquid1 = [0.2,   0.3,   0.5  ]  等于进料 -> 单液相
  UNIQUAC        Liquid1 = [0.1144,0.2827,0.6029]  不等于进料 -> 液液分层
  UNIFAC         Liquid1 = [0.0981,0.2808,0.6211]  不等于进料 -> 液液分层

即：UNIQUAC / UNIFAC 在该体系预测液液分层，其余模型不预测。

但注意：NRTL 无液液分层、塔却也不收敛 => 液液分层不是唯一原因。
需要区分「哪一类模型不收敛」与「哪一类分层」。

本脚本做关键对照：
  A. 同一条件下列出各模型是否分层 × 塔是否收敛，做成二维表
  B. 用报告的 UNIFAC 口径（N=25 R=3.665）原样测试
  C. 检查是否与「丁醇」这个第三组分有关（去掉丁醇，只测水/甲苯）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def liquid_split_count(automation, object_type, pp, compounds, z, T):
    """返回真实液相个数（LiquidN 中组成非空且非 NaN 的个数）。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
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
        return None, str(errors[0]).splitlines()[0][:50]

    count = 0
    phases = feed.Phases
    for i in range(phases.Count):
        phase = phases[i]
        try:
            name = str(phase.Name)
        except Exception:  # noqa: BLE001
            continue
        if not name.startswith("Liquid") or name == "OverallLiquid":
            continue
        try:
            values = [getattr(c, "MoleFraction") for c in phase.Compounds.Values]
        except Exception:  # noqa: BLE001
            continue
        if not values or any(v is None for v in values):
            continue
        floats = [float(v) for v in values]
        if any(v != v for v in floats):
            continue
        if all(abs(v) < 1e-12 for v in floats):
            continue
        count += 1
    return count, "OK"


def column_ok(automation, object_type, pp, *, compounds, stages, reflux,
              feed_stage, ent_stage, bottoms, wt_feed, ent_feed):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, pp)

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()

    col.SetNumberOfStages(stages)
    col.RefluxRatio = reflux
    col.MaxIterations = 5000
    temps = [325.63 + (381.28 - 325.63) * (i / max(stages - 1, 1)) for i in range(stages)]
    for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                 "UseVaporFlowEstimates"):
        try:
            setattr(col, flag, True)
        except Exception:  # noqa: BLE001
            pass
    col.SetInitialTemperatureEstimates(Array[Double](temps))
    col.SetInitialLiquidMolarFlowEstimates(Array[Double]([reflux + 1.0] * stages))
    col.SetInitialVaporMolarFlowEstimates(Array[Double]([reflux + 3.0] * stages))

    specs = []
    if ent_feed is not None:
        specs.append((ent_feed, ent_stage))
    specs.append((wt_feed, feed_stage))

    for index, ((flow, temp, comp), stage) in enumerate(specs):
        wrapper = flowsheet.AddObject(object_type.MaterialStream, 0, index * 40, f"S{index}")
        stream = wrapper.GetAsObject()
        stream.SetTemperature(temp)
        stream.SetPressure(101325.0)
        stream.SetMolarFlow(flow)
        stream.SetOverallComposition(Array[Double](comp))
        col.ConnectFeed(stream, stage)
        col.SetStreamFeedStage(stream, stage)

    cond = col.Specs["C"]
    cond.SType = ColumnSpec.SpecType.Stream_Ratio
    cond.SpecValue = reflux
    cond.SpecUnit = ""
    reb = col.Specs["R"]
    reb.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    reb.SpecValue = bottoms
    reb.SpecUnit = "mol/s"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count == 0:
        return "收敛"
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        return "物料平衡 " + first.split("Relative Error =")[-1].split("[")[0].strip()[:10]
    if "convergence error" in first:
        return "不收敛"
    return first[:40]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    PP = ("Raoult's Law", "NRTL", "UNIQUAC", "UNIFAC", "Wilson")

    print("=== A. 二维对照：是否液液分层 × 三元塔是否收敛 ===")
    print(f"{'物性包':16} {'三元分层':>8} {'三元塔':>14}   水甲苯分层   水甲苯塔")
    print("-" * 72)
    for pp in PP:
        n3, _ = liquid_split_count(automation, object_type, pp,
                                   ["Water", "Toluene", "1-butanol"], [0.2, 0.3, 0.5], 365.0)
        col3 = column_ok(
            automation, object_type, pp,
            compounds=["Water", "Toluene", "1-butanol"],
            stages=18, reflux=2.095, feed_stage=7, ent_stage=1, bottoms=2.526316,
            wt_feed=(1.0, 327.09, [0.5, 0.5, 0.0]),
            ent_feed=(2.0, 325.63, [0.0, 0.0, 1.0]),
        )
        n2, _ = liquid_split_count(automation, object_type, pp,
                                   ["Water", "Toluene"], [0.5, 0.5], 360.0)
        col2 = column_ok(
            automation, object_type, pp,
            compounds=["Water", "Toluene"],
            stages=18, reflux=2.095, feed_stage=7, ent_stage=None, bottoms=0.5,
            wt_feed=(1.0, 338.97, [0.5, 0.5]),
            ent_feed=None,
        )
        print(f"{pp:16} {str(n3):>8} {col3:>14}   {str(n2):>10} {col2:>14}")

    print("\n=== B. 报告 UNIFAC 口径原样测试（N=25 R=3.665 进料板11）===")
    for pp in PP:
        r = column_ok(
            automation, object_type, pp,
            compounds=["Water", "Toluene", "1-butanol"],
            stages=25, reflux=3.665, feed_stage=10, ent_stage=1, bottoms=2.526316,
            wt_feed=(1.0, 327.09, [0.5, 0.5, 0.0]),
            ent_feed=(2.0, 325.63, [0.0, 0.0, 1.0]),
        )
        print(f"  {pp:16} -> {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
