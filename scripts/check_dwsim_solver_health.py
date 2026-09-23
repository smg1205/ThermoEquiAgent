"""判断「塔不收敛」是本体系特有，还是我的求解调用方式通病。

剥离法已证明：连不含萃取剂的纯二元 水/甲苯 精馏都不收敛，故与 1-丁醇无关。
本脚本用最经典的教科书体系（乙醇/水）和极简塔（5 级），检验 DWSIM 能否
收敛任何塔。若连它也失败，则问题在调用方式，而非体系。

同时对比三种求解入口的行为差异：
  CalculateFlowsheet / CalculateFlowsheet2 / CalculateFlowsheet3 / CalculateFlowsheet4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build(automation, object_type, compounds, feed_comp, stages, reflux, bottoms_flow):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()
    feed_w = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = feed_w.GetAsObject()

    col.NumberOfStages = stages
    col.RefluxRatio = reflux
    col.MaxIterations = 2000

    feed.SetTemperature(365.0)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double](feed_comp))

    stage = max(1, stages // 2)
    col.ConnectFeed(feed, stage)
    col.SetStreamFeedStage(feed, stage)

    cond = col.Specs["C"]
    cond.SType = ColumnSpec.SpecType.Stream_Ratio
    cond.SpecValue = reflux
    cond.SpecUnit = ""

    reb = col.Specs["R"]
    reb.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    reb.SpecValue = bottoms_flow
    reb.SpecUnit = "mol/s"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)
    return flowsheet, dist, bot


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    # 教科书体系：乙醇/水，5 级，R=3 —— 若连它都不收敛，问题在调用方式
    trials = [
        ("乙醇/水 5级 R=3 B=0.5", ["Ethanol", "Water"], [0.5, 0.5], 5, 3.0, 0.5),
        ("乙醇/水 10级 R=5 B=0.5", ["Ethanol", "Water"], [0.5, 0.5], 10, 5.0, 0.5),
        ("水/甲苯 5级 R=3 B=0.5", ["Water", "Toluene"], [0.5, 0.5], 5, 3.0, 0.5),
    ]

    for label, compounds, comp, stages, reflux, bflow in trials:
        print(f"\n=== {label} ===")
        try:
            flowsheet, dist, bot = build(
                automation, object_type, compounds, comp, stages, reflux, bflow
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  构建异常: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
            continue

        for method in ("CalculateFlowsheet2", "CalculateFlowsheet3", "CalculateFlowsheet4"):
            try:
                if method == "CalculateFlowsheet3":
                    result = getattr(automation, method)(flowsheet, 600)
                    summary = "返回 None（无错误信息）"
                elif method == "CalculateFlowsheet4":
                    result = getattr(automation, method)(flowsheet)
                    summary = (
                        "错误数 0"
                        if result.Count == 0
                        else str(result[0]).splitlines()[0][:80]
                    )
                else:
                    result = getattr(automation, method)(flowsheet)
                    summary = f"返回 {result!r}"
                print(f"  {method:20} -> {summary}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {method:20} -> 抛异常 {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:70]}")

        print(f"  塔顶 F={float(dist.GetMolarFlow()):.4f} "
              f"x={[round(float(c.MoleFraction), 3) for c in dist.Phases[0].Compounds.Values]}")
        print(f"  塔釜 F={float(bot.GetMolarFlow()):.4f} "
              f"x={[round(float(c.MoleFraction), 3) for c in bot.Phases[0].Compounds.Values]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
