"""逐步剥离法定位 1-丁醇/水/甲苯 严格塔不收敛的原因。

已知事实（均经实测）：
  * 误差恒为「Water 物料平衡 Relative Error ≈ 0.99」，不随迭代次数/容差变化
    （1000/5000/20000 次迭代、1e-4/1e-3/1e-2 容差都停在 0.99）
  * 接线完整（GetSolverInputData 通过），两股进料板位正确（第 2 / 第 8 板）
  * 设计经 FUG 独立复算自洽；三对二元 flash 均正常收敛
  * CalculateFlowsheet2 会吞错误，必须用 CalculateFlowsheet4
  * 同体系历史上出现过「1-butanol 物料平衡 Relative Error = -4.01」，
    当时结论是组分衡算不一致

策略：从最简可收敛构型出发，一次只改一个变量，找出收敛边界。
每步都用 CalculateFlowsheet4 判定，避免被静默失败误导。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def solve(automation, object_type, *, compounds, feeds, stages, reflux,
          bottoms_flow, feed_stages=None, pressure_kpa=101.325):
    """构建并求解一个严格塔，返回 (是否收敛, 摘要)。

    feeds:       [(流量, 温度, [组成]), ...]，组成长度与 compounds 一致
    feed_stages: 各股进料的 0-based 板位；缺省时第一股取中部、其余取第 1 板
    """
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

    col.NumberOfStages = stages
    col.RefluxRatio = reflux
    col.MaxIterations = 3000

    default_stage = max(1, stages // 2 - 1)
    for index, (flow, temp, comp) in enumerate(feeds):
        wrapper = flowsheet.AddObject(object_type.MaterialStream, 0, index * 60, f"F{index}")
        stream = wrapper.GetAsObject()
        stream.SetTemperature(temp)
        stream.SetPressure(pressure_kpa * 1000.0)
        stream.SetMolarFlow(flow)
        stream.SetOverallComposition(Array[Double](comp))
        stage = (
            feed_stages[index]
            if feed_stages
            else (1 if index else default_stage)
        )
        # 先 ConnectFeed 再 SetStreamFeedStage，顺序不可颠倒
        col.ConnectFeed(stream, stage)
        col.SetStreamFeedStage(stream, stage)

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

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return False, str(errors[0]).splitlines()[0][:105]

    def describe(stream) -> str:
        phases = stream.Phases
        x = (
            [round(float(c.MoleFraction), 3) for c in phases[0].Compounds.Values]
            if phases.Count > 0
            else []
        )
        return f"F={float(stream.GetMolarFlow()):7.4f} x={x}"

    return True, f"塔顶[{describe(dist)}] 塔釜[{describe(bot)}]"


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    WTB = ["Water", "Toluene", "1-butanol"]
    wt_feeds = [
        (1.0, 327.09, [0.5, 0.5, 0.0]),
        (2.0, 325.63, [0.0, 0.0, 1.0]),
    ]

    cases = [
        (
            "1. 二元 水/甲苯 无萃取剂, 10级 R=3",
            dict(compounds=["Water", "Toluene"],
                 feeds=[(1.0, 338.97, [0.5, 0.5])],
                 stages=10, reflux=3.0, bottoms_flow=0.5),
        ),
        (
            "2. 二元 水/甲苯 无萃取剂, 18级 R=2.095",
            dict(compounds=["Water", "Toluene"],
                 feeds=[(1.0, 338.97, [0.5, 0.5])],
                 stages=18, reflux=2.095, bottoms_flow=0.5),
        ),
        (
            "3. 三元+萃取剂 设计同规格 (18级 R=2.095 B=2.526)",
            dict(compounds=WTB, feeds=wt_feeds,
                 stages=18, reflux=2.095, bottoms_flow=2.526316,
                 feed_stages=[7, 1]),
        ),
        (
            "4. 同上但塔釜流量放宽 B=2.0",
            dict(compounds=WTB, feeds=wt_feeds,
                 stages=18, reflux=2.095, bottoms_flow=2.0,
                 feed_stages=[7, 1]),
        ),
        (
            "5. 同上但高回流 R=6",
            dict(compounds=WTB, feeds=wt_feeds,
                 stages=18, reflux=6.0, bottoms_flow=2.526316,
                 feed_stages=[7, 1]),
        ),
        (
            "6. 三元但两股进料都作普通进料(无萃取剂板位)",
            dict(compounds=WTB, feeds=wt_feeds,
                 stages=18, reflux=2.095, bottoms_flow=2.526316,
                 feed_stages=[8, 4]),
        ),
    ]

    for label, kwargs in cases:
        print(f"\n=== {label} ===")
        try:
            ok, detail = solve(automation, object_type, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  抛异常: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
            continue
        print(f"  {'收敛' if ok else '未收敛'}")
        print(f"  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
