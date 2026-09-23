"""递进隔离：找出「能收敛」到「不收敛」的临界变化。

已确认：
  * 乙醇/水 10 级 R=3           -> 收敛（级数 bug 修复后）
  * 水/甲苯 18 级 R=2.095       -> 报 PT Flash 错
  * 三元 + 1-丁醇萃取剂          -> 迭代不收敛
  * UNIFAC 口径(N=25,R=3.665)   -> 同样不收敛（排除"规格太激进"）

故问题在塔构造/求解器层面，与设计量无关。需找出临界变化点。

隔离变量（一次只加一个）：
  1. 乙醇/水 二元（已知收敛）——基准
  2. 水/甲苯 二元               ——换成难分离的关键对
  3. 三元 水/甲苯/丁醇，丁醇仅微量  ——加第三组分但不成萃取剂
  4. 三元 + 全量萃取剂           ——完整构型

每一步都记录：Stages 是否同步、错误类型、是否收敛。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOLVER = "Modified Wang-Henke Bubble-Point (MBP) Solver"


def run(automation, object_type, *, compounds, feeds, stages, reflux,
        bottoms, seed=True, max_iter=5000):
    """feeds: [(流量, 温度, [组成], 板位0based), ...]"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()

    col.SetNumberOfStages(stages)
    col.RefluxRatio = reflux
    col.MaxIterations = max_iter
    col.SolvingMethodName = SOLVER
    synced = col.Stages.Count == stages

    if seed:
        temps = [300.0 + 90.0 * (i / max(stages - 1, 1)) for i in range(stages)]
        for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                     "UseVaporFlowEstimates"):
            try:
                setattr(col, flag, True)
            except Exception:  # noqa: BLE001
                pass
        col.SetInitialTemperatureEstimates(Array[Double](temps))
        col.SetInitialLiquidMolarFlowEstimates(Array[Double]([reflux + 1.0] * stages))
        col.SetInitialVaporMolarFlowEstimates(Array[Double]([reflux + 3.0] * stages))

    for index, (flow, temp, comp, stage) in enumerate(feeds):
        wrapper = flowsheet.AddObject(
            object_type.MaterialStream, 0, index * 40, f"S{index}"
        )
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
        xd = [round(float(c.MoleFraction), 3) for c in dist.Phases[0].Compounds.Values]
        return synced, "收敛 OK", (
            f"D={float(dist.GetMolarFlow()):.4f} x={xd}"
        )
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        reason = "物料平衡失配"
        m = first.split("Relative Error =")[-1].split("[")[0].strip()
        reason += f" {m[:18]}"
    elif "convergence error" in first:
        reason = "迭代不收敛"
    elif "PT Flash" in first:
        reason = "PT Flash 失败"
    elif "索引" in first or "index" in first.lower():
        reason = "索引越界"
    else:
        reason = first[:50]
    return synced, reason, first[:80]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    cases = [
        (
            "1. 乙醇/水 二元 10级 R=3 (基准)",
            dict(compounds=["Ethanol", "Water"],
                 feeds=[(1.0, 365.0, [0.5, 0.5], 4)],
                 stages=10, reflux=3.0, bottoms=0.5),
        ),
        (
            "2. 水/甲苯 二元 10级 R=3",
            dict(compounds=["Water", "Toluene"],
                 feeds=[(1.0, 338.97, [0.5, 0.5], 4)],
                 stages=10, reflux=3.0, bottoms=0.5),
        ),
        (
            "3. 水/甲苯 二元 10级 R=6 (高回流)",
            dict(compounds=["Water", "Toluene"],
                 feeds=[(1.0, 338.97, [0.5, 0.5], 4)],
                 stages=10, reflux=6.0, bottoms=0.5),
        ),
        (
            "4. 三元 丁醇仅微量(0.05) 两股进料",
            dict(compounds=["Water", "Toluene", "1-butanol"],
                 feeds=[(1.0, 327.09, [0.5, 0.5, 0.0], 7),
                        (0.05, 325.63, [0.0, 0.0, 1.0], 1)],
                 stages=18, reflux=2.095, bottoms=0.55),
        ),
        (
            "5. 三元 丁醇半量(1.0)",
            dict(compounds=["Water", "Toluene", "1-butanol"],
                 feeds=[(1.0, 327.09, [0.5, 0.5, 0.0], 7),
                        (1.0, 325.63, [0.0, 0.0, 1.0], 1)],
                 stages=18, reflux=2.095, bottoms=1.526),
        ),
        (
            "6. 三元 丁醇全量(2.0) 完整设计",
            dict(compounds=["Water", "Toluene", "1-butanol"],
                 feeds=[(1.0, 327.09, [0.5, 0.5, 0.0], 7),
                        (2.0, 325.63, [0.0, 0.0, 1.0], 1)],
                 stages=18, reflux=2.095, bottoms=2.526),
        ),
        (
            "7. 三元 全量但萃取剂当普通进料(不设剂板)",
            dict(compounds=["Water", "Toluene", "1-butanol"],
                 feeds=[(1.0, 327.09, [0.5, 0.5, 0.0], 8),
                        (2.0, 325.63, [0.0, 0.0, 1.0], 9)],
                 stages=18, reflux=2.095, bottoms=2.526),
        ),
    ]

    header = f"{'案例':42} {'同步':>4}  {'结果':16} 明细"
    print(header)
    print("-" * 120)
    for label, kwargs in cases:
        try:
            synced, reason, detail = run(automation, object_type, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:42} {'-':>4}  {'异常':16} {type(exc).__name__}")
            continue
        print(f"{label:42} {'是' if synced else '否':>4}  {reason:16} {detail[:55]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
