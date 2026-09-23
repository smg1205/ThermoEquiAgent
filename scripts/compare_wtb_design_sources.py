"""对照实验：判定残余不收敛是「规格口径问题」还是「求解器问题」。

背景：级数列表 bug 修复后（SetNumberOfStages），三元 1-丁醇/水/甲苯 塔仍报
  A convergence error was found while trying to solve the column.
  Possible reasons: unfeasible specs, initial estimates far from solution
  and/or very non-ideal (wide-boiling or azeotropic) mixtures being fed.
且对初值、求解器、回流比、进料板位、塔釜流量均不敏感。

思路：报告 §1.5 给了两套同体系设计（同一关键对 water/toluene + 1-butanol 萃取剂）：

  来源          N    R       N_min   R_min   alpha_avg
  UNIFAC       25   3.665   13.473  2.618   1.6623
  ThermoFormer 18   2.095    9.071  1.497   2.1273

UNIFAC 那套级数更多、回流比更高（更保守、更易收敛），TF 那套更激进。
两者是【同一体系、同一物性包、同一台塔结构】，仅设计量不同。

- 若 UNIFAC 组收敛而 TF 组不收敛 → 问题在【设计量/规格口径】，与求解器无关
- 若两组都不收敛             → 问题在【求解器/塔构造】，与具体设计无关

为避免混淆，两组都统一使用同一套温度口径（ThermoFormer 的 325.63/381.28 K）
与同一物性包（UNIQUAC），只让 N / R / 塔釜流量按各自设计取值。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]
SOLVER = "Modified Wang-Henke Bubble-Point (MBP) Solver"

# 报告 §1.5 表 1.5-2 的两套设计
DESIGNS = {
    "UNIFAC": dict(stages=25, reflux=3.665, feed_stage=11, ent_stage=2),
    "ThermoFormer": dict(stages=18, reflux=2.095, feed_stage=8, ent_stage=2),
}

WT_FEED = (1.0, 327.09, [0.5, 0.5, 0.0])       # 水/甲苯进料
ENT_FEED = (2.0, 325.63, [0.0, 0.0, 1.0])      # 1-丁醇萃取剂

# 两套设计的物料衡算（塔釜流量 = 总进料 - 塔顶）
TOTAL_IN = 3.0
D_DIST = 0.473684          # 塔顶流量（两套设计相同：都由纯度0.95/回收率0.90决定）
B_BOTTOMS = TOTAL_IN - D_DIST


def run(automation, object_type, *, stages, reflux, feed_stage, ent_stage,
        bottoms=B_BOTTOMS, seed=True, top_K=325.63, bottom_K=381.28,
        max_iter=5000):
    """按给定设计量建塔并求解。板位入参为「第 n 板」（1-based）。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in COMPOUNDS:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()

    col.SetNumberOfStages(stages)          # 必须用方法：属性 setter 不重建 Stages 列表
    col.RefluxRatio = reflux
    col.MaxIterations = max_iter
    col.SolvingMethodName = SOLVER

    stage_list_ok = col.Stages.Count == stages

    if seed:
        temps = [
            top_K + (bottom_K - top_K) * (i / max(stages - 1, 1))
            for i in range(stages)
        ]
        for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                     "UseVaporFlowEstimates"):
            try:
                setattr(col, flag, True)
            except Exception:  # noqa: BLE001
                pass
        liquid = reflux + 1.0
        vapor = liquid + 2.0
        col.SetInitialTemperatureEstimates(Array[Double](temps))
        col.SetInitialLiquidMolarFlowEstimates(Array[Double]([liquid] * stages))
        col.SetInitialVaporMolarFlowEstimates(Array[Double]([vapor] * stages))

    # 板位换算为 0-based
    for flow, temp, comp, stage in (
        (WT_FEED[0], WT_FEED[1], WT_FEED[2], feed_stage - 1),
        (ENT_FEED[0], ENT_FEED[1], ENT_FEED[2], ent_stage - 1),
    ):
        wrapper = flowsheet.AddObject(
            object_type.MaterialStream, 0, stage * 30, f"S{stage}"
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
        xb = [round(float(c.MoleFraction), 3) for c in bot.Phases[0].Compounds.Values]
        return (
            stage_list_ok,
            "收敛 OK",
            f"D={float(dist.GetMolarFlow()):.4f} x={xd} | "
            f"B={float(bot.GetMolarFlow()):.4f} x={xb}",
        )
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        reason = "物料平衡失配"
    elif "convergence error" in first:
        reason = "迭代不收敛"
    elif "索引" in first or "index" in first.lower():
        reason = "索引越界"
    else:
        reason = first[:60]
    return stage_list_ok, reason, first[:95]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("共同条件：UNIQUAC、101.325 kPa、水/甲苯 1.0 mol/s @327.09K、"
          "1-丁醇 2.0 mol/s @325.63K、塔釜流量 2.526316 mol/s")
    print(f"求解器：{SOLVER}\n")

    print("=== 对照实验：两套设计（同体系/同物性包/同塔结构，仅设计量不同）===")
    header = f"{'来源':14} {'N':>3} {'R':>7} {'进料板':>6} {'剂板':>4} {'Stages同步':>10}  结果"
    print(header)
    print("-" * len(header) + "-" * 20)
    results = {}
    for label, spec in DESIGNS.items():
        synced, reason, detail = run(automation, object_type, **spec)
        results[label] = (reason, detail)
        print(f"{label:14} {spec['stages']:>3} {spec['reflux']:>7.3f} "
              f"{spec['feed_stage']:>6} {spec['ent_stage']:>4} "
              f"{'是' if synced else '否':>10}  {reason}")
        print(f"{'':14} {detail}")

    print("\n=== 判定 ===")
    uf = results["UNIFAC"][0]
    tf = results["ThermoFormer"][0]
    if uf.startswith("收敛") and not tf.startswith("收敛"):
        print("  UNIFAC 组收敛、ThermoFormer 组不收敛")
        print("  => 问题在设计量/规格口径（TF 那套偏激进），与求解器无关")
    elif not uf.startswith("收敛") and not tf.startswith("收敛"):
        print("  两组都不收敛")
        print("  => 问题在塔构造/求解器层面，与具体设计量无关")
    else:
        print(f"  UNIFAC={uf}  ThermoFormer={tf}")

    print("\n=== 附加：把 TF 组逐步放宽到 UNIFAC 口径，定位敏感参数 ===")
    tf_spec = dict(DESIGNS["ThermoFormer"])
    variants = [
        ("TF 原样 (N=18,R=2.095)", {}),
        ("仅级数放宽到 25", dict(stages=25)),
        ("仅回流比放宽到 3.665", dict(reflux=3.665)),
        ("仅进料板改 11", dict(feed_stage=11)),
        ("三者都放宽(=UNIFAC 口径)", dict(stages=25, reflux=3.665, feed_stage=11)),
    ]
    for label, override in variants:
        spec = {**tf_spec, **override}
        synced, reason, detail = run(automation, object_type, **spec)
        print(f"  {label:30} -> {reason:12} {detail[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
