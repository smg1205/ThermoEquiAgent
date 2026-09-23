"""追查：为何理想模型(Raoult)接近收敛，而活度系数模型完全不收敛。

最新证据：
  Raoult's Law  -> 物料平衡 -0.0191   （误差仅 1.9%，接近收敛）
  NRTL/UNIQUAC/UNIFAC/Wilson -> 迭代不收敛（完全失败）

这不是「物性包选错」，而是「活度系数计算在塔内迭代中出问题」。
方向：检查活度系数 γ 在塔内工作组成/温度范围内是否有异常
      （极大值、突变、NaN），这类异常会让逐板迭代发散。

做法：
  A. 用 Raoult's Law 放宽规格（高回流/少级数），看能否真正收敛
     —— 若能，则确认「塔结构没问题，问题在活度系数」
  B. 扫描三元体系在塔内工作区间的 γ 值，找异常
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]


def strict(automation, object_type, pp, *, stages, reflux, feed_stage, ent_stage,
           bottoms, seed=True, max_iter=5000):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in COMPOUNDS:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, pp)

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()

    col.SetNumberOfStages(stages)
    col.RefluxRatio = reflux
    col.MaxIterations = max_iter
    if seed:
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

    for flow, temp, comp, stage in (
        (1.0, 327.09, [0.5, 0.5, 0.0], feed_stage),
        (2.0, 325.63, [0.0, 0.0, 1.0], ent_stage),
    ):
        wrapper = flowsheet.AddObject(object_type.MaterialStream, 0, stage * 30, f"S{stage}")
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
        xd = [round(float(c.MoleFraction), 4) for c in dist.Phases[0].Compounds.Values]
        xb = [round(float(c.MoleFraction), 4) for c in bot.Phases[0].Compounds.Values]
        return f"收敛 OK  D={float(dist.GetMolarFlow()):.4f} x={xd} | B={float(bot.GetMolarFlow()):.4f} x={xb}"
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        return f"物料平衡 {first.split('Relative Error =')[-1].split('[')[0].strip()[:16]}"
    if "convergence error" in first:
        return "迭代不收敛"
    return first[:65]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. Raoult's Law 放宽规格（验证塔结构本身没问题）===")
    trials = [
        ("N=18 R=2.095 原设计", dict(stages=18, reflux=2.095, feed_stage=7, ent_stage=1, bottoms=2.526)),
        ("N=18 R=5", dict(stages=18, reflux=5.0, feed_stage=7, ent_stage=1, bottoms=2.526)),
        ("N=18 R=10", dict(stages=18, reflux=10.0, feed_stage=7, ent_stage=1, bottoms=2.526)),
        ("N=10 R=10", dict(stages=10, reflux=10.0, feed_stage=4, ent_stage=1, bottoms=2.526)),
        ("N=10 R=20", dict(stages=10, reflux=20.0, feed_stage=4, ent_stage=1, bottoms=2.526)),
        ("N=5  R=20", dict(stages=5, reflux=20.0, feed_stage=2, ent_stage=1, bottoms=2.526)),
        ("N=18 R=10 B=2.0", dict(stages=18, reflux=10.0, feed_stage=7, ent_stage=1, bottoms=2.0)),
    ]
    RAOULT = "Raoult's Law"
    for label, kwargs in trials:
        print(f"  {label:22} -> {strict(automation, object_type, RAOULT, **kwargs)}")

    print("\n=== B. 同一构型下 各物性包对比（N=10 R=10，宽容规格）===")
    for pp in (RAOULT, "NRTL", "UNIQUAC", "UNIFAC", "Wilson"):
        try:
            r = strict(automation, object_type, pp, stages=10, reflux=10.0,
                       feed_stage=4, ent_stage=1, bottoms=2.526)
        except Exception as exc:  # noqa: BLE001
            r = f"抛异常 {type(exc).__name__}"
        print(f"  {pp:16} -> {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
