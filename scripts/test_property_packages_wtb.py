"""验证：改用 NRTL（不预测液液分层）后，三元萃取塔能否收敛。

结论链：
  1. 级数列表 bug（SetNumberOfStages）—— 已修复
  2. 水/甲苯在 UNIQUAC 下预测液液分层（2 个液相），NRTL 不预测（1 个液相）
  3. 严格精馏塔按气液两相逐板计算，遇液液分层即崩
     => 连 5 级 R=5 的宽松构型都不收敛

本脚本对同一设计、同一塔结构，只更换物性包，逐一测试。

物性包候选（DWSIM 常用）：
  NRTL、UNIQUAC、Raoult's Law、van Laar、Wilson、UNIFAC
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]
STAGES, REFLUX = 18, 2.095
FEED_STAGE, ENT_STAGE = 7, 1        # 0-based
BOTTOMS = 2.526316


def run(automation, object_type, pp, *, seed=True, solver=None):
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

    col.SetNumberOfStages(STAGES)          # 必须用方法
    col.RefluxRatio = REFLUX
    col.MaxIterations = 5000
    if solver:
        col.SolvingMethodName = solver

    if seed:
        temps = [325.63 + (381.28 - 325.63) * (i / (STAGES - 1)) for i in range(STAGES)]
        for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                     "UseVaporFlowEstimates"):
            try:
                setattr(col, flag, True)
            except Exception:  # noqa: BLE001
                pass
        col.SetInitialTemperatureEstimates(Array[Double](temps))
        col.SetInitialLiquidMolarFlowEstimates(Array[Double]([REFLUX + 1.0] * STAGES))
        col.SetInitialVaporMolarFlowEstimates(Array[Double]([REFLUX + 3.0] * STAGES))

    for flow, temp, comp, stage in (
        (1.0, 327.09, [0.5, 0.5, 0.0], FEED_STAGE),
        (2.0, 325.63, [0.0, 0.0, 1.0], ENT_STAGE),
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
    cond.SpecValue = REFLUX
    cond.SpecUnit = ""
    reb = col.Specs["R"]
    reb.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    reb.SpecValue = BOTTOMS
    reb.SpecUnit = "mol/s"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count == 0:
        xd = [round(float(c.MoleFraction), 4) for c in dist.Phases[0].Compounds.Values]
        xb = [round(float(c.MoleFraction), 4) for c in bot.Phases[0].Compounds.Values]
        return True, (
            f"D={float(dist.GetMolarFlow()):.4f} x={xd}  "
            f"B={float(bot.GetMolarFlow()):.4f} x={xb}"
        )
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        val = first.split("Relative Error =")[-1].split("[")[0].strip()
        return False, f"物料平衡 {val[:16]}"
    if "convergence error" in first:
        return False, "迭代不收敛"
    return False, first[:70]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print(f"设计：N={STAGES} R={REFLUX} 进料板{FEED_STAGE + 1} 剂板{ENT_STAGE + 1} "
          f"塔釜流量={BOTTOMS}")
    print("只更换物性包，其余完全相同\n")

    packages = [
        "NRTL",
        "UNIQUAC",
        "Raoult's Law",
        "UNIFAC",
        "van Laar",
        "Wilson",
        "Modified UNIFAC (Dortmund)",
    ]

    results = {}
    for pp in packages:
        try:
            ok, detail = run(automation, object_type, pp)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pp:28} 抛异常 {type(exc).__name__}: {str(exc).splitlines()[0][:50]}")
            continue
        results[pp] = ok
        print(f"  {pp:28} {'收敛 ✅' if ok else '未收敛'}  {detail}")

    print("\n=== 判定 ===")
    winners = [pp for pp, ok in results.items() if ok]
    if winners:
        print(f"  可收敛的物性包: {winners}")
        print("  => 不收敛的原因确是物性包（UNIQUAC 预测水/甲苯液液分层）")
    else:
        print("  所有物性包均不收敛 => 原因不限于物性包，需继续排查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
