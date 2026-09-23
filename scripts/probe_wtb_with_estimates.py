"""给严格塔注入初值 + 逐一切换求解器，定位三元塔不收敛的原因。

DWSIM 报的完整错误（Wang-Henke 求解器，BubblePoint.vb:1728）：
  A convergence error was found while trying to solve the column.
  Possible reasons are: unfeasible specs, initial estimates far from solution
  and/or very non-ideal (wide-boiling or azeotropic) mixtures being fed.

三个疑因中，「初值远离解」可直接用注入初值排除。设计给出的温度区间为
塔顶 282.47 K -> 塔釜 378.73 K（UNIFAC 泡点口径）或 325.63 -> 381.28 K
（ThermoFormer 口径）。本脚本对两种温度分布都试，并对比可用求解器。

注意：SetInitial*Estimates 的数组长度必须恰好等于级数。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]
SOLVERS = [
    "Wang-Henke Bubble-Point (BP) Solver",
    "Modified Wang-Henke Bubble-Point (MBP) Solver",
]


def run(automation, object_type, *, solver, stages=18, reflux=2.095,
        bottoms=2.526316, feed_stage=7, ent_stage=1,
        top_K=325.63, bottom_K=381.28, seed_estimates=True, max_iter=3000):
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

    col.SetNumberOfStages(stages)          # 必须用方法，属性 setter 不重建列表
    col.RefluxRatio = reflux
    col.MaxIterations = max_iter
    col.SolvingMethodName = solver

    if seed_estimates:
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
        liquid = 1.0 * reflux + 1.0
        vapor = liquid + 2.0
        try:
            col.SetInitialTemperatureEstimates(Array[Double](temps))
            col.SetInitialLiquidMolarFlowEstimates(Array[Double]([liquid] * stages))
            col.SetInitialVaporMolarFlowEstimates(Array[Double]([vapor] * stages))
        except Exception as exc:  # noqa: BLE001
            return f"注入初值失败 {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"

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
        return (
            f"收敛 OK  D={float(dist.GetMolarFlow()):.4f} "
            f"x={[round(float(c.MoleFraction), 3) for c in dist.Phases[0].Compounds.Values]}"
        )
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        return f"物料平衡: {first[:70]}"
    if "convergence error" in first:
        return "收敛失败（未达迭代收敛）"
    return first[:80]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. 不注入初值，对比两个求解器 ===")
    for solver in SOLVERS:
        print(f"  {solver:44} -> "
              f"{run(automation, object_type, solver=solver, seed_estimates=False)}")

    print("\n=== B. 注入初值（ThermoFormer 温度口径 325.63->381.28 K）===")
    for solver in SOLVERS:
        print(f"  {solver:44} -> "
              f"{run(automation, object_type, solver=solver)}")

    print("\n=== C. 注入初值（UNIFAC 温度口径 282.47->378.73 K）===")
    for solver in SOLVERS:
        print(f"  {solver:44} -> "
              f"{run(automation, object_type, solver=solver, top_K=282.47, bottom_K=378.73)}")

    print("\n=== D. 提高回流比 + 初值（放宽规格）===")
    for reflux in (4.0, 6.0, 10.0):
        print(f"  R={reflux:5}  -> "
              f"{run(automation, object_type, solver=SOLVERS[1], reflux=reflux)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
