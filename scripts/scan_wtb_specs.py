"""扫描塔釜流量规格，验证「规格是否可行」——最后一个未排除的疑因。

反常证据：Raoult's Law 在【原设计点】误差最小(-0.019)，
但放宽规格（提高回流比）反而变差(0.15→0.37→0.62→0.82)。
正常情况提高回流比应更易收敛，故问题不在求解难度。

另一线索：所有活度系数模型的误差都固定在 ~0.9999，
与物性包无关 —— 这种「固定值」通常指向结构性/规格问题。

设计物料衡算：
  进料 水0.5/甲苯0.5 = 1.0 mol/s，萃取剂 丁醇 = 2.0 mol/s，总进 3.0
  塔顶 D = 0.4737 (水0.95/甲苯0.05)
  塔釜 B = 2.5263 (水0.0198/甲苯0.1885/丁醇0.7917)

本脚本扫描塔釜流量规格，并对照另一种规格（塔顶流量/回收率），
检查哪个规格组合能让塔收敛。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]


def run(automation, object_type, *, pp="UNIQUAC", stages=18, reflux=2.095,
        feed_stage=7, ent_stage=1, spec2="bottoms", spec2_value=2.526316,
        spec2_component="", seed=True, max_iter=5000):
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
    mapping = {
        "bottoms": ColumnSpec.SpecType.Product_Molar_Flow_Rate,
        "distillate": ColumnSpec.SpecType.Product_Molar_Flow_Rate,
        "water_recovery": ColumnSpec.SpecType.Component_Recovery,
        "water_frac": ColumnSpec.SpecType.Component_Fraction,
    }
    reb.SType = mapping[spec2]
    reb.SpecValue = spec2_value
    reb.SpecUnit = "mol/s" if spec2 in ("bottoms", "distillate") else ""
    reb.ComponentID = spec2_component

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count == 0:
        xd = [round(float(c.MoleFraction), 4) for c in dist.Phases[0].Compounds.Values]
        xb = [round(float(c.MoleFraction), 4) for c in bot.Phases[0].Compounds.Values]
        return True, (f"收敛 OK D={float(dist.GetMolarFlow()):.4f} x={xd} | "
                      f"B={float(bot.GetMolarFlow()):.4f} x={xb}")
    first = str(errors[0]).splitlines()[0]
    if "mass balance" in first:
        return False, f"物料平衡 {first.split('Relative Error =')[-1].split('[')[0].strip()[:16]}"
    if "convergence error" in first:
        return False, "迭代不收敛"
    return False, first[:60]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. 扫描塔釜流量规格（UNIQUAC, N=18 R=2.095）===")
    print("   （理论上 B = 3.0 - D，D 由纯度0.95/回收率0.90 决定 = 0.4737 => B=2.5263）")
    for b in (0.5, 1.0, 1.5, 2.0, 2.5263, 2.8, 2.9, 2.95):
        ok, detail = run(automation, object_type, spec2="bottoms", spec2_value=b)
        print(f"  B={b:<8} {'收敛' if ok else '未收敛':6} {detail}")

    print("\n=== B. 扫描塔顶流量规格（改用塔顶流量作第二条规格）===")
    for d in (0.05, 0.2, 0.4737, 1.0, 1.5, 2.0):
        ok, detail = run(automation, object_type, spec2="distillate", spec2_value=d)
        print(f"  D={d:<8} {'收敛' if ok else '未收敛':6} {detail}")

    print("\n=== C. 用水回收率/纯度作规格 ===")
    for label, kwargs in (
        ("水回收率 0.90", dict(spec2="water_recovery", spec2_value=0.90, spec2_component="Water")),
        ("水回收率 0.50", dict(spec2="water_recovery", spec2_value=0.50, spec2_component="Water")),
        ("水回收率 0.99", dict(spec2="water_recovery", spec2_value=0.99, spec2_component="Water")),
        ("塔顶水分数 0.95", dict(spec2="water_frac", spec2_value=0.95, spec2_component="Water")),
        ("塔顶水分数 0.50", dict(spec2="water_frac", spec2_value=0.50, spec2_component="Water")),
    ):
        ok, detail = run(automation, object_type, **kwargs)
        print(f"  {label:16} {'收敛' if ok else '未收敛':6} {detail}")

    print("\n=== D. Raoult's Law 下扫描塔釜流量（误差最小的物性包）===")
    for b in (0.4, 0.45, 0.4737, 0.5, 0.6, 1.0, 2.5263):
        ok, detail = run(automation, object_type, pp="Raoult's Law",
                         spec2="bottoms", spec2_value=b)
        print(f"  B={b:<8} {'收敛' if ok else '未收敛':6} {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
