"""定位三元萃取塔 Water 物料平衡恒为 0.99999 的第二个问题。

已修复的根因（见 dswim_export.py）：NumberOfStages 属性不重建 Stages 列表，
必须用 SetNumberOfStages；否则越界。

修复后乙醇/水 10 级塔已能收敛，但 1-丁醇/水/甲苯 三元塔仍报
  Failed to fulfill mass balance for Water: Relative Error = 0.999986566328396
且该数值与修复前完全相同，说明这是独立问题。

关键观察：Relative Error ≈ 1.0 意味着「塔顶要求的水量 ≈ 实际没有水」，
即水在两个产品之间没有正确分配。怀疑点：
  A. 第二条规格（塔釜流量 Product_Molar_Flow_Rate）与回流比规格不匹配
  B. 萃取剂进料板位（第 2 板）过低，导致萃取剂直接短路到塔顶
  C. 进料温度不在泡点，造成闪蒸后相态异常

本脚本逐一施加扰动，观察误差是否变化，以定位敏感参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPOUNDS = ["Water", "Toluene", "1-butanol"]
WT_FEED = (1.0, 327.09, [0.5, 0.5, 0.0])
ENT_FEED = (2.0, 325.63, [0.0, 0.0, 1.0])


def run(automation, object_type, *, stages=18, reflux=2.095, bottoms=2.526316,
        feed_stage=7, ent_stage=1, spec2="bottoms_flow", wt_feed=WT_FEED,
        ent_feed=ENT_FEED, label=""):
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

    # 必须用 SetNumberOfStages 才能重建 Stages 列表
    col.SetNumberOfStages(stages)
    col.RefluxRatio = reflux
    col.MaxIterations = 3000

    for flow, temp, comp, stage in (
        (wt_feed[0], wt_feed[1], wt_feed[2], feed_stage),
        (ent_feed[0], ent_feed[1], ent_feed[2], ent_stage),
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
    if spec2 == "bottoms_flow":
        reb.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
        reb.SpecValue = bottoms
        reb.SpecUnit = "mol/s"
        reb.ComponentID = ""
    elif spec2 == "water_recovery":
        reb.SType = ColumnSpec.SpecType.Component_Recovery
        reb.SpecValue = 0.90
        reb.SpecUnit = ""
        reb.ComponentID = "Water"
    elif spec2 == "dist_water_frac":
        reb.SType = ColumnSpec.SpecType.Component_Fraction
        reb.SpecValue = 0.95
        reb.SpecUnit = ""
        reb.ComponentID = "Water"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count == 0:
        xd = [round(float(c.MoleFraction), 3) for c in dist.Phases[0].Compounds.Values]
        xb = [round(float(c.MoleFraction), 3) for c in bot.Phases[0].Compounds.Values]
        return f"收敛 ✅ D={float(dist.GetMolarFlow()):.4f} x={xd} | B={float(bot.GetMolarFlow()):.4f} x={xb}"
    return str(errors[0]).splitlines()[0][:88]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    tests = [
        ("基线（现行规格）", {}),
        ("规格2 = 水回收率 0.90", dict(spec2="water_recovery")),
        ("规格2 = 塔顶水纯度 0.95", dict(spec2="dist_water_frac")),
        ("回流比 6.0", dict(reflux=6.0)),
        ("回流比 10.0", dict(reflux=10.0)),
        ("萃取剂板改第 4 板", dict(ent_stage=3)),
        ("萃取剂板改第 1 板(顶)", dict(ent_stage=0)),
        ("进料板改第 10 板", dict(feed_stage=9)),
        ("塔釜流量改 2.0", dict(bottoms=2.0)),
        ("塔釜流量改 2.8", dict(bottoms=2.8)),
        ("进料温度改 338.97(泡点)", dict(wt_feed=(1.0, 338.97, [0.5, 0.5, 0.0]))),
    ]

    for label, kwargs in tests:
        try:
            result = run(automation, object_type, **kwargs)
        except Exception as exc:  # noqa: BLE001
            result = f"抛异常 {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
        print(f"  {label:26} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
