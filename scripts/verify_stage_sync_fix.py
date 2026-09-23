"""验证根因：用 SetNumberOfStages（而非 NumberOfStages=）后，塔能否收敛。

已确认：
  仅设 col.NumberOfStages = 18   -> Stages.Count 仍为 12（列表未重建）
  col.SetNumberOfStages(18)      -> Stages.Count = 18（正确）

导出代码同时调用了 SetNumberOfStages 与 set_NumberOfStages（属性 setter），
后者会把列表重新搞乱。本脚本验证「只用 SetNumberOfStages」是否解决问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def trial(automation, object_type, *, compounds, comp, stages, reflux,
          bottoms_flow, feed_temp, use_setter_method: bool):
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
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()

    # 关键差别：用方法而非属性 setter
    if use_setter_method:
        col.SetNumberOfStages(stages)
    else:
        col.NumberOfStages = stages
        col.set_NumberOfStages(stages)  # 模拟现有导出代码的后续写法

    stages_synced = col.Stages.Count
    col.RefluxRatio = reflux
    col.MaxIterations = 2000

    feed.SetTemperature(feed_temp)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double](comp))

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

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return stages_synced, False, str(errors[0]).splitlines()[0][:72]
    return (
        stages_synced,
        True,
        f"塔顶 F={float(dist.GetMolarFlow()):.4f} "
        f"x={[round(float(c.MoleFraction), 3) for c in dist.Phases[0].Compounds.Values]} | "
        f"塔釜 F={float(bot.GetMolarFlow()):.4f} "
        f"x={[round(float(c.MoleFraction), 3) for c in bot.Phases[0].Compounds.Values]}",
    )


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    cases = [
        ("乙醇/水 10级 R=3", dict(compounds=["Ethanol", "Water"], comp=[0.5, 0.5],
                              stages=10, reflux=3.0, bottoms_flow=0.5, feed_temp=365.0)),
        ("水/甲苯 18级 R=2.095", dict(compounds=["Water", "Toluene"], comp=[0.5, 0.5],
                                 stages=18, reflux=2.095, bottoms_flow=0.5, feed_temp=338.97)),
        ("三元 1-丁醇/水/甲苯 18级 R=2.095", dict(
            compounds=["Water", "Toluene", "1-butanol"], comp=[0.5, 0.5, 0.0],
            stages=18, reflux=2.095, bottoms_flow=0.5, feed_temp=327.09)),
    ]

    for label, kwargs in cases:
        print(f"\n=== {label} ===")
        for use_method in (False, True):
            mode = "SetNumberOfStages（推荐）" if use_method else "NumberOfStages= + set_（现行代码）"
            try:
                count, ok, detail = trial(
                    automation, object_type, use_setter_method=use_method, **kwargs
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {mode:28} 异常 {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
                continue
            print(f"  {mode:28} Stages.Count={count:2} {'收敛' if ok else '未收敛'}")
            print(f"      {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
