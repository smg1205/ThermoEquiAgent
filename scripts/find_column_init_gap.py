"""定位「索引超出范围」的真实来源：塔创建后缺了哪一步初始化。

事实：连乙醇/水 5 级 R=3 的教科书塔都报
  Col: 索引超出范围。必须为非负值并小于集合大小。
说明与体系、萃取剂均无关，是构造流程的系统性缺陷。

该异常此前定位到 Column.GetSolverInputData -> List.get_Item(index)，
即某个内部 List 为空/长度不足时被索引。

本脚本对比「新建塔」与「配置后塔」的全部可读写属性，找出仍为
默认/空值的关键项，并逐一尝试补齐后再求解。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from thermo_engine.dwsim_export import _add_property_package, _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    flowsheet = automation.CreateFlowsheet()
    for name in ("Ethanol", "Water"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()

    print("=== 新建塔的关键属性默认值 ===")
    for attr in (
        "NumberOfStages", "RefluxRatio", "DistillateFlowRate",
        "CondenserPressure", "ReboilerPressure", "ColumnPressureDrop",
        "TopPressure", "BottomPressure",
        "InternalLoopTolerance", "ExternalLoopTolerance", "MaxIterations",
        "SolvingMethodName", "Stages", "Specs", "MaterialStreams",
    ):
        try:
            value = getattr(col, attr)
            text = f"count={value.Count}" if hasattr(value, "Count") else repr(value)
            print(f"  {attr:26} = {text}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {attr:26} ! {type(exc).__name__}")

    print("\n=== 逐个补属性，看哪一步消除「索引超出范围」 ===")

    def attempt(label: str, apply) -> str:
        # 每次用全新塔，避免污染
        fs = automation.CreateFlowsheet()
        for name in ("Ethanol", "Water"):
            fs.AddCompound(name)
        _add_property_package(fs, "UNIQUAC")
        c = fs.AddObject(object_type.DistillationColumn, 250, 0, "C").GetAsObject()
        d = fs.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
        b = fs.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()
        f = fs.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()

        c.NumberOfStages = 5
        c.RefluxRatio = 3.0
        f.SetTemperature(365.0)
        f.SetPressure(101325.0)
        f.SetMolarFlow(1.0)
        f.SetOverallComposition(Array[Double]([0.5, 0.5]))
        c.ConnectFeed(f, 2)
        c.SetStreamFeedStage(f, 2)
        sp_c = c.Specs["C"]
        sp_c.SType = ColumnSpec.SpecType.Stream_Ratio
        sp_c.SpecValue = 3.0
        sp_c.SpecUnit = ""
        sp_r = c.Specs["R"]
        sp_r.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
        sp_r.SpecValue = 0.5
        sp_r.SpecUnit = "mol/s"
        c.ConnectDistillate(d)
        c.ConnectBottoms(b)

        try:
            apply(c)
        except Exception as exc:  # noqa: BLE001
            return f"设置 {label} 失败: {type(exc).__name__}"

        errs = automation.CalculateFlowsheet4(fs)
        if errs.Count == 0:
            return f"收敛 ✅ 塔顶F={float(d.GetMolarFlow()):.4f} 塔釜F={float(b.GetMolarFlow()):.4f}"
        return str(errs[0]).splitlines()[0][:80]

    print(f"  [基线]            {attempt('基线', lambda c: None)}")
    print(f"  [设 CondPressure] {attempt('CondPressure', lambda c: setattr(c, 'CondenserPressure', 101325.0))}")
    print(f"  [设 RebPressure]  {attempt('RebPressure', lambda c: setattr(c, 'ReboilerPressure', 101325.0))}")
    print(f"  [设两者]          {attempt('两者', lambda c: (setattr(c, 'CondenserPressure', 101325.0), setattr(c, 'ReboilerPressure', 101325.0)))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
