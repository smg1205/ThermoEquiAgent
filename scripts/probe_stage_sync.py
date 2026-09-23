"""追查两个具体嫌疑：Stages 列表未同步、ColumnPressureDrop 为 NaN。

上一轮发现新建塔的 ColumnPressureDrop = nan，且 Stages 是独立于
NumberOfStages 的 List。若设置 NumberOfStages 不重建 Stages 列表，
内部按级数索引该 List 时就会抛
  ArgumentOutOfRangeException: 索引超出范围

本脚本验证：
  A. NumberOfStages 与 Stages.Count 是否同步
  B. ColumnPressureDrop 的 NaN 是否导致失败
  C. DWSIM 自带的 AddStages / 其它初始化方法能否修复
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_column(automation, object_type, stages):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in ("Ethanol", "Water"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")
    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    col.NumberOfStages = stages
    return flowsheet, col


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. NumberOfStages 与 Stages.Count 是否同步 ===")
    for requested in (5, 10, 18):
        flowsheet, col = make_column(automation, object_type, requested)
        print(
            f"  设 NumberOfStages={requested:2}  ->  NumberOfStages={col.NumberOfStages:2}  "
            f"Stages.Count={col.Stages.Count:2}  "
            f"同步={'是' if col.Stages.Count == col.NumberOfStages else '否 ← 问题'}"
        )

    print("\n=== B. ColumnPressureDrop 默认值 ===")
    flowsheet, col = make_column(automation, object_type, 10)
    print(f"  默认 ColumnPressureDrop = {col.ColumnPressureDrop!r}")
    print(f"  是否为 NaN: {col.ColumnPressureDrop != col.ColumnPressureDrop}")

    print("\n=== C. 设置压降能否消除失败 ===")
    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    def trial(label, pressure_drop, do_add_stages):
        fs = automation.CreateFlowsheet()
        for name in ("Ethanol", "Water"):
            fs.AddCompound(name)
        from thermo_engine.dwsim_export import _add_property_package

        _add_property_package(fs, "UNIQUAC")
        c = fs.AddObject(object_type.DistillationColumn, 250, 0, "C").GetAsObject()
        d = fs.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
        b = fs.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()
        f = fs.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()

        c.NumberOfStages = 10
        c.RefluxRatio = 3.0
        if do_add_stages:
            try:
                c.AddStages(10)
            except Exception as exc:  # noqa: BLE001
                return f"AddStages 失败 {type(exc).__name__}"
        if pressure_drop is not None:
            try:
                c.ColumnPressureDrop = pressure_drop
            except Exception as exc:  # noqa: BLE001
                return f"设压降失败 {type(exc).__name__}"

        f.SetTemperature(365.0)
        f.SetPressure(101325.0)
        f.SetMolarFlow(1.0)
        f.SetOverallComposition(Array[Double]([0.5, 0.5]))
        c.ConnectFeed(f, 4)
        c.SetStreamFeedStage(f, 4)
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

        errs = automation.CalculateFlowsheet4(fs)
        if errs.Count == 0:
            return f"收敛 ✅ 顶F={float(d.GetMolarFlow()):.4f} 釜F={float(b.GetMolarFlow()):.4f}"
        return str(errs[0]).splitlines()[0][:75]

    print(f"  压降=None  AddStages=否 : {trial('a', None, False)}")
    print(f"  压降=0     AddStages=否 : {trial('b', 0.0, False)}")
    print(f"  压降=5000  AddStages=否 : {trial('c', 5000.0, False)}")
    print(f"  压降=5000  AddStages=是 : {trial('d', 5000.0, True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
