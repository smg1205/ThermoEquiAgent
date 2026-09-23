"""验证「严格塔求解器能力不足」假设，并用 DWSIM 简捷塔作为可行替代。

隔离实验显示误差随体系难度单调增大（呈连续谱）：
  乙醇/水 二元  物料平衡误差 -0.00011  （几乎收敛）
  水/甲苯 二元  物料平衡误差 -0.342
  水/甲苯 高回流 物料平衡误差 -0.150
  三元完整设计  迭代不收敛

这不像「某个开关坏了」，更像求解器在难度上升时逐步失效。
本脚本做两件事：

  A. 简捷塔（ShortcutColumn）对照
     简捷塔基于 FUG，正是 §1.5 设计数据的来源，不依赖逐板迭代。
     若它能算出结果，则「文件算不出结果」这一问题有可行解。

  B. 严格塔求解器能力探针
     对同一体系逐步降低要求（减少级数 / 提高回流比 / 简化组成），
     找出严格塔能收敛的边界，以确认是否为能力问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def shortcut_column(automation, object_type, *, light, heavy, feed_comp,
                    feed_flow, feed_temp, stages, reflux, distillate_flow,
                    pressure_kpa=101.325):
    """用 DWSIM 简捷塔（FUG）计算，返回摘要。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in ("Water", "Toluene"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double

    col = flowsheet.AddObject(object_type.ShortcutColumn, 250, 0, "SC").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()

    feed.SetTemperature(feed_temp)
    feed.SetPressure(pressure_kpa * 1000.0)
    feed.SetMolarFlow(feed_flow)
    feed.SetOverallComposition(Array[Double](feed_comp))

    print(f"  简捷塔成员: {[m for m in dir(col) if 'Light' in m or 'Heavy' in m or 'Reflux' in m][:8]}")

    for attr, value in (
        ("LightKeyCompound", light),
        ("HeavyKeyCompound", heavy),
        ("NumberOfStages", stages),
        ("RefluxRatio", reflux),
        ("DistillateFlowRate", distillate_flow),
    ):
        try:
            setattr(col, attr, value)
        except Exception as exc:  # noqa: BLE001
            print(f"    设 {attr} 失败: {type(exc).__name__}")

    for obj, other in ((col, feed), (col, dist)):
        try:
            flowsheet.ConnectObjects(obj.GraphicObject, other.GraphicObject, 0, 0)
        except Exception:  # noqa: BLE001
            pass
    try:
        flowsheet.ConnectObjects(col.GraphicObject, bot.GraphicObject, 1, 0)
    except Exception:  # noqa: BLE001
        pass

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return f"失败: {str(errors[0]).splitlines()[0][:80]}"
    xd = [round(float(c.MoleFraction), 4) for c in dist.Phases[0].Compounds.Values]
    xb = [round(float(c.MoleFraction), 4) for c in bot.Phases[0].Compounds.Values]
    return (f"OK  D={float(dist.GetMolarFlow()):.4f} x={xd} | "
            f"B={float(bot.GetMolarFlow()):.4f} x={xb}")


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. DWSIM 简捷塔（FUG）能否算出 水/甲苯 分离 ==")
    print(f"  输入: 水/甲苯 0.5/0.5, 1.0 mol/s @338.97K, N=18, R=2.095, D=0.4737")
    print(f"  结果: {shortcut_column(automation, object_type, light='Water', heavy='Toluene',
                                   feed_comp=[0.5, 0.5], feed_flow=1.0, feed_temp=338.97,
                                   stages=18, reflux=2.095, distillate_flow=0.4737)}")

    print("\n=== B. 严格塔能力边界：水/甲苯，逐级简化 ===")
    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec
    from thermo_engine.dwsim_export import _add_property_package

    def strict(stages, reflux, feed_stage, bottoms, feed_temp=338.97):
        fs = automation.CreateFlowsheet()
        for name in ("Water", "Toluene"):
            fs.AddCompound(name)
        _add_property_package(fs, "UNIQUAC")
        col = fs.AddObject(object_type.DistillationColumn, 250, 0, "C").GetAsObject()
        d = fs.AddObject(object_type.MaterialStream, 500, -80, "D").GetAsObject()
        b = fs.AddObject(object_type.MaterialStream, 500, 80, "B").GetAsObject()
        f = fs.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()

        col.SetNumberOfStages(stages)
        col.RefluxRatio = reflux
        col.MaxIterations = 5000
        f.SetTemperature(feed_temp)
        f.SetPressure(101325.0)
        f.SetMolarFlow(1.0)
        f.SetOverallComposition(Array[Double]([0.5, 0.5]))
        col.ConnectFeed(f, feed_stage)
        col.SetStreamFeedStage(f, feed_stage)
        c = col.Specs["C"]
        c.SType = ColumnSpec.SpecType.Stream_Ratio
        c.SpecValue = reflux
        c.SpecUnit = ""
        r = col.Specs["R"]
        r.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
        r.SpecValue = bottoms
        r.SpecUnit = "mol/s"
        col.ConnectDistillate(d)
        col.ConnectBottoms(b)

        errs = automation.CalculateFlowsheet4(fs)
        if errs.Count == 0:
            xd = [round(float(cc.MoleFraction), 3) for cc in d.Phases[0].Compounds.Values]
            return f"收敛 OK  D={float(d.GetMolarFlow()):.4f} x={xd}"
        first = str(errs[0]).splitlines()[0]
        if "mass balance" in first:
            val = first.split("Relative Error =")[-1].split("[")[0].strip()
            return f"物料平衡 {val[:16]}"
        return first[:55]

    trials = [
        ("5级 R=5 进料板3 B=0.5", dict(stages=5, reflux=5.0, feed_stage=2, bottoms=0.5)),
        ("8级 R=5 进料板4 B=0.5", dict(stages=8, reflux=5.0, feed_stage=3, bottoms=0.5)),
        ("10级 R=8 进料板5 B=0.5", dict(stages=10, reflux=8.0, feed_stage=4, bottoms=0.5)),
        ("18级 R=10 进料板9 B=0.5", dict(stages=18, reflux=10.0, feed_stage=8, bottoms=0.5)),
        ("18级 R=2.095 进料板8 B=0.4737(设计)", dict(stages=18, reflux=2.095, feed_stage=7, bottoms=0.4737)),
    ]
    for label, kwargs in trials:
        print(f"  {label:34} -> {strict(**kwargs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
