"""验证假设：水/甲苯的液液分相导致严格精馏塔不收敛。

重大发现：Water/Toluene 在 360 K、101.325 kPa 下 TP flash 分出两个液相：
  Liquid1 = 0.0034 / 0.9966   （富甲苯）
  Liquid2 = 0.9998 / 0.0002   （富水）
即水/甲苯是【部分互溶】体系，塔内会出现液液分层。

（注：先前把 OverallLiquid 槽位的 NaN 误判为「闪蒸产生 NaN」。8 个
 Phases 槽位是 DWSIM 固定槽位，非 8 个真实相；真实相为 Liquid1/Liquid2。）

严格精馏塔（DistillationColumn）按气液两相逐板计算，塔内一旦液液分层，
逐板迭代就会失效——这与「求解器能力」无关，是模型适用范围问题。

本脚本对比不同物性包 / 组成条件，检验该假设：
  A. 水/甲苯：UNIQUAC vs NRTL vs Raoult 的相分裂行为
  B. 若避开液液分层区（把水/甲苯稀释到单液相），严格塔是否收敛
  C. 确认乙醇/水（可收敛）与 水/甲苯（不收敛）的相行为差异
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_SLOTS = ("Mixture", "OverallLiquid", "Vapor", "Liquid1", "Liquid2",
               "Liquid3", "Aqueous", "Solid")


def flash(automation, object_type, compounds, z, T, pp="UNIQUAC", P=101325.0):
    """返回 (错误或None, {相名: 组成})，只看真实填充的液相/气相槽位。"""
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, pp)

    from System import Array, Double

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double](z))

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return str(errors[0]).splitlines()[0][:70], {}

    found = {}
    phases = feed.Phases
    for i in range(phases.Count):
        phase = phases[i]
        try:
            name = str(phase.Name)
        except Exception:  # noqa: BLE001
            name = PHASE_SLOTS[i] if i < len(PHASE_SLOTS) else f"phase{i}"
        try:
            values = [getattr(c, "MoleFraction") for c in phase.Compounds.Values]
        except Exception:  # noqa: BLE001
            continue
        if not values or any(v is None for v in values):
            continue
        floats = [float(v) for v in values]
        if any(v != v for v in floats):     # NaN 槽位跳过
            continue
        if all(abs(v) < 1e-12 for v in floats):
            continue
        found[name] = [round(v, 4) for v in floats]
    return None, found


def liquid_split(found: dict) -> int:
    """统计真实液相个数（Liquid1/Liquid2/...）。"""
    return sum(1 for k in found if k.startswith("Liquid") and k != "OverallLiquid")


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. 水/甲苯 等摩尔 360 K：不同物性包的相分裂 ===")
    for pp in ("UNIQUAC", "NRTL", "Raoult's Law"):
        try:
            err, found = flash(automation, object_type, ["Water", "Toluene"],
                               [0.5, 0.5], 360.0, pp=pp)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pp:14} 抛异常 {type(exc).__name__}")
            continue
        if err:
            print(f"  {pp:14} 失败: {err}")
            continue
        n_liq = liquid_split(found)
        print(f"  {pp:14} 真实液相数={n_liq}  {found}")

    print("\n=== B. 水/甲苯 组成扫描：何时成为单液相（UNIQUAC, 360 K）===")
    for z_w in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        try:
            err, found = flash(automation, object_type, ["Water", "Toluene"],
                               [z_w, 1 - z_w], 360.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  x_水={z_w:<6} 抛异常 {type(exc).__name__}")
            continue
        if err:
            print(f"  x_水={z_w:<6} 失败 {err}")
            continue
        n_liq = liquid_split(found)
        mark = "← 分层" if n_liq > 1 else ""
        print(f"  x_水={z_w:<6} 液相数={n_liq} {mark}")

    print("\n=== C. 对照：能收敛的体系是否单液相 ===")
    for label, compounds, T in (
        ("乙醇/水", ["Ethanol", "Water"], 360.0),
        ("水/1-丁醇", ["Water", "1-butanol"], 360.0),
        ("甲醇/水", ["Methanol", "Water"], 360.0),
        ("水/甲苯", ["Water", "Toluene"], 360.0),
    ):
        try:
            err, found = flash(automation, object_type, compounds, [0.5, 0.5], T)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:12} 抛异常 {type(exc).__name__}")
            continue
        if err:
            print(f"  {label:12} 失败 {err}")
            continue
        print(f"  {label:12} 液相数={liquid_split(found)}  {found}")

    print("\n=== D. 三元完整体系（含丁醇）在塔内工作温度的相行为 ===")
    for label, z, T in (
        ("富水端(塔顶附近)", [0.90, 0.05, 0.05], 340.0),
        ("富丁醇端(塔釜附近)", [0.03, 0.19, 0.78], 380.0),
        ("中部", [0.2, 0.3, 0.5], 365.0),
    ):
        try:
            err, found = flash(
                automation, object_type, ["Water", "Toluene", "1-butanol"], z, T
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:20} 抛异常 {type(exc).__name__}")
            continue
        if err:
            print(f"  {label:20} 失败 {err}")
            continue
        print(f"  {label:20} 液相数={liquid_split(found)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
