"""追查 PT Flash 失败：「Error calculating amount of the vapor phase in the mixture」。

级数同步问题修复后（Stages.Count 正确），水/甲苯二元 18 级塔报出新错误：
  Col: PT Flash: Error calculating amount of the vapor phase in the mixture

这比「索引超出范围」更靠后，说明求解器已真正进入计算。该错误来自
DWSIM 的闪蒸例程，通常指 flash 的收敛/物性计算失败。

三元 1-丁醇/水/甲苯 塔仍报 Water 物料平衡 0.99999（与修复前完全相同），
提示这是与级数无关的第二个独立问题。

本脚本直接对相关混合物做 TP flash，定位哪个组成/温度下闪蒸失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def flash(automation, object_type, compounds, z, T, P=101325.0):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double](z))

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return False, str(errors[0]).splitlines()[0][:95]

    phases = feed.Phases
    n_phases = phases.Count
    detail = []
    for i in range(n_phases):
        phase = phases[i]
        # MoleFraction 可能返回 None（该属性在此上下文不适用），
        # 依次尝试其它读法，避免把读取失败误判为 flash 失败。
        comp = None
        for attr in ("MoleFraction", "MolarFraction", "x", "Composition"):
            try:
                values = [getattr(c, attr) for c in phase.Compounds.Values]
            except Exception:  # noqa: BLE001
                continue
            if values and all(v is not None for v in values):
                comp = [round(float(v), 4) for v in values]
                break
        if comp is None:
            detail.append(f"相{i}: <组成读取失败>")
        else:
            detail.append(f"相{i}: {comp}")
    return True, f"相数={n_phases} " + " ".join(detail)


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    # 先做一次最小 flash，确认调用路径可用；失败则打印完整异常
    print("=== 0. 调用路径自检 ===")
    try:
        ok, detail = flash(
            automation, object_type, ["Water", "Toluene"], [0.5, 0.5], 350.0
        )
        print(f"  自检: {'OK ' if ok else 'FAIL'} {detail}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"  自检抛异常: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("=== 1. 单相/两相基本闪蒸（检验物性包可用性）===")
    checks = [
        ("Water/Toluene", ["Water", "Toluene"], [0.5, 0.5], 350.0),
        ("Water/Toluene", ["Water", "Toluene"], [0.5, 0.5], 373.15),
        ("Water/1-butanol", ["Water", "1-butanol"], [0.5, 0.5], 365.0),
        ("Toluene/1-butanol", ["Toluene", "1-butanol"], [0.5, 0.5], 380.0),
        ("水/甲苯/丁醇 三元", ["Water", "Toluene", "1-butanol"], [0.333, 0.333, 0.334], 360.0),
    ]
    for label, compounds, z, T in checks:
        try:
            ok, detail = flash(automation, object_type, compounds, z, T)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:22} T={T:6.1f}  抛异常 {type(exc).__name__}")
            continue
        print(f"  {label:22} T={T:6.1f}  {'OK ' if ok else 'FAIL'} {detail}")

    print("\n=== 2. 沿塔温度扫描三元闪蒸（找失败区间）===")
    wtb = ["Water", "Toluene", "1-butanol"]
    z_mid = [0.2, 0.2, 0.6]
    for T in (300.0, 320.0, 340.0, 360.0, 375.0, 380.0, 390.0, 400.0):
        try:
            ok, detail = flash(automation, object_type, wtb, z_mid, T)
        except Exception as exc:  # noqa: BLE001
            print(f"  T={T:6.1f}  抛异常 {type(exc).__name__}")
            continue
        flag = "OK  " if ok else "FAIL"
        print(f"  T={T:6.1f}  {flag} {detail}")

    print("\n=== 3. 塔顶/塔釜设计组成的闪蒸 ===")
    design = [
        ("塔顶 D (水0.95/甲苯0.05)", [0.95, 0.05, 0.0], 282.47),
        ("塔釜 B (水0.0198/甲苯0.1885/丁醇0.7917)", [0.0198, 0.1885, 0.7917], 378.73),
    ]
    for label, z, T in design:
        try:
            ok, detail = flash(automation, object_type, wtb, z, T)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:42} 抛异常 {type(exc).__name__}")
            continue
        print(f"  {label:42} {'OK ' if ok else 'FAIL'} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
