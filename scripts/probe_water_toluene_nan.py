"""聚焦验证：UNIQUAC 下水/甲苯的闪蒸是否产生 NaN（相态计算失效）。

线索：早先的 flash 探针中，水/甲苯在 373.15 K 时某个相的组成读到 [nan, nan]，
而其它体系（乙醇/水、水/丁醇、甲苯/丁醇）未出现 NaN。
同时严格塔对水/甲苯始终不收敛，连 5 级 R=5 都不行。

若闪蒸在塔内的工作区间产生 NaN，逐板迭代必然崩掉 —— 这与「求解器能力不足」
是两回事，属于物性/参数问题。

本脚本：
  A. 扫描水/甲苯在不同 T 与组成下的闪蒸，统计 NaN / 失败出现的位置
  B. 对比其它二元对，确认是否为水/甲苯特有
  C. 检查 UNIQUAC 对水/甲苯的参数是否异常（如 DWSIM 估算值过大）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def flash(automation, object_type, compounds, z, T, P=101325.0):
    """返回 (成功与否, 摘要, 是否出现 NaN)。"""
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
        return False, str(errors[0]).splitlines()[0][:70], False

    phases = feed.Phases
    nan_seen = False
    detailed = []
    for i in range(phases.Count):
        phase = phases[i]
        for attr in ("MoleFraction", "MolarFraction"):
            try:
                values = [getattr(c, attr) for c in phase.Compounds.Values]
            except Exception:  # noqa: BLE001
                continue
            if not values or any(v is None for v in values):
                continue
            floats = [float(v) for v in values]
            if any(v != v for v in floats):      # NaN 检测
                nan_seen = True
            # 只记录有实际组成的相（全 0 相无意义）
            if any(abs(v) > 1e-9 for v in floats):
                detailed.append(f"相{i}={[round(v, 3) for v in floats]}")
            break
    return True, f"相数={phases.Count} " + " ".join(detailed[:3]), nan_seen


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. 水/甲苯 温度扫描（找 NaN / 失败区间）===")
    nan_range = []
    for T in (300.0, 320.0, 340.0, 350.0, 360.0, 365.0, 370.0, 373.0, 375.0, 380.0, 390.0):
        try:
            ok, detail, nan = flash(automation, object_type, ["Water", "Toluene"], [0.5, 0.5], T)
        except Exception as exc:  # noqa: BLE001
            print(f"  T={T:6.1f}  抛异常 {type(exc).__name__}")
            continue
        flag = "NaN!" if nan else ("OK  " if ok else "FAIL")
        if nan:
            nan_range.append(T)
        print(f"  T={T:6.1f}  {flag} {detail[:80]}")
    print(f"  -> 出现 NaN 的温度: {nan_range if nan_range else '无'}")

    print("\n=== B. 组成扫描（固定 T=360 K）===")
    for z_w in (0.1, 0.3, 0.5, 0.7, 0.9):
        try:
            ok, detail, nan = flash(
                automation, object_type, ["Water", "Toluene"], [z_w, 1 - z_w], 360.0
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  x_水={z_w:.1f}  抛异常 {type(exc).__name__}")
            continue
        flag = "NaN!" if nan else ("OK  " if ok else "FAIL")
        print(f"  x_水={z_w:.1f}  {flag} {detail[:80]}")

    print("\n=== C. 其它二元对对照（T=360 K, 等摩尔）===")
    for label, compounds in (
        ("乙醇/水", ["Ethanol", "Water"]),
        ("水/1-丁醇", ["Water", "1-butanol"]),
        ("甲苯/1-丁醇", ["Toluene", "1-butanol"]),
        ("水/甲苯", ["Water", "Toluene"]),
        ("甲醇/水", ["Methanol", "Water"]),
    ):
        try:
            ok, detail, nan = flash(automation, object_type, compounds, [0.5, 0.5], 360.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:12} 抛异常 {type(exc).__name__}")
            continue
        flag = "NaN!" if nan else ("OK  " if ok else "FAIL")
        print(f"  {label:12} {flag} {detail[:75]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
