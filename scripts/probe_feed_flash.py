"""最终定位：「PT Flash: Error calculating amount of the vapor phase」的真实来源。

上一轮所有物性包、所有构型（含纯二元水/甲苯）都报：
  Col: PT Flash: Error calculating amount of the vapor phase in the mixture

这条消息的关键是「计算气相量失败」。它出现在塔的逐板闪蒸里。
需要区分两种可能：
  (a) 物性/闪蒸算法在该体系上真的失效
  (b) 塔的某股进料处于「全气相」或「全液相」边界，导致闪蒸的相分率求解失败

做法：直接对塔的**两股进料**在各自温度压力下做独立闪蒸，看是否正常。
进料条件：
  水/甲苯 1.0 mol/s @ 327.09 K, 101.325 kPa, z=[0.5,0.5,0]
  1-丁醇  2.0 mol/s @ 325.63 K, 101.325 kPa, z=[0,0,1]

特别注意：纯 1-丁醇在 325.63 K、101.325 kPa 下远低于其沸点(390.7 K)，
应为全液相；若 DWSIM 在该条件下无法判定相态，就会报「计算气相量失败」。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def flash(automation, object_type, pp, compounds, z, T, P=101325.0, F=1.0):
    from thermo_engine.dwsim_export import _add_property_package

    flowsheet = automation.CreateFlowsheet()
    for name in compounds:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, pp)

    from System import Array, Double

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "F").GetAsObject()
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(F)
    feed.SetOverallComposition(Array[Double](z))

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return f"失败: {str(errors[0]).splitlines()[0][:75]}"

    found = []
    phases = feed.Phases
    for i in range(phases.Count):
        phase = phases[i]
        try:
            name = str(phase.Name)
        except Exception:  # noqa: BLE001
            continue
        if name in ("Mixture", "Solid", "Liquid3", "Aqueous"):
            continue
        try:
            values = [getattr(c, "MoleFraction") for c in phase.Compounds.Values]
        except Exception:  # noqa: BLE001
            continue
        if not values or any(v is None for v in values):
            continue
        floats = [float(v) for v in values]
        if any(v != v for v in floats):
            continue
        if all(abs(v) < 1e-12 for v in floats):
            continue
        found.append(f"{name}={[round(v, 3) for v in floats]}")
    return "OK  " + "  ".join(found)


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== A. 塔的两股进料在各自条件下单独闪蒸 ===")
    feeds = [
        ("水/甲苯进料 @327.09K", ["Water", "Toluene", "1-butanol"], [0.5, 0.5, 0.0], 327.09, 1.0),
        ("1-丁醇萃取剂 @325.63K", ["Water", "Toluene", "1-butanol"], [0.0, 0.0, 1.0], 325.63, 2.0),
        ("纯1-丁醇 @390.68K(沸点)", ["Water", "Toluene", "1-butanol"], [0.0, 0.0, 1.0], 390.68, 2.0),
        ("纯水 @319.55K(沸点)", ["Water", "Toluene", "1-butanol"], [1.0, 0.0, 0.0], 319.55, 1.0),
        ("纯甲苯 @374.15K(沸点)", ["Water", "Toluene", "1-butanol"], [0.0, 1.0, 0.0], 374.15, 1.0),
    ]
    for label, compounds, z, T, F in feeds:
        parts = []
        for pp in ("UNIQUAC", "NRTL"):
            try:
                r = flash(automation, object_type, pp, compounds, z, T, F=F)
            except Exception as exc:  # noqa: BLE001
                r = f"异常 {type(exc).__name__}"
            parts.append(f"{pp}: {r}")
        print(f"  {label:26}")
        for p in parts:
            print(f"      {p}")

    print("\n=== B. 纯组分闪蒸（最能暴露相态边界问题）===")
    for label, compounds, z, T in (
        ("纯 1-丁醇", ["Water", "Toluene", "1-butanol"], [0.0, 0.0, 1.0], 325.63),
        ("纯 水", ["Water", "Toluene", "1-butanol"], [1.0, 0.0, 0.0], 327.09),
        ("纯 甲苯", ["Water", "Toluene", "1-butanol"], [0.0, 1.0, 0.0], 327.09),
    ):
        for pp in ("UNIQUAC", "NRTL"):
            try:
                r = flash(automation, object_type, pp, compounds, z, T)
            except Exception as exc:  # noqa: BLE001
                r = f"异常 {type(exc).__name__}"
            print(f"  {label:10} {pp:8} @{T}K -> {r[:85]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
