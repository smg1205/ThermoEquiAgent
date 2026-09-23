"""校验 水/乙酸 + DMSO 的 DWSIM 萃取塔文件。

重新打开生成的 .dwxmz，逐项核对：组分、级数、回流比、两股进料的板位与组成、
产品出口是否接好，并尝试求解（用 CalculateFlowsheet4 读取真实错误，
CalculateFlowsheet2 会吞掉错误，不可用于判定）。

用法: python verify_aw_dmso_file.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = (
    ROOT / "data" / "exports" / "flow_examples"
    / "water_acetic_acid_dmso_extractive_thermoformer.dwxmz"
)

EXPECTED_COMPOUNDS = ["Water", "Acetic acid", "Dimethyl sulfoxide"]
EXPECTED_STAGES = 15
EXPECTED_R = 1.441
EXPECTED_ENTRAINER_TRAY = 1        # 0-based -> 第 2 板
EXPECTED_FEED_TRAY = 6             # 0-based -> 第 7 板
TOL = 1e-3

failures: list[str] = []


def unwrap(obj: object) -> object:
    get_as_object = getattr(obj, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else obj


def check(label: str, actual: object, expected: object, tol: float | None = None) -> None:
    if tol is not None and isinstance(actual, float) and isinstance(expected, float):
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:30} actual={actual!r} expected={expected!r}")
    if not ok:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"文件不存在: {TARGET}")

    from thermo_engine.dwsim_export import _automation_factory

    factory, _object_type = _automation_factory()
    automation = factory()
    flowsheet = automation.LoadFlowsheet(str(TARGET))

    print(f"file: {TARGET.name} ({TARGET.stat().st_size} bytes)\n")

    compounds = [str(k) for k in flowsheet.SelectedCompounds.Keys]
    print("=== 组分 ===")
    for name in compounds:
        print(f"  - {name}")
    check("compounds", compounds, EXPECTED_COMPOUNDS)

    sim_objs = flowsheet.SimulationObjects
    column = None
    streams: dict[str, object] = {}
    for key in [str(k) for k in sim_objs.Keys]:
        inner = unwrap(sim_objs[key])
        type_name = str(inner.GetType().Name)
        if "Column" in type_name:
            column = inner
        elif type_name == "MaterialStream":
            streams[key] = inner

    if column is None:
        raise SystemExit("未找到塔对象")

    print(f"\n=== 塔 ({column.GetType().Name}) ===")
    check("NumberOfStages", int(column.NumberOfStages), EXPECTED_STAGES)
    check("Stages.Count（须与级数同步）", int(column.Stages.Count), EXPECTED_STAGES)
    check("RefluxRatio", round(float(column.RefluxRatio), 3), EXPECTED_R, tol=TOL)

    print("\n=== 规格 ===")
    for key in list(column.Specs.Keys):
        spec = column.Specs[str(key)]
        print(f"  {key}: {spec.SType} = {spec.SpecValue} {spec.SpecUnit!r}")

    print("\n=== 进料 / 产品 ===")
    feeds: dict[int, object] = {}
    products: list[str] = []
    attached = {str(k) for k in column.MaterialStreams.Keys}
    check("已连接的流股数", len(attached), len(streams))

    for key, stream in streams.items():
        try:
            tray = int(column.GetStreamFeedStageIndex(stream))
        except Exception:  # noqa: BLE001
            tray = -1
        if tray < 0:
            products.append(key)
            print(f"  product {key[:24]}（无进料板 => 产品）")
            continue
        feeds[tray] = stream
        print(f"  feed    {key[:24]} -> 板索引 {tray}（第 {tray + 1} 板）")

    check("进料股数", len(feeds), 2)
    check("产品出口数", len(products), 2)

    def composition(stream: object) -> list[float]:
        phases = stream.Phases
        if phases.Count == 0:
            return []
        return [round(float(c.MoleFraction), 3) for c in phases[0].Compounds.Values]

    if EXPECTED_ENTRAINER_TRAY in feeds:
        ent = feeds[EXPECTED_ENTRAINER_TRAY]
        print("\n  萃取剂 DMSO:")
        check("  板位", EXPECTED_ENTRAINER_TRAY, EXPECTED_ENTRAINER_TRAY)
        check("  组成", composition(ent), [0.0, 0.0, 1.0])
        check("  流量 mol/s", round(float(ent.GetMolarFlow()), 6), 2.0, tol=TOL)
        check("  温度 K", round(float(ent.GetTemperature()), 2), 354.1, tol=0.01)
    else:
        failures.append(f"第 {EXPECTED_ENTRAINER_TRAY} 板无萃取剂进料")

    if EXPECTED_FEED_TRAY in feeds:
        feed = feeds[EXPECTED_FEED_TRAY]
        print("\n  进料 水/乙酸:")
        check("  板位", EXPECTED_FEED_TRAY, EXPECTED_FEED_TRAY)
        check("  组成", composition(feed), [0.5, 0.5, 0.0])
        check("  流量 mol/s", round(float(feed.GetMolarFlow()), 6), 1.0, tol=TOL)
        check("  温度 K", round(float(feed.GetTemperature()), 2), 366.01, tol=0.01)
    else:
        failures.append(f"第 {EXPECTED_FEED_TRAY} 板无进料")

    print("\n=== 求解（CalculateFlowsheet4 才报告真实错误）===")
    try:
        errors = automation.CalculateFlowsheet4(flowsheet)
        if errors.Count == 0:
            print("  收敛 ✅")
            for key, stream in streams.items():
                t = float(stream.GetTemperature())
                f = float(stream.GetMolarFlow())
                x = composition(stream)
                print(f"    {key[:24]} T={t:7.2f} F={f:7.4f} x={x}")
        else:
            print(f"  未收敛（{errors.Count} 个错误）:")
            for i in range(min(errors.Count, 3)):
                print(f"    {str(errors[i]).splitlines()[0][:130]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  抛异常: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

    print("\n=== 结果 ===")
    if failures:
        for failure in failures:
            print(f"  ! {failure}")
        print(f"\n  {len(failures)} 项未通过")
        return 1
    print("  结构校验全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
