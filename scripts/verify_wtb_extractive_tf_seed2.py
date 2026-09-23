"""Verify the generated §1.5 case-2 DWSIM column file.

Reopens ``water_toluene_butanol_extractive_thermoformer.dwxmz`` through the
DWSIM Automation API and asserts what the file actually contains: the three
compounds, the 18-stage column, its reflux ratio and pressure drop, and -- the
check that matters most -- that **both feeds are bound to their design trays**
with the right composition, temperature and flow.

Notes on the DWSIM API (learned the hard way):
  * Objects from ``flowsheet.SimulationObjects`` are generic wrappers; real
    properties live on the inner object from ``GetAsObject()``.
  * ``MaterialStream`` exposes state through ``GetTemperature/GetPressure/
    GetMolarFlow()`` and ``Phases[0].Compounds`` -- not as plain attributes.
  * A feed's tray is read from the **column** via ``GetStreamFeedStageIndex``;
    product outlets raise ``NullReferenceException`` there (they are not feeds).

Run with full process access (pythonnet needs OpenProcess).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = (
    ROOT / "data" / "exports" / "flow_examples"
    / "water_toluene_butanol_extractive_thermoformer.dwxmz"
)

EXPECTED_COMPOUNDS = ["Water", "Toluene", "1-butanol"]
EXPECTED_STAGES = 18
EXPECTED_R = 2.095
EXPECTED_DP_PA = 5000.0
#: 0-based tray indices, matching design stages 2 (entrainer) and 8 (feed).
EXPECTED_ENTRAINER_TRAY = 1
EXPECTED_FEED_TRAY = 7
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
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:34} actual={actual!r} expected={expected!r}")
    if not ok:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing file: {TARGET}")

    from thermo_engine.dwsim_export import _automation_factory

    factory, _object_type = _automation_factory()
    automation = factory()

    print(f"file: {TARGET.name} ({TARGET.stat().st_size} bytes)\n")
    flowsheet = automation.LoadFlowsheet(str(TARGET))

    compounds = [str(name) for name in flowsheet.SelectedCompounds.Keys]
    print("=== 组分 ===")
    for name in compounds:
        print(f"  - {name}")
    check("compounds", compounds, EXPECTED_COMPOUNDS)

    sim_objs = flowsheet.SimulationObjects
    column = None
    streams: dict[str, object] = {}
    for key in [str(k) for k in sim_objs.Keys]:
        inner = unwrap(sim_objs[key])
        type_name = inner.GetType().Name
        if "Column" in str(type_name):
            column = inner
        elif str(type_name) == "MaterialStream":
            streams[key] = inner

    if column is None:
        raise SystemExit("no column object found in the flowsheet")
    if len(streams) != 4:
        failures.append(f"expected 4 material streams, found {len(streams)}")

    print(f"\n=== 塔 ({column.GetType().Name}) ===")
    check("NumberOfStages", int(column.NumberOfStages), EXPECTED_STAGES)
    check("RefluxRatio", round(float(column.RefluxRatio), 3), EXPECTED_R, tol=TOL)
    check("ColumnPressureDrop", float(column.ColumnPressureDrop), EXPECTED_DP_PA, tol=TOL)

    # Classify streams into feeds vs products.
    #
    # Only ``GetStreamFeedStageIndex`` identifies a feed, but its "not a feed"
    # result is NOT reliable: this build returns -1 for product outlets in some
    # states and raises NullReferenceException in others.  Both are treated as
    # "not a feed"; a genuine feed returns a non-negative 0-based tray index.
    # ``column.MaterialStreams`` is the authoritative registry of what is
    # attached at all, so count against it rather than against the flowsheet.
    print("\n=== 进料 / 产品 ===")
    feeds: dict[int, object] = {}
    products: list[str] = []
    attached = {str(k) for k in column.MaterialStreams.Keys}
    check("streams attached to column", len(attached), len(streams))

    for key, stream in streams.items():
        try:
            tray = int(column.GetStreamFeedStageIndex(stream))
        except Exception:  # noqa: BLE001 - product outlets may raise here
            tray = -1
        if tray < 0:
            products.append(key)
            print(f"  product {key[:24]} (no feed tray)")
            continue
        feeds[tray] = stream
        print(f"  feed    {key[:24]} -> tray index {tray} (stage {tray + 1})")

    check("number of bound feeds", len(feeds), 2)
    check("product outlet count", len(products), 2)

    def composition(stream: object) -> list[float]:
        phases = stream.Phases
        if phases.Count == 0:
            return []
        return [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]

    if EXPECTED_ENTRAINER_TRAY in feeds:
        ent = feeds[EXPECTED_ENTRAINER_TRAY]
        print("\n  萃取剂 (1-丁醇):")
        check("  tray", EXPECTED_ENTRAINER_TRAY, EXPECTED_ENTRAINER_TRAY)
        check("  x", composition(ent), [0.0, 0.0, 1.0])
        check("  flow mol/s", round(float(ent.GetMolarFlow()), 6), 2.0, tol=TOL)
        check("  T K", round(float(ent.GetTemperature()), 2), 325.63, tol=0.01)
    else:
        failures.append(f"no entrainer feed on tray {EXPECTED_ENTRAINER_TRAY}")

    if EXPECTED_FEED_TRAY in feeds:
        f = feeds[EXPECTED_FEED_TRAY]
        print("\n  进料 (水/甲苯):")
        check("  tray", EXPECTED_FEED_TRAY, EXPECTED_FEED_TRAY)
        check("  x", composition(f), [0.5, 0.5, 0.0])
        check("  flow mol/s", round(float(f.GetMolarFlow()), 6), 1.0, tol=TOL)
        check("  T K", round(float(f.GetTemperature()), 2), 327.09, tol=0.01)
    else:
        failures.append(f"no key feed on tray {EXPECTED_FEED_TRAY}")

    print("\n=== 结果 ===")
    if failures:
        for failure in failures:
            print(f"  ! {failure}")
        print(f"\n  {len(failures)} 项未通过")
        return 1
    print("  全部校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
