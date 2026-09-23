"""Solve a DWSIM column flowsheet and report the converged product streams.

Shared diagnostic for the §1.5 case-2 work.  It reports the two product
draw-offs and the two feeds, and -- critically -- computes the overall material
balance, because "CalculateFlowsheet2 did not raise" is NOT evidence that the
column solved correctly: earlier revisions returned success while both product
streams stayed at zero flow.

Usage:
    python solve_and_report_wtb_column.py [path-to-dwxmz]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT = (
    ROOT / "data" / "exports" / "flow_examples"
    / "water_toluene_butanol_extractive_thermoformer.dwxmz"
)


def unwrap(obj: object) -> object:
    get_as_object = getattr(obj, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else obj


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT
    if not target.is_file():
        raise SystemExit(f"missing file: {target}")

    from thermo_engine.dwsim_export import _automation_factory

    factory, _object_type = _automation_factory()
    automation = factory()
    flowsheet = automation.LoadFlowsheet(str(target))

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

    print(f"file: {target.name}")
    print(f"stages: {column.NumberOfStages}   reflux: {column.RefluxRatio}")
    for key in list(column.Specs.Keys):
        spec = column.Specs[str(key)]
        print(f"  spec {key}: {spec.SType} = {spec.SpecValue} {spec.SpecUnit!r}")

    print("\nsolving ...")
    # CalculateFlowsheet2 is silent: it returns normally even when the column
    # fails its mass balance, so a "no exception" reading is meaningless.
    # CalculateFlowsheet4 returns the error list and is the only trustworthy
    # signal.  Both are run so the difference is visible.
    try:
        automation.CalculateFlowsheet2(flowsheet)
        print("  CalculateFlowsheet2: returned without raising (NOT evidence of success)")
    except Exception as exc:  # noqa: BLE001
        print(f"  CalculateFlowsheet2 raised: {type(exc).__name__}")

    try:
        errors = automation.CalculateFlowsheet4(flowsheet)
    except Exception as exc:  # noqa: BLE001
        print(f"  CalculateFlowsheet4 raised: {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    print(f"  CalculateFlowsheet4: {errors.Count} error(s)")
    for i in range(errors.Count):
        print(f"    [{i}] {str(errors[i]).splitlines()[0][:160]}")
    converged = errors.Count == 0

    print(f"\n{'stream':14} {'role':12} {'T(K)':>8} {'F(mol/s)':>10}  x(Water/Tol/BuOH)")
    print("-" * 72)
    total = 0.0
    inflow = 0.0
    for key, stream in streams.items():
        try:
            tray = int(column.GetStreamFeedStageIndex(stream))
            role = f"feed@{tray + 1}"
        except Exception:  # noqa: BLE001 - product outlets are not feeds
            role = "PRODUCT"
        t = float(stream.GetTemperature())
        f = float(stream.GetMolarFlow())
        phases = stream.Phases
        x = (
            [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]
            if phases.Count > 0
            else []
        )
        print(f"{key[:14]:14} {role:12} {t:8.2f} {f:10.4f}  {x}")
        if role == "PRODUCT":
            total += f
        else:
            inflow += f

    print("-" * 72)
    print(f"feed total   = {inflow:.4f} mol/s")
    print(f"product total= {total:.4f} mol/s")
    if not converged:
        print("=> column did NOT converge; product flows are meaningless")
        return 2
    if inflow > 0 and abs(total - inflow) < 1e-3:
        print("=> converged and the overall balance CLOSES")
        return 0
    print("=> converged but the overall balance does not close")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
