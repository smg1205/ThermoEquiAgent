"""Probe DWSIM rigorous-column convergence with a controlled water/1-butanol case.

Purpose: establish a *known-good* baseline so the §1.5 case-2 column can be
judged.  Reported symptoms to explain:
  * ``CalculateFlowsheet2`` returns without raising, yet product streams hold
    zero flow;
  * one product carried the other's flow rate.

Those symptoms mean "did not raise" is not evidence of convergence, so this
script prints the solver's own convergence report and the per-stage profile
using the *reflected* member names (``MaterialStreams``, ``Stages``,
``GetStreamFeedStageIndex``) rather than guessed attribute spellings.

Run with full process access (pythonnet needs OpenProcess).
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
    for name in ("Water", "1-butanol"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()

    stages, reflux = 12, 3.0
    col.NumberOfStages = stages
    col.RefluxRatio = reflux

    feed.SetTemperature(365.0)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double]([0.5, 0.5]))
    col.ConnectFeed(feed, 5)

    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    bottoms_flow = 0.5
    spec = col.Specs["R"]
    spec.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    spec.SpecValue = bottoms_flow
    spec.SpecUnit = "mol/s"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    # Seed a starting profile; a cold start with no estimates is the documented
    # cause of iteration-cap failures in this repo's exporter.
    for i in range(stages):
        try:
            stage = col.Stages[i]
            stage.Temperature = 365.0 + 20.0 * (i / max(stages - 1, 1))
        except Exception:  # noqa: BLE001 - optional warm start
            break

    print(f"stages={stages} reflux={reflux} bottoms_spec={bottoms_flow} mol/s")
    print(f"registered streams: {col.MaterialStreams.Count}")
    print("\nsolving ...")
    try:
        automation.CalculateFlowsheet2(flowsheet)
        print("  no exception")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:250]}")

    print(f"\n{'stream':8} {'tray':>5} {'T(K)':>9} {'F(mol/s)':>10}  x(Water/BuOH)")
    print("-" * 56)
    for label, stream in (("feed", feed), ("dist", dist), ("bot", bot)):
        try:
            tray = int(col.GetStreamFeedStageIndex(stream))
        except Exception:  # noqa: BLE001
            tray = -1
        t = float(stream.GetTemperature())
        f = float(stream.GetMolarFlow())
        phases = stream.Phases
        x = (
            [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]
            if phases.Count > 0
            else []
        )
        print(f"{label:8} {tray:5} {t:9.2f} {f:10.4f}  {x}")

    print("\nper-stage liquid composition:")
    for i in range(stages):
        try:
            stage = col.Stages[i]
            t = stage.Temperature
            comps = stage.LiquidPhaseCompounds
            x = [round(float(c.MoleFraction), 3) for c in comps.Values]
        except Exception as exc:  # noqa: BLE001
            print(f"  stage{i + 1}: <{type(exc).__name__}>")
            continue
        print(f"  stage{i + 1:2}: T={t:7.2f}  x={x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
