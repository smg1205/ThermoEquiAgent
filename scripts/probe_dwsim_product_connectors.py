"""Determine the real semantics of DWSIM's ConnectDistillate / ConnectBottoms.

Builds a small, definitely-converging distillation column in code, then runs it
several times with different product-wiring combinations, so the *observable*
result identifies which connector feeds which draw-off.  Reading connector names
or overloads is not enough: the earlier version of this export called
``ConnectObjects(column, stream, 0|1, 0)``, which returned without raising and
still left both products unwired, so a solve "succeeded" with zero-flow products.

The test system is deliberately trivial (water / 1-butanol, 10 stages, high
reflux) so a correct wiring must produce a water-rich distillate and a
butanol-rich bottoms.

Run with full process access (pythonnet needs OpenProcess).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build(automation, object_type, wiring: str):
    """Create the test column; ``wiring`` selects which outlets get connected."""
    flowsheet = automation.CreateFlowsheet()
    for name in ("Water", "1-butanol"):
        flowsheet.AddCompound(name)

    from thermo_engine.dwsim_export import _add_property_package

    _add_property_package(flowsheet, "UNIQUAC")

    col_wrapper = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col")
    col = col_wrapper.GetAsObject()
    feed_w = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    dist_w = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist")
    bot_w = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot")
    feed, dist, bot = (
        feed_w.GetAsObject(),
        dist_w.GetAsObject(),
        bot_w.GetAsObject(),
    )

    col.NumberOfStages = 10
    col.RefluxRatio = 3.0

    from System import Array, Double

    feed.SetTemperature(365.0)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double]([0.5, 0.5]))
    col.ConnectFeed(feed, 4)

    # Bottoms product flow spec closes the balance; condenser stays on R.
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    spec = col.Specs["R"]
    spec.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    spec.SpecValue = 0.5
    spec.SpecUnit = "mol/s"

    if wiring in ("both", "distillate-first"):
        col.ConnectDistillate(dist)
    if wiring in ("both", "bottoms-first"):
        col.ConnectBottoms(bot)

    return flowsheet, col, dist, bot


def report(automation, flowsheet, dist, bot) -> tuple[float, float]:
    automation.CalculateFlowsheet2(flowsheet)
    rows = []
    for label, stream in (("distillate", dist), ("bottoms", bot)):
        t = float(stream.GetTemperature())
        f = float(stream.GetMolarFlow())
        phases = stream.Phases
        x = (
            [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]
            if phases.Count > 0
            else []
        )
        rows.append((label, t, f, x))
    for label, t, f, x in rows:
        print(f"    {label:11} T={t:7.2f} K  F={f:7.4f} mol/s  x={x}")
    return rows[0][2], rows[1][2]


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    for wiring, label in (
        ("both", "both connectors (distillate then bottoms)"),
        ("distillate-first", "ONLY ConnectDistillate"),
        ("bottoms-first", "ONLY ConnectBottoms"),
        ("none", "neither connected (control)"),
    ):
        print(f"\n=== {label} ===")
        flowsheet, col, dist, bot = build(automation, object_type, wiring)
        try:
            report(automation, flowsheet, dist, bot)
        except Exception as exc:  # noqa: BLE001 - a failure is itself the datum
            print(f"    solve failed: {type(exc).__name__}: {str(exc)[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
