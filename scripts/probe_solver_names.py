"""Find the solver name string that DWSIM actually resolves at solve time.

``SolvingMethodName`` accepts any string silently -- validation happens only when
the column solves, where a wrong name yields "Unable to find column solver with
name '...'".  So candidate names must be tested by *solving* a column, and the
result read from ``CalculateFlowsheet4`` (``CalculateFlowsheet2`` swallows the
error entirely).

The test column is deliberately trivial (water / 1-butanol, 10 stages) so a
correct solver name should converge.

Run with full process access (pythonnet needs OpenProcess).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES = [
    "Wang-Henke Bubble-Point (BP) Solver",
    "Modified Wang-Henke Bubble-Point (MBP) Solver",
    "Sum-Rates (SR) Method",
    "Simultaneous Correction (SC) Method",
    "Simultaneous Correction (SC) Method MESH Equations Calculator",
]

#: Solver names proven to resolve in this build (others fail with "Unable to find
#: column solver"), used for the convergence experiments below.
WORKING_SOLVERS = [
    "Wang-Henke Bubble-Point (BP) Solver",
    "Modified Wang-Henke Bubble-Point (MBP) Solver",
]


def try_solver(
    automation,
    object_type,
    name: str,
    *,
    stages: int = 10,
    reflux: float = 3.0,
    bottoms_flow: float = 0.5,
    max_iterations: int = 1000,
    estimates: bool = False,
) -> tuple[str, str]:
    flowsheet = automation.CreateFlowsheet()
    for compound in ("Water", "1-butanol"):
        flowsheet.AddCompound(compound)

    from thermo_engine.dwsim_export import _add_property_package

    _add_property_package(flowsheet, "UNIQUAC")

    # Imported only now: creating the flowsheet and adding the property package is
    # what loads the DWSIM assemblies into the CLR, so these namespaces do not
    # exist at module import time.
    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()

    col.NumberOfStages = stages
    col.RefluxRatio = reflux
    col.SolvingMethodName = name
    col.MaxIterations = max_iterations

    if estimates:
        for flag in ("UseTemperatureEstimates", "UseLiquidFlowEstimates",
                     "UseVaporFlowEstimates"):
            try:
                setattr(col, flag, True)
            except Exception:  # noqa: BLE001
                pass
        temps = [365.0 + 20.0 * (i / max(stages - 1, 1)) for i in range(stages)]
        col.SetInitialTemperatureEstimates(Array[Double](temps))
        col.SetInitialLiquidMolarFlowEstimates(Array[Double]([reflux] * stages))
        col.SetInitialVaporMolarFlowEstimates(Array[Double]([reflux + 1.0] * stages))

    feed.SetTemperature(365.0)
    feed.SetPressure(101325.0)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(Array[Double]([0.5, 0.5]))
    col.ConnectFeed(feed, 5)

    spec = col.Specs["R"]
    spec.SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    spec.SpecValue = bottoms_flow
    spec.SpecUnit = "mol/s"

    col.ConnectDistillate(dist)
    col.ConnectBottoms(bot)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        return ("ERROR", str(errors[0]).splitlines()[0][:110])
    return ("OK", f"dist={float(dist.GetMolarFlow()):.4f} bot={float(bot.GetMolarFlow()):.4f}")


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, object_type = _automation_factory()
    automation = factory()

    print("=== which solver names resolve in this build? ===")
    for name in CANDIDATES:
        status, detail = try_solver(automation, object_type, name)
        print(f"[{status:5}] {name}")
        print(f"          {detail}")

    print("\n=== can a trivial water/1-butanol column converge at all? ===")
    for name in WORKING_SOLVERS:
        for est in (False, True):
            status, detail = try_solver(
                automation, object_type, name, estimates=est
            )
            print(f"[{status:5}] {name} | estimates={est}")
            print(f"          {detail}")

    print("\n=== easy separation: 5 stages, R=10, 50/50 split ===")
    for name in WORKING_SOLVERS:
        status, detail = try_solver(
            automation, object_type, name, stages=5, reflux=10.0,
            bottoms_flow=0.5, estimates=True,
        )
        print(f"[{status:5}] {name}")
        print(f"          {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
