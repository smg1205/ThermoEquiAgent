"""Minimal DWSIM rigorous-column baseline, solved via CalculateFlowsheet4.

Establishes a known-good reference for the §1.5 case-2 diagnostics.

Two things this proves, both of which earlier probing got wrong:

1. ``CalculateFlowsheet2`` does NOT report failures -- it returns silently even
   when the column fails its mass balance.  Only ``CalculateFlowsheet4`` returns
   the error list, so every earlier "solved OK" reading was unreliable.
2. ``MaterialStreams`` is the authoritative registry of a column's attached
   streams (feed + products); it grows as ConnectFeed/ConnectDistillate/
   ConnectBottoms are called.

System: water / 1-butanol, 12 stages, R = 3, trivial enough that any working
setup must converge.

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

    # Created first so the CLR has loaded the DWSIM assemblies before the
    # DWSIM.* namespaces are imported below.
    flowsheet = automation.CreateFlowsheet()
    for name in ("Water", "1-butanol"):
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "UNIQUAC")

    from System import Array, Double
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec

    col = flowsheet.AddObject(object_type.DistillationColumn, 250, 0, "Col").GetAsObject()
    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed").GetAsObject()
    dist = flowsheet.AddObject(object_type.MaterialStream, 500, -80, "Dist").GetAsObject()
    bot = flowsheet.AddObject(object_type.MaterialStream, 500, 80, "Bot").GetAsObject()

    stages, reflux, bottoms_flow = 12, 3.0, 0.5
    col.NumberOfStages = stages
    col.RefluxRatio = reflux

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

    print(f"stages={stages} reflux={reflux} bottoms_spec={bottoms_flow}")
    print(f"registered streams on column: {col.MaterialStreams.Count}")

    print("\n-- CalculateFlowsheet2 (silent) --")
    try:
        automation.CalculateFlowsheet2(flowsheet)
        print("   returned without raising")
    except Exception as exc:  # noqa: BLE001
        print(f"   raised {type(exc).__name__}")

    print("\n-- CalculateFlowsheet4 (returns error list) --")
    errors = automation.CalculateFlowsheet4(flowsheet)
    print(f"   error count: {errors.Count}")
    for i in range(errors.Count):
        print(f"     [{i}] {str(errors[i])[:200]}")

    print(f"\n{'stream':8} {'T(K)':>9} {'F(mol/s)':>10}  composition")
    print("-" * 56)
    for label, stream in (("feed", feed), ("dist", dist), ("bot", bot)):
        t = float(stream.GetTemperature())
        f = float(stream.GetMolarFlow())
        phases = stream.Phases
        x = (
            [round(float(c.MoleFraction), 4) for c in phases[0].Compounds.Values]
            if phases.Count > 0
            else []
        )
        print(f"{label:8} {t:9.2f} {f:10.4f}  {x}")

    return 0 if errors.Count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
