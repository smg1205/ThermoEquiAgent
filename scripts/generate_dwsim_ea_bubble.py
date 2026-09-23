"""Generate a DWSIM .dwxmz file for ethanol-ethyl acetate bubble-point validation.

Builds a flowsheet with two compounds (Ethanol, Ethyl acetate), an NRTL property
package, one Feed material stream, and one equilibrium flash vessel.  You open
the generated file in the DWSIM GUI, set the feed's liquid composition and
pressure (760 mmHg), and DWSIM returns the bubble behaviour you compare against
the experiment and the ThermoFormer prediction.

Run from the repo root in your NORMAL terminal (not a sandboxed shell), where
pythonnet can load the DWSIM .NET assemblies:

    conda run -n thermo python scripts/generate_dwsim_ea_bubble.py

Preconditions:
  - DWSIM installed at the path in .env (DWSIM_HOME), or the conventional
    Windows location auto-detected by thermo_engine.dwsim_export.
  - pythonnet installed in the active environment.

Output:
  docs/BubbleData_case1_EA_bubble.dwxmz
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

from thermo_engine import dwsim_export as ded  # noqa: E402

DEST = _REPO / "docs" / "BubbleData_case1_EA_bubble.dwxmz"
# A representative feed composition near the azeotrope (x_ethanol = 0.44).
# Change these to other values and re-run to generate a new file.
COMPONENTS = ["ethanol", "ethyl acetate"]
FEED_MOL_FRAC = [0.44, 0.56]
FETCH_FLOW_MOL_S = 1.0
FEED_TEMP_K = 353.15
FEED_PRESSURE_PA = 760.0 * 133.322


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()
    for name in COMPONENTS:
        flowsheet.AddCompound(ded._dwsim_compound_name(name))  # Ethanol, Ethyl acetate
    ded._add_property_package(flowsheet, "NRTL")

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    sep = flowsheet.AddObject(object_type.Vessel, 400, 0, "Equilibrium Flash")
    vapor = flowsheet.AddObject(object_type.MaterialStream, 700, -80, "Vapor Product")
    liquid = flowsheet.AddObject(object_type.MaterialStream, 700, 80, "Liquid Product")

    feed_stream = ded._simulation_object(feed)
    feed_stream.SetTemperature(FEED_TEMP_K)
    feed_stream.SetPressure(FEED_PRESSURE_PA)
    feed_stream.SetMolarFlow(FETCH_FLOW_MOL_S)
    feed_stream.SetOverallComposition(ded._composition_argument(list(FEED_MOL_FRAC)))

    flowsheet.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(sep.GraphicObject, liquid.GraphicObject, 1, 0)

    ded._save_flowsheet(automation, flowsheet, DEST)
    print(f"wrote: {DEST}")
    print(f"components: {COMPONENTS}  feed x_ethanol={FEED_MOL_FRAC[0]}")
    print("property package: NRTL")


if __name__ == "__main__":
    main()
