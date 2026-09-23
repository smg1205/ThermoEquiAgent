"""Generate + drive a DWSIM simulation for 2-propanol-water bubble-point validation.

This script
  1. builds a DWSIM flowsheet (2-propanol + water, NRTL property package, one
     Feed -> equilibrium Flash) for a chosen liquid composition,
  2. saves it as a .dwxmz file you can open in the DWSIM GUI.

Because the exact bubble-point solver call and 2-propanol compound key vary by
DWSIM version, the reliable, version-portable path is: generate the file, open
it in the DWSIM GUI, press Calculate, and read the flash / use the Phase
Envelope (Bubble & Dew) tool. The script tries several candidate compound keys
for 2-propanol to minimise AddCompound failures.

Run from the repo root in a NORMAL terminal where pythonnet can load the DWSIM
.NET assemblies (DWSIM's automation_reg.bat must have run once on the machine):

    conda run -n thermo python scripts/generate_dwsim_ipa_water.py

Preconditions (checked by thermo_engine.dwsim_export):
  - DWSIM installed; .env sets DWSIM_HOME=C:\\Users\\34861\\AppData\\Local\\DWSIM
  - pythonnet installed in the active environment

Outputs:
  docs/BubbleData_ipa_water_EA_bubble.dwxmz
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

# 2-propanol's exact DWSIM compound key varies across versions; try candidates.
IPA_CANDIDATES = ["Isopropanol", "2-Propanol", "Propan-2-ol", "Isopropyl alcohol"]
WATER = "Water"

# Feed composition (change x_IPA to probe other points).
FEED_MOL_FRAC = [0.50, 0.50]
FEED_FLOW_MOL_S = 1.0
# Reference bubble temperature at x_IPA=0.5 (~80.2 C experiment / ~81.8 C TF).
# Used as the flash temperature so the file opens near the bubble point.
FEED_TEMP_K = 80.2 + 273.15
# 760 mmHg in Pa for DWSIM's pressure call.
FEED_PRESSURE_PA = 760.0 * 133.322


def main() -> None:
    from thermo_engine import dwsim_export as ded

    destination = _REPO / "docs" / "BubbleData_ipa_water_EA_bubble.dwxmz"
    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()

    # Add compounds; accept any candidate key that DWSIM resolves.
    added = []
    try:
        flowsheet.AddCompound(ded._dwsim_compound_name(WATER))
        added.append(ded._dwsim_compound_name(WATER))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not add water compound: {exc}")
    ipa_added = None
    for cand in IPA_CANDIDATES:
        try:
            flowsheet.AddCompound(cand)
            ipa_added = cand
            break
        except Exception:
            continue
    if ipa_added is None:
        raise RuntimeError(
            "Could not add the 2-propanol compound to DWSIM. Try editing "
            "IPA_CANDIDATES in this script with the exact name DWSIM uses."
        )
    added.append(ipa_added)
    print(f"added compounds: {added}")

    ded._add_property_package(flowsheet, "NRTL")

    feed = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    sep = flowsheet.AddObject(object_type.Vessel, 450, 0, "Equilibrium Flash")
    vapor = flowsheet.AddObject(object_type.MaterialStream, 750, -80, "Vapor Product")
    liquid = flowsheet.AddObject(object_type.MaterialStream, 750, 80, "Liquid Product")

    feed_stream = ded._simulation_object(feed)
    feed_stream.SetTemperature(FEED_TEMP_K)
    feed_stream.SetPressure(FEED_PRESSURE_PA)
    feed_stream.SetMolarFlow(FEED_FLOW_MOL_S)
    feed_stream.SetOverallComposition(ded._composition_argument(list(FEED_MOL_FRAC)))

    flowsheet.ConnectObjects(feed.GraphicObject, sep.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(sep.GraphicObject, vapor.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(sep.GraphicObject, liquid.GraphicObject, 1, 0)

    ded._save_flowsheet(automation, flowsheet, destination)
    print(f"wrote: {destination}")
    print(f"feed x_IPA={FEED_MOL_FRAC[0]}  T_flash={FEED_TEMP_K:.2f} K  P={FEED_PRESSURE_PA:.1f} Pa")
    print("property package: NRTL")
    print("\nNext: open the .dwxmz in DWSIM GUI, press Calculate, read Vapor/Liquid.")
    print("For the bubble point, set T so vapor fraction -> 0 (or use Phase Envelope).")


if __name__ == "__main__":
    main()
