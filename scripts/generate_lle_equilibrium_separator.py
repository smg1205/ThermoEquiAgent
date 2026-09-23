"""Generate a recalculable DWSIM ternary LLE equilibrium separator.

DWSIM 9.0.5's rigorous Extractor can collapse a valid LLE estimate to the
trivial K=1 solution.  This generator first calculates the UNIQUAC LLE split
with DWSIM's Nested Loops (VLLE) algorithm, then writes the equilibrium
component recoveries into a Component Separator.  The resulting flowsheet can
be recalculated in DWSIM without invoking the failing rigorous-column branch.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402


COMPONENTS = ["Ethanol", "Ethyl acetate", "Water"]
FEED_COMPOSITION = [0.02, 0.49, 0.49]
FEED_FLOW_MOL_S = 2.0
TEMPERATURE_K = 298.15
PRESSURE_PA = 101325.0
OUTPUT = ROOT / "exports" / "test" / "ethanol_eac_water_lle_working.dwxmz"


def main() -> None:
    factory, object_type = ded._automation_factory()

    # Calculate a genuine two-liquid-phase equilibrium split in DWSIM.
    automation = factory()
    flowsheet = automation.CreateFlowsheet()
    for name in COMPONENTS:
        flowsheet.AddCompound(name)
    ded._add_property_package(flowsheet, "UNIQUAC")

    feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Ternary_Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(TEMPERATURE_K)
    feed.SetPressure(PRESSURE_PA)
    feed.SetMolarFlow(FEED_FLOW_MOL_S)
    feed.SetOverallComposition(ded._composition_argument(FEED_COMPOSITION))

    package = next(
        prop.GetValue(feed, None)
        for prop in feed.GetType().GetProperties()
        if prop.Name == "PropertyPackage"
        and "Thermodynamics.PropertyPackages.PropertyPackage" in prop.PropertyType.FullName
    )
    package_type = package.GetType()
    package_type.GetProperty("CurrentMaterialStream").SetValue(package, feed, None)
    flash_type = package_type.Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
    )
    flash = flash_type.GetConstructor([]).Invoke([])
    flash.StabSearchSeverity = 3

    from System import Activator, Array, Double, Enum  # type: ignore[import-not-found]

    result = flash.Flash_PT(
        Array[Double](FEED_COMPOSITION),
        PRESSURE_PA,
        TEMPERATURE_K,
        package,
        False,
        None,
    )
    organic_fraction = float(result[5])
    organic_composition = [float(value) for value in result[6]]
    if not 1.0e-8 < organic_fraction < 1.0 - 1.0e-8:
        raise RuntimeError("UNIQUAC VLLE calculation did not return two liquid phases")

    component_recoveries = [
        100.0 * organic_fraction * organic_composition[i] / FEED_COMPOSITION[i]
        for i in range(len(COMPONENTS))
    ]

    separator_go = flowsheet.AddObject(
        object_type.ComponentSeparator,
        350,
        0,
        "LLE_Equilibrium_Separator",
    )
    organic_go = flowsheet.AddObject(object_type.MaterialStream, 700, -70, "Organic_Phase")
    aqueous_go = flowsheet.AddObject(object_type.MaterialStream, 700, 70, "Aqueous_Phase")
    separator = ded._simulation_object(separator_go)
    separator.SpecifiedStreamIndex = 0

    spec_type = separator.GetType().Assembly.GetType(
        "DWSIM.UnitOperations.UnitOperations.Auxiliary.ComponentSeparationSpec"
    )
    spec_enum_type = separator.GetType().Assembly.GetType(
        "DWSIM.UnitOperations.UnitOperations.Auxiliary.SeparationSpec"
    )
    percent_molar = Enum.Parse(spec_enum_type, "PercentInletMolarFlow")
    for component, recovery in zip(COMPONENTS, component_recoveries, strict=True):
        spec = Activator.CreateInstance(
            spec_type,
            [component, percent_molar, float(recovery), "%"],
        )
        separator.ComponentSepSpecs.Add(component, spec)

    flowsheet.ConnectObjects(feed_go.GraphicObject, separator_go.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(separator_go.GraphicObject, organic_go.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(separator_go.GraphicObject, aqueous_go.GraphicObject, 1, 0)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        raise RuntimeError(str(errors[0]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    automation.SaveFlowsheet2(flowsheet, str(OUTPUT))
    print(OUTPUT)
    print(f"organic phase fraction: {organic_fraction:.12g}")
    print(f"organic phase composition: {organic_composition}")


if __name__ == "__main__":
    main()
