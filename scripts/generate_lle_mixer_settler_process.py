"""Generate a recalculable ethanol/ethyl-acetate/water LLE process in DWSIM.

The flowsheet contains two live feeds and a single-stage liquid-liquid
mixer-settler. Its embedded DWSIM IronPython calculation invokes DWSIM's own
UNIQUAC Nested Loops (VLLE) flash on every solve. No product composition,
component recovery, or phase fraction is fixed in advance.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402


COMPONENTS = ["Ethanol", "Ethyl acetate", "Water"]
TEMPERATURE_K = 298.15
PRESSURE_PA = 101325.0

# The combined composition is [0.02, 0.49, 0.49], inside the two-liquid region
# predicted by the installed DWSIM 9.0.5 UNIQUAC data.
ORGANIC_FLOW_MOL_S = 1.02
ORGANIC_COMPOSITION = [0.04 / 1.02, 0.98 / 1.02, 0.0]
WATER_FLOW_MOL_S = 0.98
WATER_COMPOSITION = [0.0, 0.0, 1.0]

OUTPUT = ROOT / "exports" / "test" / "ethanol_eac_water_mixer_settler_process.dwxmz"


MIXER_SETTLER_SCRIPT = r'''from System import Activator, Array, Double

if ims1 is None or ims2 is None or oms1 is None or oms2 is None:
    raise Exception("The mixer-settler requires two inlets and two material outlets.")

feed1_flow = float(ims1.GetMolarFlow())
feed2_flow = float(ims2.GetMolarFlow())
feed_flow = feed1_flow + feed2_flow
if feed_flow <= 0.0:
    raise Exception("The combined mixer-settler inlet flow must be positive.")

feed1_z = [float(value) for value in ims1.GetOverallComposition()]
feed2_z = [float(value) for value in ims2.GetOverallComposition()]
feed_z = Array[Double]([
    (feed1_flow * feed1_z[index] + feed2_flow * feed2_z[index]) / feed_flow
    for index in range(len(feed1_z))
])
temperature = (
    feed1_flow * float(ims1.GetTemperature())
    + feed2_flow * float(ims2.GetTemperature())
) / feed_flow
pressure = min(float(ims1.GetPressure()), float(ims2.GetPressure()))

pp = Me.PropertyPackage
if pp is None:
    pp = ims1.PropertyPackage
pp.CurrentMaterialStream = ims1

flash_type = pp.GetType().Assembly.GetType(
    "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
)
flash = Activator.CreateInstance(flash_type)
flash.StabSearchSeverity = 3
result = flash.Flash_PT(feed_z, pressure, temperature, pp, False, None)

phase1_fraction = float(result[0])
phase1_x = [float(value) for value in result[2]]
phase2_fraction = float(result[5])
phase2_x = [float(value) for value in result[6]]
if phase1_fraction <= 1.0e-10 or phase2_fraction <= 1.0e-10:
    raise Exception(
        "UNIQUAC/VLLE found only one liquid phase at the current feed condition."
    )

# Always place the ethyl-acetate-richer phase in oms1.
if phase1_x[1] >= phase2_x[1]:
    organic_fraction = phase1_fraction
    organic_x = phase1_x
    aqueous_fraction = phase2_fraction
    aqueous_x = phase2_x
else:
    organic_fraction = phase2_fraction
    organic_x = phase2_x
    aqueous_fraction = phase1_fraction
    aqueous_x = phase1_x

oms1.SetTemperature(temperature)
oms1.SetPressure(pressure)
oms1.SetMolarFlow(feed_flow * organic_fraction)
oms1.SetOverallComposition(Array[Double](organic_x))

oms2.SetTemperature(temperature)
oms2.SetPressure(pressure)
oms2.SetMolarFlow(feed_flow * aqueous_fraction)
oms2.SetOverallComposition(Array[Double](aqueous_x))

organic_phase_fraction = organic_fraction
aqueous_phase_fraction = aqueous_fraction
'''


def _stream_values(stream):
    stream_type = stream.GetType()

    def invoke(name: str):
        return stream_type.GetMethod(name).Invoke(stream, None)

    return {
        "flow_mol_s": float(invoke("GetMolarFlow")),
        "temperature_K": float(invoke("GetTemperature")),
        "pressure_Pa": float(invoke("GetPressure")),
        "z": [float(value) for value in invoke("GetOverallComposition")],
    }


def _calculate(automation, flowsheet) -> None:
    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        raise RuntimeError("\n".join(str(errors[index]) for index in range(errors.Count)))


def _find_by_tag(flowsheet, tag: str):
    for item in flowsheet.SimulationObjects.Values:
        if str(item.GraphicObject.Tag) == tag:
            return item
    raise KeyError(tag)


def _set_molar_flow(stream, value: float) -> None:
    from System import Array, Double, Object  # type: ignore[import-not-found]

    method = next(
        candidate
        for candidate in stream.GetType().GetMethods()
        if candidate.Name == "SetMolarFlow"
        and candidate.GetParameters().Length == 1
        and candidate.GetParameters()[0].ParameterType == Double
    )
    method.Invoke(stream, Array[Object]([Double(value)]))


def _assert_lle_results(organic, aqueous, expected_total_flow: float) -> None:
    organic_values = _stream_values(organic)
    aqueous_values = _stream_values(aqueous)
    if organic_values["flow_mol_s"] <= 1.0e-8:
        raise RuntimeError("DWSIM did not produce an organic liquid phase")
    if aqueous_values["flow_mol_s"] <= 1.0e-8:
        raise RuntimeError("DWSIM did not produce an aqueous liquid phase")
    if organic_values["z"][1] <= aqueous_values["z"][1]:
        raise RuntimeError("the organic outlet is not enriched in ethyl acetate")
    if aqueous_values["z"][2] <= organic_values["z"][2]:
        raise RuntimeError("the aqueous outlet is not enriched in water")
    total_out = organic_values["flow_mol_s"] + aqueous_values["flow_mol_s"]
    if abs(total_out - expected_total_flow) > 1.0e-8:
        raise RuntimeError("the mixer-settler total molar balance did not close")


def _assert_component_balance(feed1, feed2, organic, aqueous) -> None:
    inlet1 = _stream_values(feed1)
    inlet2 = _stream_values(feed2)
    outlet1 = _stream_values(organic)
    outlet2 = _stream_values(aqueous)
    for index, component in enumerate(COMPONENTS):
        inlet = (
            inlet1["flow_mol_s"] * inlet1["z"][index]
            + inlet2["flow_mol_s"] * inlet2["z"][index]
        )
        outlet = (
            outlet1["flow_mol_s"] * outlet1["z"][index]
            + outlet2["flow_mol_s"] * outlet2["z"][index]
        )
        if abs(inlet - outlet) > 1.0e-8:
            raise RuntimeError(f"component molar balance failed for {component}")


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()

    for component in COMPONENTS:
        flowsheet.AddCompound(component)
    ded._add_property_package(flowsheet, "UNIQUAC")

    organic_go = flowsheet.AddObject(
        object_type.MaterialStream, 0, -70, "Organic_Feed"
    )
    water_go = flowsheet.AddObject(object_type.MaterialStream, 0, 70, "Water_Feed")
    settler_go = flowsheet.AddObject(
        object_type.CustomUO, 350, 0, "LLE_Mixer_Settler_UNIQUAC_VLLE"
    )
    organic_product_go = flowsheet.AddObject(
        object_type.MaterialStream, 700, -60, "Organic_Phase"
    )
    aqueous_product_go = flowsheet.AddObject(
        object_type.MaterialStream, 700, 60, "Aqueous_Phase"
    )

    organic_feed = ded._simulation_object(organic_go)
    organic_feed.SetTemperature(TEMPERATURE_K)
    organic_feed.SetPressure(PRESSURE_PA)
    organic_feed.SetMolarFlow(ORGANIC_FLOW_MOL_S)
    organic_feed.SetOverallComposition(ded._composition_argument(ORGANIC_COMPOSITION))

    water_feed = ded._simulation_object(water_go)
    water_feed.SetTemperature(TEMPERATURE_K)
    water_feed.SetPressure(PRESSURE_PA)
    water_feed.SetMolarFlow(WATER_FLOW_MOL_S)
    water_feed.SetOverallComposition(ded._composition_argument(WATER_COMPOSITION))

    settler = ded._simulation_object(settler_go)
    settler.ScriptText = MIXER_SETTLER_SCRIPT
    settler.ComponentDescription = (
        "Single-stage LLE mixer-settler using DWSIM UNIQUAC and Nested Loops (VLLE)."
    )

    flowsheet.ConnectObjects(organic_go.GraphicObject, settler_go.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(water_go.GraphicObject, settler_go.GraphicObject, 0, 1)
    flowsheet.ConnectObjects(
        settler_go.GraphicObject, organic_product_go.GraphicObject, 0, 0
    )
    flowsheet.ConnectObjects(
        settler_go.GraphicObject, aqueous_product_go.GraphicObject, 1, 0
    )

    _calculate(automation, flowsheet)
    organic_product = ded._simulation_object(organic_product_go)
    aqueous_product = ded._simulation_object(aqueous_product_go)
    total_feed = ORGANIC_FLOW_MOL_S + WATER_FLOW_MOL_S
    _assert_lle_results(organic_product, aqueous_product, total_feed)
    _assert_component_balance(
        organic_feed, water_feed, organic_product, aqueous_product
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, flowsheet, OUTPUT)

    # Reload and solve again to verify script and ports survive serialization.
    reloaded = automation.LoadFlowsheet2(str(OUTPUT))
    _calculate(automation, reloaded)
    r_organic_feed = _find_by_tag(reloaded, "Organic_Feed")
    r_water_feed = _find_by_tag(reloaded, "Water_Feed")
    r_organic = _find_by_tag(reloaded, "Organic_Phase")
    r_aqueous = _find_by_tag(reloaded, "Aqueous_Phase")
    _assert_lle_results(r_organic, r_aqueous, total_feed)
    _assert_component_balance(r_organic_feed, r_water_feed, r_organic, r_aqueous)

    # Prove that this is a live process model rather than a saved fixed split.
    baseline_organic_flow = _stream_values(r_organic)["flow_mol_s"]
    perturbed_water_flow = 1.08
    _set_molar_flow(r_water_feed, perturbed_water_flow)
    _calculate(automation, reloaded)
    _assert_lle_results(
        r_organic,
        r_aqueous,
        ORGANIC_FLOW_MOL_S + perturbed_water_flow,
    )
    _assert_component_balance(r_organic_feed, r_water_feed, r_organic, r_aqueous)
    perturbed_organic_flow = _stream_values(r_organic)["flow_mol_s"]
    if abs(perturbed_organic_flow - baseline_organic_flow) <= 1.0e-8:
        raise RuntimeError("outlet phase flow did not respond to a feed-flow change")

    _set_molar_flow(r_water_feed, WATER_FLOW_MOL_S)
    _calculate(automation, reloaded)
    _assert_lle_results(r_organic, r_aqueous, total_feed)
    _assert_component_balance(r_organic_feed, r_water_feed, r_organic, r_aqueous)

    print(OUTPUT)
    print(
        "recalculation check (organic flow, mol/s):",
        baseline_organic_flow,
        "->",
        perturbed_organic_flow,
        "-> restored",
        _stream_values(r_organic)["flow_mol_s"],
    )
    print("organic feed:", _stream_values(r_organic_feed))
    print("water feed:", _stream_values(r_water_feed))
    print("organic:", _stream_values(r_organic))
    print("aqueous:", _stream_values(r_aqueous))


if __name__ == "__main__":
    main()
