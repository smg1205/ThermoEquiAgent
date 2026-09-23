"""Binary MIBK / water LLE: force the stream layer to show TWO liquid phases.

DWSIM 9's MaterialStream has read-only Liquid1/Liquid2 (write=False) and no
standalone Decanter object; the correct way to make the GUI show two liquid
phases is the single-stage mixer-settler route: a CustomUO whose IronPython
script runs the Nested Loops (VLLE) flash and writes the two equilibrium phases
to two separate outlet material streams (Organic_Phase / Aqueous_Phase).

This produces, in the DWSIM GUI, two live streams whose compositions are the
two coexistence phases -- the same visual evidence as the "Liquid 1 / Liquid 2"
Fraction row, but expressed as two outlet streams instead of one stream's phase
dictionary.

All phase numbers come from DWSIM's UNIQUAC + Nested Loops (VLLE) flash.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine import dwsim_export as ded  # noqa: E402

COMPONENTS = ["Methyl isobutyl ketone", "Water"]
TEMPERATURE_K = 298.15
PRESSURE_PA = 101325.0
FEED_COMPOSITION = [0.4, 0.6]  # x_MIBK, x_water -- inside two-phase region
OUTPUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_binary_LLE_decanter.dwxmz"

# IronPython script run inside the CustomUO on every solve.
DECANTER_SCRIPT = r'''from System import Activator, Array, Double, Object

if ims1 is None or oms1 is None or oms2 is None:
    raise Exception("Decanter needs one inlet and two outlets.")

feed_flow = float(ims1.GetMolarFlow())
if feed_flow <= 0.0:
    raise Exception("Feed flow must be positive.")
z = [float(v) for v in ims1.GetOverallComposition()]
temperature = float(ims1.GetTemperature())
pressure = float(ims1.GetPressure())

pp = Me.PropertyPackage
if pp is None:
    pp = ims1.PropertyPackage
pp.CurrentMaterialStream = ims1

flash_type = pp.GetType().Assembly.GetType(
    "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
)
flash = Activator.CreateInstance(flash_type)
flash.StabSearchSeverity = 3
result = flash.Flash_PT(Array[Double](z), pressure, temperature, pp, False, None)

phase1_frac = float(result[0]); phase1_x = [float(v) for v in result[2]]
phase2_frac = float(result[5]); phase2_x = [float(v) for v in result[6]]
if phase1_frac <= 1.0e-10 or phase2_frac <= 1.0e-10:
    raise Exception("Single liquid phase (no LLE split) at this feed condition.")

# component order: [MIBK, Water]; organic = MIBK-richer phase (index 0 larger)
if phase1_x[0] >= phase2_x[0]:
    organic_frac, organic_x = phase1_frac, phase1_x
    aqueous_frac, aqueous_x = phase2_frac, phase2_x
else:
    organic_frac, organic_x = phase2_frac, phase2_x
    aqueous_frac, aqueous_x = phase1_frac, phase1_x

oms1.SetTemperature(temperature)
oms1.SetPressure(pressure)
oms1.SetMolarFlow(feed_flow * organic_frac)
oms1.SetOverallComposition(Array[Double](organic_x))

oms2.SetTemperature(temperature)
oms2.SetPressure(pressure)
oms2.SetMolarFlow(feed_flow * aqueous_frac)
oms2.SetOverallComposition(Array[Double](aqueous_x))
'''


def _stream_values(stream):
    stream_type = stream.GetType()

    def invoke(name: str):
        return stream_type.GetMethod(name).Invoke(stream, None)

    return {
        "flow_mol_s": float(invoke("GetMolarFlow")),
        "z": [float(v) for v in invoke("GetOverallComposition")],
    }


def main() -> None:
    factory, object_type = ded._automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()

    try:
        flowsheet.AddCompound(COMPONENTS[0])
    except Exception:
        flowsheet.AddCompound("4-Methyl-2-pentanone")
    flowsheet.AddCompound(COMPONENTS[1])
    ded._add_property_package(flowsheet, "UNIQUAC")

    feed_go = flowsheet.AddObject(object_type.MaterialStream, 0, 0, "Feed_MIBK_Water")
    decanter_go = flowsheet.AddObject(object_type.CustomUO, 350, 0, "LLE_Decanter")
    organic_go = flowsheet.AddObject(object_type.MaterialStream, 700, -60, "Organic_Phase")
    aqueous_go = flowsheet.AddObject(object_type.MaterialStream, 700, 60, "Aqueous_Phase")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(TEMPERATURE_K)
    feed.SetPressure(PRESSURE_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_COMPOSITION))

    decanter = ded._simulation_object(decanter_go)
    decanter.ScriptText = DECANTER_SCRIPT
    decanter.ComponentDescription = "Single-stage liquid-liquid decanter (UNIQUAC + Nested Loops VLLE)."

    flowsheet.ConnectObjects(feed_go.GraphicObject, decanter_go.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(decanter_go.GraphicObject, organic_go.GraphicObject, 0, 0)
    flowsheet.ConnectObjects(decanter_go.GraphicObject, aqueous_go.GraphicObject, 1, 0)

    errors = automation.CalculateFlowsheet4(flowsheet)
    if errors.Count:
        raise RuntimeError("\n".join(str(errors[i]) for i in range(errors.Count)))

    organic = ded._simulation_object(organic_go)
    aqueous = ded._simulation_object(aqueous_go)
    ov = _stream_values(organic)
    av = _stream_values(aqueous)

    print("================ binary MIBK/water LLE decanter ================")
    print(f"feed overall: x_MIBK={FEED_COMPOSITION[0]}, T={TEMPERATURE_K} K, P=1 atm")
    print(f"Organic_Phase : flow={ov['flow_mol_s']:.5f} mol/s  x=[MIBK {ov['z'][0]:.5f}, water {ov['z'][1]:.5f}]")
    print(f"Aqueous_Phase : flow={av['flow_mol_s']:.5f} mol/s  x=[MIBK {av['z'][0]:.5f}, water {av['z'][1]:.5f}]")
    print("--------------------------------------------------------------")
    print("The GUI now shows TWO outlet streams carrying the two equilibrium")
    print("liquid phases (organic MIBK-rich vs aqueous water-rich).")

    if ov["flow_mol_s"] <= 1e-8 or av["flow_mol_s"] <= 1e-8:
        raise RuntimeError("one phase flow is ~0: no LLE split produced")
    if ov["z"][0] <= av["z"][0]:
        raise RuntimeError("organic outlet not enriched in MIBK")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, flowsheet, OUTPUT)
    print(f"\n[OK] saved: {OUTPUT}")


if __name__ == "__main__":
    main()
