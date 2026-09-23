"""Native-Vessel LLE for MIBK/water: connect ONLY via ConnectObjects (graphic layer).

The earlier failure came from mixing two connection APIs (ConnectFeed/Product
MaterialStream + ConnectObjects).  DWSIM's graphic ConnectObjects performs the
logical connection automatically, so we connect only at the graphic layer, using
the confirmed connector indices:
    vessel output[0]=Vapor, [1]=Light Liquid, [2]=Heavy Liquid
    vessel input[0]=Inlet Stream #0
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def _values(stream):
    t = stream.GetType()
    return {
        "flow": float(t.GetMethod("GetMolarFlow").Invoke(stream, None)),
        "z": [float(v) for v in t.GetMethod("GetOverallComposition").Invoke(stream, None)],
    }


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()

    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))

    vessel = ded._simulation_object(vessel_go)
    try:
        vessel.FlashTemperature = T
        vessel.FlashPressure = P
        # PreferredFlashAlgorithmTag expects the FlashMethod enum NAME (underscore
        # form), not the GUI's spaced label "Nested Loops (VLLE)".
        vessel.PreferredFlashAlgorithmTag = "Nested_Loops_VLLE"
        print("[OK] vessel: VLLE flash @", T, "K")
    except Exception as e:
        print("[WARN] vessel config:", type(e).__name__, e)

    # Turn ON the immiscible-water / three-phase / phase-identification switches
    # on the active property package so the Vessel's own flash actually searches
    # for a second liquid phase (these are all 'off' by default).
    try:
        from DWSIM.Interfaces.Enums import FlashSetting
        sp = vessel.GetType().GetProperty("PropertyPackage")
        pp = sp.GetValue(vessel, None) or list(fs.PropertyPackages.Values)[0]
        fsp = pp.GetType().GetProperty("FlashSettings")
        settings = fsp.GetValue(pp, None)
        settings[FlashSetting.ImmiscibleWaterOption] = "True"
        settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, settings, None)
        print("[OK] PP switches: ImmiscibleWater=True, StabSeverity=3, PhaseIdent=True")
    except Exception as e:
        print("[WARN] PP switch config failed:", type(e).__name__, e)

    # connect ONLY at the graphic layer (this also performs logical connection)
    # feed.OutputConnectors[0] -> vessel.InputConnectors[0]
    # vessel.OutputConnectors[1] -> light.InputConnectors[0]
    # vessel.OutputConnectors[2] -> heavy.InputConnectors[0]
    conns = (
        (feed_go.GraphicObject, vessel_go.GraphicObject, 0, 0, "feed -> vessel inlet0"),
        (vessel_go.GraphicObject, vapor_go.GraphicObject, 0, 0, "vessel vapor -> Vapor"),
        (vessel_go.GraphicObject, light_go.GraphicObject, 1, 0, "vessel light -> Light_Liquid"),
        (vessel_go.GraphicObject, heavy_go.GraphicObject, 2, 0, "vessel heavy -> Heavy_Liquid"),
    )
    for from_go, to_go, fidx, tidx, label in conns:
        try:
            fs.ConnectObjects(from_go, to_go, fidx, tidx)
            print("[OK]", label)
        except Exception as e:
            print("[ERR]", label, type(e).__name__, str(e)[:120])

    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("solve errors:")
        for i in range(errors.Count):
            print("  ", str(errors[i])[:200])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    vapor = ded._simulation_object(vapor_go)
    lv = _values(light)
    hv = _values(heavy)
    vv = _values(vapor)
    print("\n===== native Vessel LLE result =====")
    print("Vapor          : flow=%.5f" % vv["flow"])
    print("Light_Liquid (organic): flow=%.5f  z=[MIBK %.5f, water %.5f]" % (lv["flow"], lv["z"][0], lv["z"][1]))
    print("Heavy_Liquid (aqueous): flow=%.5f  z=[MIBK %.5f, water %.5f]" % (hv["flow"], hv["z"][0], hv["z"][1]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
