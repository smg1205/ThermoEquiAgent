"""Final: set the property-package UniversalFlash.AlgoType to the immiscible-VLLE
algorithm, then run the NATIVE Vessel with Light/Heavy liquid outlets.

This is the actual mechanism DWSIM uses to switch flash algorithms: the package's
FlashBase is a UniversalFlash whose AlgoType (FlashMethod enum) selects the real
sub-algorithm.  Default is `Universal` (auto, VLE-only for this system), which is
why every earlier attempt returned a single liquid phase.  We point AlgoType at
`Nested_Loops_Immiscible_VLLE` and let the native Vessel do the split.
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


def _flow(s):
    return float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))


def _z(s):
    return [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)]


def main():
    factory, object_type = ded._automation_factory()
    import System  # noqa: E402
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashMethod, FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")
    pp = list(fs.PropertyPackages.Values)[0]

    # 1) switch the package's flash sub-algorithm to immiscible-VLLE
    fb = pp.GetType().GetProperty("FlashBase").GetValue(pp, None)
    try:
        fb.AlgoType = Enum.Parse(FlashMethod, "Nested_Loops_Immiscible_VLLE")
        print("[OK] FlashBase.AlgoType =", fb.AlgoType)
    except Exception as e:
        print("[WARN] AlgoType set failed:", type(e).__name__, e)

    # 2) also enable immiscible-water + 3-phase + phase-id switches
    try:
        fsp = pp.GetType().GetProperty("FlashSettings")
        settings = fsp.GetValue(pp, None)
        settings[FlashSetting.ImmiscibleWaterOption] = "True"
        settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, settings, None)
        print("[OK] PP FlashSettings switches enabled")
    except Exception as e:
        print("[WARN] FlashSettings:", type(e).__name__, e)

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
    except Exception as e:
        print("[WARN] vessel T/P:", type(e).__name__)

    for from_go, to_go, fidx, tidx, label in (
        (feed_go, vessel_go, 0, 0, "feed"),
        (vessel_go, vapor_go, 0, 0, "vapor"),
        (vessel_go, light_go, 1, 0, "light"),
        (vessel_go, heavy_go, 2, 0, "heavy"),
    ):
        try:
            fs.ConnectObjects(from_go.GraphicObject, to_go.GraphicObject, fidx, tidx)
        except Exception as e:
            print("[WARN] conn", label, type(e).__name__)

    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("solve errors:")
        for i in range(errors.Count):
            print("  ", str(errors[i])[:150])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    vapor = ded._simulation_object(vapor_go)
    print("\n===== native Vessel (AlgoType=Immiscible_VLLE) result =====")
    print("Vapor       : flow=%.5f" % _flow(vapor))
    print("Light_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(light), _z(light)[0], _z(light)[1]))
    print("Heavy_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(heavy), _z(heavy)[0], _z(heavy)[1]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
