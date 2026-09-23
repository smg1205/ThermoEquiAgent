"""Last attempt: NestedLoops + ImmiscibleWater + ForceNewFlashAlgorithmInstance.

UniversalFlash internally routes to VLLE/immiscible kernels when water immiscibility
is flagged; forcing a fresh algorithm instance after setting ImmiscibleWaterOption
may make the default (NestedLoops -> UniversalFlash) path do the LLE split WITHOUT
the unstable Gibbs PH flash.  If this works, recalc stays stable because NestedLoops
uses TP (Flash_PT), not PH.
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
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
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

    # keep NestedLoops (stable TP), but flag immiscible water + force fresh algo
    try:
        ap = pp.GetType().GetProperty("FlashCalculationApproach")
        ap.SetValue(pp, Enum.Parse(ap.PropertyType, "NestedLoops"), None)
        fr = pp.GetType().GetProperty("ForceNewFlashAlgorithmInstance")
        fr.SetValue(pp, True, None)
        fsp = pp.GetType().GetProperty("FlashSettings")
        settings = fsp.GetValue(pp, None)
        settings[FlashSetting.ImmiscibleWaterOption] = "True"
        settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, settings, None)
        print("[OK] NestedLoops + ImmiscibleWater + ForceNewFlash alg=True")
    except Exception as e:
        print("[WARN] config:", type(e).__name__, e)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    vessel = ded._simulation_object(vessel_go)
    try:
        vessel.FlashTemperature = T; vessel.FlashPressure = P
    except Exception:
        pass

    for f, t2, fi, ti in ((feed_go, vessel_go, 0, 0), (vessel_go, vapor_go, 0, 0),
                          (vessel_go, light_go, 1, 0), (vessel_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t2.GraphicObject, fi, ti)
        except Exception:
            pass

    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("solve errors:")
        for i in range(errors.Count):
            print("  ", str(errors[i])[:150])

    print("\n===== result (NestedLoops + ImmiscibleWater, fresh algo) =====")
    print("Vapor      : %.5f" % _flow(ded._simulation_object(vapor_go)))
    l = ded._simulation_object(light_go); h = ded._simulation_object(heavy_go)
    print("Light      : %.5f  [MIBK %.5f water %.5f]" % (_flow(l), _z(l)[0], _z(l)[1]))
    print("Heavy      : %.5f  [MIBK %.5f water %.5f]" % (_flow(h), _z(h)[0], _z(h)[1]))

    # recalc test
    print("\n-- recalc test --")
    e2 = automation.CalculateFlowsheet4(fs)
    if e2 and e2.Count:
        print("  recalc errors:", str(e2[0])[:150])
    else:
        print("  recalc OK; Light=%.5f Heavy=%.5f" % (_flow(l), _flow(h)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
