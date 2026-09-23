"""MIBK/water LLE via UNIFAC-LL (group-contribution, NO binary parameters).

UNIFAC-LL predicts activity coefficients from molecular group contributions, so
it needs NO regressed binary interaction parameters -- exactly what we want to
avoid fabricating.  Run the native Vessel with UNIFAC-LL + GibbsMinimization and
check whether the two-phase compositions are physically correct (organic x_water
small, aqueous x_MIBK small) at 298.15 K.
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
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_unifacll_lle.dwxmz"


def _flow(s):
    return float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))
def _z(s):
    return [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)]


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

    # try UNIFAC-LL variants (no binary params needed)
    pp_added = None
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    for pkg in ("UNIFAC-LL", "UNIFAC LL", "UNIFACLL", "UNIFAC (Modified)", "UNIFAC"):
        try:
            ded._add_property_package(fs, pkg)
            pp_added = pkg
            print(f"[OK] property package: {pkg}")
            break
        except Exception as e:
            print(f"[WARN] {pkg}: {type(e).__name__}")

    if pp_added is None:
        print("[FATAL] no UNIFAC-LL package could be added")
        return

    pp = list(fs.PropertyPackages.Values)[0]
    # GibbsMinimization + immiscible water (for reliable two-liquid flash)
    try:
        ap = pp.GetType().GetProperty("FlashCalculationApproach")
        ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
        fsp = pp.GetType().GetProperty("FlashSettings")
        s = fsp.GetValue(pp, None)
        s[FlashSetting.ImmiscibleWaterOption] = "True"
        s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, s, None)
        print("[OK] GibbsMinimization + immiscible-water on")
    except Exception as e:
        print("[WARN] flash config:", type(e).__name__, e)

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
    for f, t, fi, ti in ((feed_go, vessel_go, 0, 0), (vessel_go, vapor_go, 0, 0),
                         (vessel_go, light_go, 1, 0), (vessel_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:", str(e[0])[:160])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    print("\n===== UNIFAC-LL MIBK/water @ 298.15 K =====")
    print(f"Light_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(light), _z(light)[0], _z(light)[1]))
    print(f"Heavy_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(heavy), _z(heavy)[0], _z(heavy)[1]))
    print("[physical target] organic x_water ~ 0.02, aqueous x_MIBK ~ 0.002")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
