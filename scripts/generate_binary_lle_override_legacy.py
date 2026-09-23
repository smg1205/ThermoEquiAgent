"""Try to make the Gibbs Vessel LLE stable on recalc by seeding outlet streams.

The PH-flash failure comes from the outlet streams (SpecType = Pressure_and_Enthalpy,
hardcoded by Vessel.vb) running an unstable Gibbs Flash_PH on recalc.  We seed each
outlet with a concrete temperature + enthalpy so the PH solver starts near the answer.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_gibbs.dwxmz"
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def _find(fs, tag):
    for item in fs.SimulationObjects.Values:
        if str(item.GraphicObject.Tag) == tag:
            return item
    raise KeyError(tag)


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()

    # rebuild fresh Gibbs flowsheet (not load) so we control everything
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    from System import Enum
    from DWSIM.Interfaces.Enums import FlashSetting
    pp = list(fs.PropertyPackages.Values)[0]
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)

    import System as S2
    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(298.15); feed.SetPressure(101325.0); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument([0.4, 0.6]))

    vessel = ded._simulation_object(vessel_go)
    # force Legacy (TP-flash) explicitly via OverrideT/OverrideP
    try:
        vessel.OverrideT = True
        vessel.OverrideP = True
        vessel.FlashTemperature = 298.15
        vessel.FlashPressure = 101325.0
        print("[OK] OverrideT/OverrideP -> Legacy TP")
    except Exception as e:
        print("[WARN] override:", type(e).__name__)

    for f, t, fi, ti in ((feed_go, vessel_go, 0, 0), (vessel_go, vapor_go, 0, 0),
                         (vessel_go, light_go, 1, 0), (vessel_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass

    def flow(s):
        return float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))
    def z(s):
        return [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)]

    def calc(label):
        e = automation.CalculateFlowsheet4(fs)
        if e and e.Count:
            print(f"  [{label}] ERRORS:", str(e[0])[:140])
        else:
            print(f"  [{label}] OK  Light=%.5f (MIBK %.4f) Heavy=%.5f (MIBK %.4f)" %
                  (flow(ded._simulation_object(light_go)), z(ded._simulation_object(light_go))[0],
                   flow(ded._simulation_object(heavy_go)), z(ded._simulation_object(heavy_go))[0]))

    print("first calc:"); calc("first")
    print("recalc 1:"); calc("recalc1")
    print("recalc 2:"); calc("recalc2")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
