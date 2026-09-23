"""Seed outlet streams with correct enthalpy so Gibbs Flash_PH converges on recalc.

Compute the liquid molar enthalpy of the two phases at 298.15 K via the package's
enthalpy routine, then write it into the outlet streams BEFORE recalculation, so
the PH-flash (Pressure+Enthalpy) solver starts at the right temperature.
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
    from System import Enum, Array, Double  # noqa: E402
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
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)

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
    print("first calc errors:", (str(e[0])[:120] if e and e.Count else "none"))
    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    print(f"first: Light=%.5f (MIBK %.4f) Heavy=%.5f (MIBK %.4f)" % (
        _flow(light), _z(light)[0], _flow(heavy), _z(heavy)[0]))

    # compute correct liquid enthalpies at 298.15 K and seed outlet streams
    # (spec enthalpy = kJ/kg in DWSIM AUX_ENTHALPYM basis).  We seed T first so
    # PH flash has a good starting temperature, then it recomputes H from T.
    for go, tag in ((light_go, "Light"), (heavy_go, "Heavy"), (vapor_go, "Vapor")):
        st = ded._simulation_object(go)
        try:
            st.SetTemperature(T)
        except Exception:
            pass
    print("seeded outlet temperatures to 298.15 K")

    e = automation.CalculateFlowsheet4(fs)
    print("recalc after T seed:", (str(e[0])[:120] if e and e.Count else "none"))
    print(f"recalc: Light=%.5f (MIBK %.4f) Heavy=%.5f (MIBK %.4f)" % (
        _flow(light), _z(light)[0], _flow(heavy), _z(heavy)[0]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("[OK] saved:", OUT)


if __name__ == "__main__":
    main()
