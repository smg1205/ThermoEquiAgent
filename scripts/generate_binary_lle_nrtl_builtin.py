"""Use DWSIM's built-in NRTL package with its OWN estimate for MIBK/water, no manual params.

Pure-native route: NRTL property package + let DWSIM estimate the MIBK/water
interaction + GibbsMinimization + native Vessel.  We do NOT write any parameter;
we just inspect what DWSIM's built-in/estimated NRTL gives and run the flash.
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
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_nrtl_builtin_lle.dwxmz"


def _flow(s):
    return float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))
def _z(s):
    return [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)]


def main():
    factory, object_type = ded._automation_factory()
    from System import Enum, Activator, Array, Double  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]

    # inspect what NRTL already has for MIBK/water (built-in or estimated)
    try:
        m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
        ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
        print("existing NRTL pairs containing MIBK or Water (sample):")
        for k in list(ips.Keys):
            if "Methyl" in str(k) or "isobutyl" in str(k).lower():
                inner = ips[k]
                for k2 in list(inner.Keys):
                    d = inner[k2]
                    print(f"  {k} -> {k2}: A12={d.A12:.2f} A21={d.A21:.2f} alpha={d.alpha12:.4f}")
    except Exception as e:
        print("inspect err:", type(e).__name__, e)

    # enable auto-estimate (let DWSIM fill/estimate missing pairs) and reconfigure
    try:
        pp.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(pp, True, None)
        pp.GetType().GetProperty("AreModelParametersDirty").SetValue(pp, True, None)
        cfg = pp.GetType().GetMethod("ConfigParameters")
        if cfg is not None:
            cfg.Invoke(pp, None)
        print("[OK] AutoEstimate=True, ConfigParameters() done")
    except Exception as e:
        print("[WARN] estimate config:", type(e).__name__, e)

    # GibbsMinimization + immiscible water
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
    if e and e.Count:
        print("calc errors:", str(e[0])[:160])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    print("\n===== built-in NRTL (auto-estimate) MIBK/water @ 298.15 K =====")
    print(f"Light_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(light), _z(light)[0], _z(light)[1]))
    print(f"Heavy_Liquid: flow=%.5f  z=[MIBK %.5f, water %.5f]" % (_flow(heavy), _z(heavy)[0], _z(heavy)[1]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
