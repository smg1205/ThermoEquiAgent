"""Use ForceEquilibriumCalculationType = 'VLLE' (the documented mode) to force a
vapour-liquid-LIQUID equilibrium calculation for ethanol/ethyl acetate/water.

PropertyPackage.SetPhaseEquilibriaCalculationMode documents the accepted values:
    'Default', 'VLE', 'VLLE', 'SVLE', 'SVLLE'
'VLLE' should make the package perform a liquid-liquid-capable flash, which is
what the native Vessel needs.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FEED_Z = [0.129, 0.188, 0.683]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "ternary_vessel_files" / "etoh_eac_water_vessel_vlle.dwxmz"


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def main():
    factory, object_type = ded._automation_factory()
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    fs.AddCompound("Ethanol")
    fs.AddCompound("Ethyl acetate")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]

    # documented API for forcing the equilibrium type (not exposed on the
    # IPropertyPackage automation interface, so write the backing setting直接)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    for mode in ("VLLE", "SVLLE"):
        try:
            s[FlashSetting.ForceEquilibriumCalculationType] = mode
            print(f"[OK] ForceEquilibriumCalculationType = {mode!r}")
            break
        except Exception as e:
            print(f"[WARN] {mode}: {type(e).__name__} {e}")

    # keep the multi-phase approach as well
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    print("   ForceEquilibriumCalculationType =",
          s.get(FlashSetting.ForceEquilibriumCalculationType))
    fsp.SetValue(pp, s, None)

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    v_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))
    v = ded._simulation_object(v_go)
    try:
        v.FlashTemperature = T; v.FlashPressure = P
    except Exception:
        pass
    for f, t, fi, ti in ((feed_go, v_go, 0, 0), (v_go, vap_go, 0, 0),
                         (v_go, light_go, 1, 0), (v_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception:
            pass
    for go in (light_go, heavy_go, vap_go):
        try:
            ded._simulation_object(go).SetTemperature(T)
        except Exception:
            pass

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        err = str(e[0])[:100] if e and e.Count else "none"
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        print(f"[{label}] err={err}")
        print(f"   Light: flow={l['flow']:.5f} z=[{l['z'][0]:.4f},{l['z'][1]:.4f},{l['z'][2]:.4f}]")
        print(f"   Heavy: flow={h['flow']:.5f} z=[{h['z'][0]:.4f},{h['z'][1]:.4f},{h['z'][2]:.4f}]")

    print(f"\n===== ForceEquilibriumCalculationType=VLLE, feed={FEED_Z} =====")
    print("experiment: organic [0.172,0.350,0.478]  aqueous [0.086,0.026,0.888]")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
