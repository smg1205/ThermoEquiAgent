"""Test the native Vessel LLE at 333.15 K (60 C) instead of 298.15 K.

At 60 C the two phases are still strongly split but less extreme than at 25 C, so
the Gibbs minimisation flash may be numerically better behaved.  60/70/80 C also
have experimental tie-lines (doi 10.1016/j.fluid.2016.11.005) to compare against:
organic x_MIBK = 0.8109, aqueous x_MIBK = 0.0023 at 333.15 K.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 333.15
P = 101325.0
Z = [0.4, 0.6]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle_333K.dwxmz"


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
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)
    print(f"[OK] NRTL + GibbsMinimization @ {T} K")

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
    # seed outlet temperatures
    for go in (light_go, heavy_go, vapor_go):
        try:
            ded._simulation_object(go).SetTemperature(T)
        except Exception:
            pass

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        err = str(e[0])[:110] if e and e.Count else "none"
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        print(f"[{label}] err={err}")
        print(f"   Light: flow={l['flow']:.5f} [MIBK {l['z'][0]:.5f}, water {l['z'][1]:.5f}]")
        print(f"   Heavy: flow={h['flow']:.5f} [MIBK {h['z'][0]:.5f}, water {h['z'][1]:.5f}]")
        # per-component balance
        for i, nm in enumerate(("MIBK", "water")):
            cin = 1.0 * Z[i]
            cout = l["flow"] * l["z"][i] + h["flow"] * h["z"][i]
            print(f"   {nm}: in={cin:.5f} out={cout:.5f} diff={cin-cout:+.5f}")

    print(f"\n===== native Vessel LLE @ {T} K =====")
    print("experimental: organic x_MIBK=0.8109, aqueous x_MIBK=0.0023")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
