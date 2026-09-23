"""Native Vessel ternary LLE for ethanol / ethyl acetate / water (mature NRTL BIPs).

This system has reliable built-in DWSIM NRTL parameters:
    Ethyl acetate -> Water : A12=1285.99, A21=1606.08, alpha=0.4393 (cal/mol)
    Ethyl acetate -> Ethanol, Ethanol -> Water also present.
So no parameter needs to be invented, and the mutual solubility is moderate
(unlike the extreme MIBK/water pair) so the Gibbs flash should behave better.

Feed: ethanol + ethyl acetate + water (a composition inside the two-liquid region).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
# ethanol / ethyl acetate / water.  Feed = midpoint of a REAL 298.15 K tie-line
# from datasets/lle/ternary_lle.csv (phaseA [0.086,0.026,0.888] / phaseB [0.172,0.35,0.478]),
# so the composition is guaranteed to lie inside the two-liquid region.
FEED_Z = [0.129, 0.188, 0.683]
# reference experimental tie-line at 298.15 K:
EXP_ORGANIC = [0.172, 0.350, 0.478]
EXP_AQUEOUS = [0.086, 0.026, 0.888]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "etoh_eac_water_native_vessel_lle.dwxmz"


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
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)
    print("[OK] NRTL (built-in BIPs) + GibbsMinimization")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))
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
    for go in (light_go, heavy_go, vapor_go):
        try:
            ded._simulation_object(go).SetTemperature(T)
        except Exception:
            pass

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        err = str(e[0])[:100] if e and e.Count else "none"
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        v = _vals(ded._simulation_object(vapor_go))
        print(f"[{label}] err={err}")
        print(f"   Vapor: flow={v['flow']:.5f}")
        print(f"   Light: flow={l['flow']:.5f} z=[EtOH {l['z'][0]:.4f}, EAc {l['z'][1]:.4f}, W {l['z'][2]:.4f}]")
        print(f"   Heavy: flow={h['flow']:.5f} z=[EtOH {h['z'][0]:.4f}, EAc {h['z'][1]:.4f}, W {h['z'][2]:.4f}]")
        for i, nm in enumerate(("ethanol", "EtAc", "water")):
            cout = l["flow"] * l["z"][i] + h["flow"] * h["z"][i] + v["flow"] * v["z"][i]
            print(f"   {nm}: in={FEED_Z[i]:.4f} out={cout:.5f} diff={FEED_Z[i]-cout:+.5f}")

    print(f"\n===== native Vessel ternary LLE, feed z={FEED_Z} =====")
    print(f"experimental tie-line @298.15K: organic={EXP_ORGANIC}  aqueous={EXP_AQUEOUS}")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
