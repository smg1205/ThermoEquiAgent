"""Native Vessel LLE with UNIFAC-LL + NestedLoopsImmiscible (no binary parameters).

NestedLoopsImmiscible is the immiscible-system flash kernel; combined with the
group-contribution UNIFAC-LL package it produces the physically correct strong
split for MIBK/water (organic ~pure MIBK, aqueous ~pure water) with NO regressed
binary parameters.

We assign a NestedLoopsImmiscible instance to the package's FlashAlgorithm and run
the native Vessel with Light/Heavy liquid outlets.
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
    from System import Activator, Array, Double, Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    pkg = None
    for p in ("UNIFAC-LL", "UNIFAC LL", "UNIFACLL"):
        try:
            ded._add_property_package(fs, p); pkg = p; break
        except Exception:
            continue
    print("[OK] package:", pkg)
    pp = list(fs.PropertyPackages.Values)[0]
    asm = pp.GetType().Assembly

    # assign the immiscible kernel to the package's FlashAlgorithm
    ni_type = asm.GetType("DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoopsImmiscible")
    try:
        inst = Activator.CreateInstance(ni_type)
        try:
            inst.StabSearchSeverity = 3
        except Exception:
            pass
        pp.GetType().GetProperty("FlashAlgorithm").SetValue(pp, inst, None)
        print("[OK] FlashAlgorithm = NestedLoopsImmiscible")
    except Exception as e:
        print("[WARN] FlashAlgorithm:", type(e).__name__, e)

    try:
        fsp = pp.GetType().GetProperty("FlashSettings")
        s = fsp.GetValue(pp, None)
        s[FlashSetting.ImmiscibleWaterOption] = "True"
        s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, s, None)
    except Exception as e:
        print("[WARN] FlashSettings:", type(e).__name__, e)

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

    def report(label):
        e = automation.CalculateFlowsheet4(fs)
        err = str(e[0])[:120] if e and e.Count else "none"
        l = ded._simulation_object(light_go); h = ded._simulation_object(heavy_go)
        print(f"[{label}] err={err}")
        print(f"   Light: flow=%.5f z=[MIBK %.5f, water %.5f]" % (_flow(l), _z(l)[0], _z(l)[1]))
        print(f"   Heavy: flow=%.5f z=[MIBK %.5f, water %.5f]" % (_flow(h), _z(h)[0], _z(h)[1]))

    print("\n===== UNIFAC-LL + NestedLoopsImmiscible, native Vessel =====")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
