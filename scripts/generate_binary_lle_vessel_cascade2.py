"""Two-stage counter-current LLE cascade from NATIVE DWSIM Vessels (proof of concept).

Stage = native Vessel with Vapor / Light Liquid / Heavy Liquid outlets,
FlashCalculationApproach = GibbsMinimization (the switch that enables a second
liquid phase), plus a temperature seed on the outlets so recalculation is stable.

Counter-current wiring:
    Feed (MIBK+water) -> Stage 1 inlet 0
    Solvent (water)   -> Stage 2 inlet 0
    Stage 1 light (organic) -> Stage 2 inlet 1
    Stage 2 heavy (aqueous) -> Stage 1 inlet 1
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FEED_Z = [0.4, 0.6]
SOLVENT_Z = [0.0, 1.0]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_vessel_cascade_2stage.dwxmz"


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
    print("[OK] NRTL + GibbsMinimization + immiscible-water")

    def make_stage(tag, x, y):
        v_go = fs.AddObject(object_type.Vessel, x, y, tag)
        v = ded._simulation_object(v_go)
        try:
            v.FlashTemperature = T; v.FlashPressure = P
        except Exception:
            pass
        l_go = fs.AddObject(object_type.MaterialStream, x + 220, y - 45, tag + "_light")
        h_go = fs.AddObject(object_type.MaterialStream, x + 220, y + 45, tag + "_heavy")
        vp_go = fs.AddObject(object_type.MaterialStream, x + 220, y - 110, tag + "_vapor")
        for f, t, fi, ti in ((v_go, l_go, 1, 0), (v_go, h_go, 2, 0), (v_go, vp_go, 0, 0)):
            try:
                fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            except Exception as e:
                print(f"[WARN] outlet {tag}: {type(e).__name__}")
        return {"v": v_go, "light": l_go, "heavy": h_go, "vapor": vp_go}

    s1 = make_stage("Stage_1", 400, 0)
    s2 = make_stage("Stage_2", 400, 300)

    feed_go = fs.AddObject(object_type.MaterialStream, 100, 0, "Feed")
    solv_go = fs.AddObject(object_type.MaterialStream, 100, 300, "Solvent")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))
    solv = ded._simulation_object(solv_go)
    solv.SetTemperature(T); solv.SetPressure(P); solv.SetMolarFlow(1.0)
    solv.SetOverallComposition(ded._composition_argument(SOLVENT_Z))

    def conn(f, t, fi, ti, label):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            print("[OK]", label)
        except Exception as e:
            print("[WARN]", label, type(e).__name__, str(e)[:70])

    conn(feed_go, s1["v"], 0, 0, "Feed -> Stage1.in0")
    conn(solv_go, s2["v"], 0, 0, "Solvent -> Stage2.in0")
    conn(s1["light"], s2["v"], 0, 1, "Stage1.light -> Stage2.in1")

    # The aqueous phase returns from stage 2 to stage 1 -> a recycle loop.
    # Insert DWSIM's native Recycle block plus an intermediate stream (the Recycle
    # outlet cannot connect to a Vessel inlet directly).
    rec_go = fs.AddObject(object_type.OT_Recycle, 300, 640, "Heavy_Recycle")
    rec = ded._simulation_object(rec_go)
    try:
        rec.MaximumIterations = 100
    except Exception:
        pass
    rec_stream_go = fs.AddObject(object_type.MaterialStream, 200, 640, "Recycle_to_S1")
    conn(s2["heavy"], rec_go, 0, 0, "Stage2.heavy -> Recycle.Inlet")
    conn(rec_go, rec_stream_go, 0, 0, "Recycle.Outlet -> Recycle_to_S1")
    conn(rec_stream_go, s1["v"], 0, 1, "Recycle_to_S1 -> Stage1.in1")

    # seed all stage outlets with a temperature so the PH flash on each outlet
    # starts from a sensible point (prevents 'PT Flash: Invalid solution').
    for st in (s1, s2):
        for key in ("light", "heavy", "vapor"):
            try:
                ded._simulation_object(st[key]).SetTemperature(T)
            except Exception:
                pass
    try:
        ded._simulation_object(rec_stream_go).SetTemperature(T)
    except Exception:
        pass

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:")
        for i in range(min(3, e.Count)):
            print("  ", str(e[i])[:170])

    print("\n===== 2-stage native Vessel cascade =====")
    for i, st in enumerate((s1, s2), start=1):
        l = _vals(ded._simulation_object(st["light"]))
        h = _vals(ded._simulation_object(st["heavy"]))
        print(f"  Stage {i}: light flow={l['flow']:.5f} [MIBK {l['z'][0]:.4f}, water {l['z'][1]:.4f}]")
        print(f"           heavy flow={h['flow']:.5f} [MIBK {h['z'][0]:.4f}, water {h['z'][1]:.4f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
