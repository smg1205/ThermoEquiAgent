"""Multi-stage counter-current LLE cascade built from NATIVE DWSIM Vessels.

Each stage is a native Vessel (Vapor / Light Liquid / Heavy Liquid outlets) with
FlashCalculationApproach = GibbsMinimization, which is the only switch that makes
the package flash search for a second liquid phase.  Stages are chained
counter-currently:

    Feed (MIBK+water)  -> stage 1
    Solvent (water)    -> stage N
    light (organic/MIBK-rich) phase flows 1 -> N   (toward the solvent inlet)
    heavy (aqueous/water-rich) phase flows N -> 1  (toward the feed inlet)

Everything is native; the "column" is realised as a cascade of Vessels.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FEED_Z = [0.4, 0.6]     # MIBK, water
SOLVENT_Z = [0.0, 1.0]  # pure water
FEED_FLOW = 1.0
SOLVENT_FLOW = 1.0
NSTAGES = 3
USE_UNIFAC_LL = False   # use NRTL (has MIBK/water pair); UNIFAC-LL also selectable
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_vessel_cascade.dwxmz"


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
    pkg = "NRTL" if not USE_UNIFAC_LL else "UNIFAC-LL"
    ded._add_property_package(fs, pkg)
    pp = list(fs.PropertyPackages.Values)[0]
    print("[OK] package:", pkg)

    # the one switch that enables a two-liquid flash at every stage
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationStyle if False else FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)
    print("[OK] GibbsMinimization + immiscible-water on all flashes")

    # ---- build stages -----------------------------------------------------
    stages = []      # list of dicts: vessel_go, light_go, heavy_go, vapor_go
    for i in range(NSTAGES):
        v_go = fs.AddObject(object_type.Vessel, 400, -200 + i * 200, f"Stage_{i+1}")
        v = ded._simulation_object(v_go)
        try:
            v.FlashTemperature = T
            v.FlashPressure = P
        except Exception:
            pass
        light_go = fs.AddObject(object_type.MaterialStream, 650, -240 + i * 200, f"L{i+1}_out")
        heavy_go = fs.AddObject(object_type.MaterialStream, 650, -160 + i * 200, f"H{i+1}_out")
        vapor_go = fs.AddObject(object_type.MaterialStream, 650, -300 + i * 200, f"V{i+1}_out")
        for f, t, fi, ti in ((v_go, light_go, 1, 0), (v_go, heavy_go, 2, 0), (v_go, vapor_go, 0, 0)):
            try:
                fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            except Exception:
                pass
        stages.append({"v": v_go, "light": light_go, "heavy": heavy_go, "vapor": vapor_go})

    # external feeds
    feed_go = fs.AddObject(object_type.MaterialStream, 100, -200, "Feed")
    solv_go = fs.AddObject(object_type.MaterialStream, 900, -200 + (NSTAGES - 1) * 200, "Solvent")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(FEED_FLOW)
    feed.SetOverallComposition(ded._composition_argument(FEED_Z))
    solv = ded._simulation_object(solv_go)
    solv.SetTemperature(T); solv.SetPressure(P); solv.SetMolarFlow(SOLVENT_FLOW)
    solv.SetOverallComposition(ded._composition_argument(SOLVENT_Z))

    # internal inter-stage streams:  light i -> stage i+1 ; heavy i+1 -> stage i
    inter_light = []
    inter_heavy = []
    for i in range(NSTAGES - 1):
        lg = fs.AddObject(object_type.MaterialStream, 520, -200 + i * 200 + 100, f"L{i+1}to{i+2}")
        hg = fs.AddObject(object_type.MaterialStream, 520, -200 + i * 200 - 100, f"H{i+2}to{i+1}")
        inter_light.append(lg)
        inter_heavy.append(hg)

    # ---- connect feeds ----------------------------------------------------
    def conn(f, t, fi, ti, label):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
            return True
        except Exception as e:
            print(f"[WARN] {label}: {type(e).__name__}")
            return False

    # Feed into stage 1 (inlet 0); solvent into stage N (inlet 0)
    conn(feed_go, stages[0]["v"], 0, 0, "Feed -> stage1")
    conn(solv_go, stages[NSTAGES - 1]["v"], 0, 0, "Solvent -> stageN")

    # light phase: stage i light -> stage i+1 inlet 1
    for i in range(NSTAGES - 1):
        conn(stages[i]["light"], stages[i + 1]["v"], 0, 1, f"L{i+1}->stage{i+2}")
    # heavy phase: stage i+1 heavy -> stage i inlet 2
    for i in range(NSTAGES - 1):
        conn(stages[i + 1]["heavy"], stages[i]["v"], 0, 2, f"H{i+2}->stage{i+1}")

    # make inter-stage streams actual objects we connected (they are the vessels' outlets)
    # NOTE: we connected vessel outlet ports directly to the next stage's inlet; the
    # 'inter_light/inter_heavy' streams above are unused placeholders -> remove them.
    for go in inter_light + inter_heavy:
        try:
            fs.DeleteObject(go)
        except Exception:
            pass

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:")
        for i in range(min(3, e.Count)):
            print("  ", str(e[i])[:170])

    print(f"\n===== native Vessel cascade ({NSTAGES} stages, {pkg}) =====")
    for i, st in enumerate(stages):
        l = _vals(ded._simulation_object(st["light"]))
        h = _vals(ded._simulation_object(st["heavy"]))
        print(f"  Stage {i+1}: light flow={l['flow']:.5f} [MIBK {l['z'][0]:.4f}, water {l['z'][1]:.4f}]")
        print(f"           heavy flow={h['flow']:.5f} [MIBK {h['z'][0]:.4f}, water {h['z'][1]:.4f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
