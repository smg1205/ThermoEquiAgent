"""Open (cross-current) multi-stage LLE cascade from NATIVE DWSIM Vessels.

No Recycle block -> no iteration -> every stage converges in one pass and the
overall mass balance closes.

Structure (cross-current extraction):
    Feed (MIBK+water) -> Stage 1  (fresh solvent)  -> organic product 1 + aqueous raffinate 1
    raffinate 1       -> Stage 2  (fresh solvent)  -> organic product 2 + aqueous raffinate 2
    ...
    raffinate N-1     -> Stage N  (fresh solvent)  -> organic product N + final raffinate

Each stage is a native Vessel (Vapor / Light Liquid / Heavy Liquid outlets) with
FlashCalculationApproach = GibbsMinimization (the switch enabling a 2nd liquid
phase) and a temperature seed on the outlets for stable TP flashes.

All units are native DWSIM objects (Vessel + MaterialStream); no CustomUO script.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FEED_Z = [0.4, 0.6]      # MIBK, water
SOLVENT_Z = [0.0, 1.0]   # fresh water solvent
FEED_FLOW = 1.0
SOLVENT_FLOW = 0.5       # fresh solvent per stage
NSTAGES = 3
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_cascade_open.dwxmz"


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

    def stream(tag, x, y, z=None, flow=None):
        go = fs.AddObject(object_type.MaterialStream, x, y, tag)
        o = ded._simulation_object(go)
        if z is not None:
            o.SetTemperature(T); o.SetPressure(P); o.SetMolarFlow(flow)
            o.SetOverallComposition(ded._composition_argument(z))
        else:
            try:
                o.SetTemperature(T)
            except Exception:
                pass
        return go

    def vessel(tag, x, y):
        go = fs.AddObject(object_type.Vessel, x, y, tag)
        o = ded._simulation_object(go)
        try:
            o.FlashTemperature = T; o.FlashPressure = P
        except Exception:
            pass
        return go

    def conn(a, b, ai, bi, label):
        try:
            fs.ConnectObjects(a.GraphicObject, b.GraphicObject, ai, bi)
            return True
        except Exception as e:
            print(f"[WARN] {label}: {type(e).__name__}")
            return False

    feed_go = stream("Feed", 0, 0, FEED_Z, FEED_FLOW)

    stages = []
    # The MIBK-bearing stream is the ORGANIC (light) phase; each stage washes it
    # with fresh water, so the organic phase carries on to the next stage while
    # the loaded aqueous extract is withdrawn.  (Carrying the aqueous phase
    # forward would just dilute it into pure water and nothing could split.)
    prev_organic = feed_go
    for i in range(NSTAGES):
        y = -260 + i * 260
        v_go = vessel(f"Stage_{i+1}", 400, y)
        sol_go = stream(f"Solvent_{i+1}", 120, y + 90, SOLVENT_Z, SOLVENT_FLOW)
        light_go = stream(f"Organic_Product_{i+1}", 700, y - 60)
        heavy_go = stream(f"Aqueous_Extract_{i+1}", 700, y + 60)
        vap_go = stream(f"Vapor_{i+1}", 700, y - 130)

        conn(sol_go, v_go, 0, 0, f"solvent{i+1}->stage{i+1}")
        conn(prev_organic, v_go, 0, 1, f"organic{i}->stage{i+1}")
        conn(v_go, light_go, 1, 0, f"stage{i+1}->organic{i+1}")
        conn(v_go, heavy_go, 2, 0, f"stage{i+1}->aqueous{i+1}")
        conn(v_go, vap_go, 0, 0, f"stage{i+1}->vapor{i+1}")

        stages.append({"v": v_go, "light": light_go, "heavy": heavy_go})
        prev_organic = light_go   # light (organic) phase carries to next stage

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:")
        for i in range(min(3, e.Count)):
            print("  ", str(e[i])[:160])

    print(f"\n===== open cross-current cascade ({NSTAGES} native Vessels) =====")
    print(f"feed: {FEED_FLOW} mol/s z=[MIBK {FEED_Z[0]}, water {FEED_Z[1]}]  "
          f"solvent: {SOLVENT_FLOW} mol/s per stage")
    for i, st in enumerate(stages, start=1):
        l = _vals(ded._simulation_object(st["light"]))
        h = _vals(ded._simulation_object(st["heavy"]))
        print(f"  Stage {i}: organic flow={l['flow']:.5f} [MIBK {l['z'][0]:.4f}, water {l['z'][1]:.4f}]")
        print(f"           aqueous flow={h['flow']:.5f} [MIBK {h['z'][0]:.4f}, water {h['z'][1]:.4f}]")

    # overall balance: in = feed + N*fresh solvent
    # out = all aqueous extracts + the FINAL organic product
    feed_v = _vals(ded._simulation_object(feed_go))
    in_mibk = feed_v["flow"] * feed_v["z"][0]
    in_water = feed_v["flow"] * feed_v["z"][1] + NSTAGES * SOLVENT_FLOW * SOLVENT_Z[1]
    out_mibk = out_water = 0.0
    for st in stages:
        h = _vals(ded._simulation_object(st["heavy"]))     # aqueous extracts
        out_mibk += h["flow"] * h["z"][0]
        out_water += h["flow"] * h["z"][1]
    final_l = _vals(ded._simulation_object(stages[-1]["light"]))   # final organic product
    out_mibk += final_l["flow"] * final_l["z"][0]
    out_water += final_l["flow"] * final_l["z"][1]
    print("\n--- overall mass balance ---")
    print(f"  MIBK  in={in_mibk:.5f} out={out_mibk:.5f} diff={in_mibk-out_mibk:+.2e}")
    print(f"  water in={in_water:.5f} out={out_water:.5f} diff={in_water-out_water:+.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
