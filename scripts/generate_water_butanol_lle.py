"""Water + 1-butanol binary LLE: check DWSIM's built-in NRTL pair, then run a native
Vessel to see whether it splits into two liquid phases.

Classic partially-miscible pair.  We first report the built-in NRTL parameters for
butanol/water (if present), then run the TP flash + native Vessel.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
# 1-butanol / water, overall composition inside the two-phase region
FEED_Z = [0.3, 0.7]
BUTANOL_CANDIDATES = ["1-butanol", "N-butanol", "n-butanol", "1-Butanol", "Butanol"]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_butanol_native_vessel_lle.dwxmz"


def _vals(s):
    t = s.GetType()
    return (float(t.GetMethod("GetMolarFlow").Invoke(s, None)),
            [float(v) for v in t.GetMethod("GetOverallComposition").Invoke(s, None)])


def main():
    factory, object_type = ded._automation_factory()
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()

    added = None
    for cand in BUTANOL_CANDIDATES:
        try:
            fs.AddCompound(cand)
            added = cand
            break
        except Exception:
            continue
    print("butanol compound resolved as:", added)
    if added is None:
        print("[FATAL] no butanol key worked")
        return
    fs.AddCompound("Water")

    for pkg in ("NRTL", "UNIQUAC", "UNIFAC-LL"):
        try:
            ded._add_property_package(fs, pkg)
            print("[OK] package:", pkg)
            break
        except Exception as e:
            print(f"[WARN] {pkg}: {type(e).__name__}")

    pp = list(fs.PropertyPackages.Values)[0]

    # report the built-in NRTL pair for butanol/water
    try:
        m_uni = pp.GetType().GetProperty("m_uni").GetValue(pp, None)
        ips = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
        found = False
        for k in list(ips.Keys):
            if "butanol" in str(k).lower() or "Butanol" in str(k):
                inner = ips[k]
                for k2 in list(inner.Keys):
                    if "water" in str(k2).lower() or "Agua" in str(k2):
                        d = inner[k2]
                        print(f"  built-in NRTL {k} -> {k2}: A12={d.A12:.3f} A21={d.A21:.3f} alpha={d.alpha12:.4f}")
                        found = True
        for k in list(ips.Keys):
            if "water" in str(k).lower() or k == "Agua":
                inner = ips[k]
                for k2 in list(inner.Keys):
                    if "butanol" in str(k2).lower():
                        d = inner[k2]
                        print(f"  built-in NRTL {k} -> {k2}: A12={d.A12:.3f} A21={d.A21:.3f} alpha={d.alpha12:.4f}")
                        found = True
        if not found:
            print("  (no built-in butanol/water pair found in the dictionary listing)")
    except Exception as e:
        print("  inspect err:", type(e).__name__, e)

    # multi-phase flash approach
    try:
        ap = pp.GetType().GetProperty("FlashCalculationApproach")
        ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
        fsp = pp.GetType().GetProperty("FlashSettings")
        s = fsp.GetValue(pp, None)
        s[FlashSetting.ImmiscibleWaterOption] = "True"
        s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, s, None)
    except Exception as e:
        print("[WARN] flash config:", type(e).__name__)

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
        err = str(e[0])[:90] if e and e.Count else "none"
        # feed phase state (TP flash, the cleanest indicator)
        try:
            l1 = feed.GetType().GetProperty("Liquid1").GetValue(feed, None)
            l2 = feed.GetType().GetProperty("Liquid2").GetValue(feed, None)
            f1 = float(l1.Properties.molarfraction) if l1 is not None else None
            f2 = float(l2.Properties.molarfraction) if l2 is not None else None
            ids = list(feed.PhaseIds) if feed.PhaseIds else None
        except Exception:
            f1 = f2 = ids = None
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        print(f"[{label}] err={err}")
        print(f"   Feed phase: Liquid1={f1} Liquid2={f2} ids={ids}")
        print(f"   Light: flow={l[0]:.5f} z=[butanol {l[1][0]:.4f}, water {l[1][1]:.4f}]")
        print(f"   Heavy: flow={h[0]:.5f} z=[butanol {h[1][0]:.4f}, water {h[1][1]:.4f}]")

    print("\n===== water / 1-butanol LLE @ 298.15 K, native Vessel =====")
    report("first")
    report("recalc")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
