"""Target A: produce DWSIM files where ethanol/ethyl acetate/water visibly splits.

Strategy: lower temperature (283.15 K widens the two-phase region), feeds taken from
the interior of the two-phase region, and three property packages (NRTL / UNIQUAC /
UNIFAC-LL) to find one that actually resolves the split.

Each file: Feed (TP flash) -> native Vessel (Vapor / Light Liquid / Heavy Liquid).
The Feed stream is also reported because its TP flash is the cleanest indicator of
whether the package predicts two liquid phases at all.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 283.15
P = 101325.0
OUTDIR = ROOT / "lunwen" / "dwsim_demonstration" / "ternary_split_test"

# 283.15 K tie-line midpoints (interior of the two-phase region)
FEEDS = {
    "a": [0.0604, 0.3793, 0.5603],
    "b": [0.1060, 0.2971, 0.5969],
    "c": [0.1279, 0.2451, 0.6271],
    "d": [0.1431, 0.2015, 0.6553],
}
PACKAGES = ["NRTL", "UNIQUAC", "UNIFAC-LL"]


def feed_phase_state(automation, feed_go):
    """Read the Feed stream's own phase split after its TP flash."""
    feed = ded._simulation_object(feed_go)
    out = {}
    for nm in ("Vapor", "Liquid1", "Liquid2"):
        try:
            p = feed.GetType().GetProperty(nm).GetValue(feed, None)
            out[nm] = float(p.Properties.molarfraction) if p is not None else None
        except Exception:
            out[nm] = None
    try:
        out["PhaseIds"] = list(feed.PhaseIds) if feed.PhaseIds else None
    except Exception:
        out["PhaseIds"] = None
    return out


def build(factory, object_type, automation, z, pkg, out_path):
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402

    fs = automation.CreateFlowsheet()
    fs.AddCompound("Ethanol")
    fs.AddCompound("Ethyl acetate")
    fs.AddCompound("Water")
    added = None
    for cand in (pkg, "NRTL", "UNIQUAC", "UNIFAC-LL"):
        try:
            ded._add_property_package(fs, cand)
            added = cand
            break
        except Exception:
            continue
    pp = list(fs.PropertyPackages.Values)[0]

    # multi-phase approach + VLLE forcing + immiscible water
    try:
        ap = pp.GetType().GetProperty("FlashCalculationApproach")
        ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    except Exception:
        pass
    try:
        fsp = pp.GetType().GetProperty("FlashSettings")
        s = fsp.GetValue(pp, None)
        s[FlashSetting.ForceEquilibriumCalculationType] = "VLLE"
        s[FlashSetting.ImmiscibleWaterOption] = "True"
        s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, s, None)
    except Exception:
        pass

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    v_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(z))
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

    errs = automation.CalculateFlowsheet4(fs)
    err = str(errs[0])[:60] if errs and errs.Count else None

    ph = feed_phase_state(automation, feed_go)
    lf = float(ded._simulation_object(light_go).GetType().GetMethod("GetMolarFlow").Invoke(
        ded._simulation_object(light_go), None))
    hf = float(ded._simulation_object(heavy_go).GetType().GetMethod("GetMolarFlow").Invoke(
        ded._simulation_object(heavy_go), None))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, out_path)
    return {"added_pkg": added, "err": err, "feed_phases": ph, "light": lf, "heavy": hf}


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    print(f"=== target A: T={T} K, feeds inside the two-phase region ===")
    for fkey, z in FEEDS.items():
        for pkg in PACKAGES:
            name = f"split_{fkey}_{pkg.replace('-', '').replace(' ', '')}.dwxmz"
            try:
                r = build(factory, object_type, automation, z, pkg, OUTDIR / name)
                ph = r["feed_phases"]
                l2 = ph.get("Liquid2")
                verdict = ("SPLIT" if (l2 is not None and l2 > 1e-8)
                           or (r["light"] > 1e-6 and r["heavy"] > 1e-6) else "single")
                print(f"  {name:28s} pkg={r['added_pkg']:8s} "
                      f"Feed[L1={ph.get('Liquid1')}, L2={l2}] "
                      f"Vessel[L={r['light']:.4f}, H={r['heavy']:.4f}] "
                      f"{'-> ' + verdict}{'  ERR:' + r['err'] if r['err'] else ''}")
            except Exception as e:
                print(f"  {name:28s} EXC {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
