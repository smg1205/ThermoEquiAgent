"""Water + 1-butanol LLE using DWSIM's DEFAULT flash path (no forced Gibbs).

The built-in NRTL pair for water/1-butanol exists and looks physically sensible
(A12=2633.695, A21=504.038 cal/mol, alpha=0.4447), so we let the stream use its
own default TP flash rather than forcing GibbsMinimization (which is the
numerically fragile kernel).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

P = 101325.0
CASES = [
    ("T25", 298.15, [0.3, 0.7]),
    ("T25b", 298.15, [0.5, 0.5]),
    ("T25c", 298.15, [0.1, 0.9]),
    ("T40", 313.15, [0.3, 0.7]),
    ("T60", 333.15, [0.3, 0.7]),
]
OUTDIR = ROOT / "lunwen" / "dwsim_demonstration" / "butanol_test"


def phases(feed):
    out = {}
    for nm in ("Vapor", "Liquid1", "Liquid2"):
        try:
            p = feed.GetType().GetProperty(nm).GetValue(feed, None)
            out[nm] = float(p.Properties.molarfraction) if p is not None else None
        except Exception:
            out[nm] = None
    try:
        out["ids"] = list(feed.PhaseIds) if feed.PhaseIds else None
    except Exception:
        out["ids"] = None
    return out


def comp_of(feed):
    return [float(v) for v in feed.GetType().GetMethod("GetOverallComposition").Invoke(feed, None)]


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    print("=== water / 1-butanol, DEFAULT flash path (no Gibbs forcing) ===")
    for label, T, z in CASES:
        for pkg in ("NRTL", "UNIQUAC"):
            fs = automation.CreateFlowsheet()
            try:
                fs.AddCompound("1-butanol")
            except Exception:
                fs.AddCompound("N-butanol")
            fs.AddCompound("Water")
            try:
                ded._add_property_package(fs, pkg)
            except Exception as e:
                print(f"  {label} {pkg}: pkg fail {type(e).__name__}")
                continue

            feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
            feed = ded._simulation_object(feed_go)
            feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
            feed.SetOverallComposition(ded._composition_argument(z))
            errs = automation.CalculateFlowsheet4(fs)
            err = str(errs[0])[:40] if errs and errs.Count else None
            ph = phases(feed)
            split = ph.get("Liquid2") is not None and ph["Liquid2"] > 1e-8
            print(f"  {label:4s} {pkg:8s} T={T} z={z} -> L1={ph.get('Liquid1')} L2={ph.get('Liquid2')} "
                  f"ids={ph.get('ids')} {'*** SPLIT ***' if split else 'single'}"
                  f"{'  ERR:'+err if err else ''}")

    # build one file for GUI inspection using the most promising setting
    fs = automation.CreateFlowsheet()
    try:
        fs.AddCompound("1-butanol")
    except Exception:
        fs.AddCompound("N-butanol")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    v_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vap_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(298.15); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument([0.3, 0.7]))
    v = ded._simulation_object(v_go)
    try:
        v.FlashTemperature = 298.15; v.FlashPressure = P
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
            ded._simulation_object(go).SetTemperature(298.15)
        except Exception:
            pass
    automation.CalculateFlowsheet4(fs)
    out = OUTDIR / "water_butanol_NRTL_default.dwxmz"
    out.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, out)
    print("\n[OK] saved:", out)


if __name__ == "__main__":
    main()
