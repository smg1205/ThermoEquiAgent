"""Single native Vessel, ethanol / ethyl acetate / water LLE -- files for GUI testing.

Generates one native-Vessel flowsheet per feed composition so the user can open
them in the DWSIM GUI and test which one splits.

Native units only: MaterialStream + Vessel (Vapor / Light Liquid / Heavy Liquid).
Property package: NRTL with DWSIM's built-in BIPs (nothing is invented).
FlashCalculationApproach = GibbsMinimization (the switch that lets the package
flash look for a second liquid phase).
Outlet streams get a temperature seed so their PH flash has a sane starting point.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0

# Feeds: real 298.15 K tie-line midpoints (guaranteed inside the two-liquid region)
# plus the raw experimental endpoints for reference.
FEEDS = {
    "m1": [0.129, 0.188, 0.683],
    "m2": [0.105, 0.250, 0.645],
    "m3": [0.0737, 0.3314, 0.5949],
    "m4": [0.0428, 0.3911, 0.5661],
}
OUTDIR = ROOT / "lunwen" / "dwsim_demonstration" / "ternary_vessel_files"


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def build(factory, object_type, automation, tag, z, out_path):
    from System import Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402

    fs = automation.CreateFlowsheet()
    fs.AddCompound("Ethanol")
    fs.AddCompound("Ethyl acetate")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]

    # only the multi-phase approach lets the flash search for a 2nd liquid
    ap = pp.GetType().GetProperty("FlashCalculationApproach")
    ap.SetValue(pp, Enum.Parse(ap.PropertyType, "GibbsMinimization"), None)
    fsp = pp.GetType().GetProperty("FlashSettings")
    s = fsp.GetValue(pp, None)
    s[FlashSetting.ImmiscibleWaterOption] = "True"
    s[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    s[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    fsp.SetValue(pp, s, None)

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
        v.FlashTemperature = T
        v.FlashPressure = P
    except Exception:
        pass

    for f, t, fi, ti in ((feed_go, v_go, 0, 0),
                         (v_go, vap_go, 0, 0),
                         (v_go, light_go, 1, 0),
                         (v_go, heavy_go, 2, 0)):
        try:
            fs.ConnectObjects(f.GraphicObject, t.GraphicObject, fi, ti)
        except Exception as e:
            print(f"   [WARN] connect: {type(e).__name__}")

    # temperature seed on the three outlets (their spec is Pressure_and_Enthalpy)
    for go in (light_go, heavy_go, vap_go):
        try:
            ded._simulation_object(go).SetTemperature(T)
        except Exception:
            pass

    errs = automation.CalculateFlowsheet4(fs)
    if errs and errs.Count:
        status = "ERROR: " + str(errs[0])[:70]
    else:
        l = _vals(ded._simulation_object(light_go))
        h = _vals(ded._simulation_object(heavy_go))
        split = l["flow"] > 1e-6 and h["flow"] > 1e-6
        status = (f"{'SPLIT' if split else 'single-liquid'}  "
                  f"Light={l['flow']:.4f} [{l['z'][0]:.3f},{l['z'][1]:.3f},{l['z'][2]:.3f}]  "
                  f"Heavy={h['flow']:.4f} [{h['z'][0]:.3f},{h['z'][1]:.3f},{h['z'][2]:.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, out_path)
    return status


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    print("=== single native Vessel, ethanol / ethyl acetate / water, 298.15 K ===")
    for tag, z in FEEDS.items():
        name = f"etoh_eac_water_vessel_{tag}.dwxmz"
        try:
            st = build(factory, object_type, automation, tag, z, OUTDIR / name)
        except Exception as e:
            st = f"EXC {type(e).__name__}: {str(e)[:60]}"
        print(f"  feed {tag} z={z}")
        print(f"     -> {name}: {st}")


if __name__ == "__main__":
    main()
