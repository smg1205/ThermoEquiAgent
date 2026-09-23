"""Try the dedicated immiscible-water flash algorithm tag on the native Vessel.

Iterates over candidate PreferredFlashAlgorithmTag strings (the FlashMethod
enum names most relevant to a water-immiscible two-liquid split) and reports
the Light/Heavy liquid outlet flows for each, so we can see which tag actually
makes the native Vessel return two liquid phases.
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

TAG_CANDIDATES = [
    "Nested_Loops_Immiscible_VLLE",
    "Nested_Loops_VLLE",
    "Nested_Loops_SVLLE",
    "Gibbs_Minimization_VLLE",
    "Gibbs_Minimization_Multiphase",
    "Simple_LLE",
]


def _flow(stream):
    return float(stream.GetType().GetMethod("GetMolarFlow").Invoke(stream, None))


def _z(stream):
    return [float(v) for v in stream.GetType().GetMethod("GetOverallComposition").Invoke(stream, None)]


def _run(tag):
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))

    vessel = ded._simulation_object(vessel_go)
    try:
        vessel.FlashTemperature = T
        vessel.FlashPressure = P
        vessel.PreferredFlashAlgorithmTag = tag
    except Exception as e:
        return f"config-err {type(e).__name__}"

    # immiscible-water + three-phase + phase-id switches on the package
    try:
        from DWSIM.Interfaces.Enums import FlashSetting
        pps = list(fs.PropertyPackages.Values)
        pp = vessel.GetType().GetProperty("PropertyPackage").GetValue(vessel, None) or pps[0]
        fsp = pp.GetType().GetProperty("FlashSettings")
        settings = fsp.GetValue(pp, None)
        settings[FlashSetting.ImmiscibleWaterOption] = "True"
        settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
        settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
        fsp.SetValue(pp, settings, None)
    except Exception:
        pass

    for from_go, to_go, fidx, tidx in (
        (feed_go, vessel_go, 0, 0),
        (vessel_go, vapor_go, 0, 0),
        (vessel_go, light_go, 1, 0),
        (vessel_go, heavy_go, 2, 0),
    ):
        try:
            fs.ConnectObjects(from_go.GraphicObject, to_go.GraphicObject, fidx, tidx)
        except Exception:
            pass

    automation.CalculateFlowsheet4(fs)
    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    lf, hf = _flow(light), _flow(heavy)
    lz, hz = _z(light), _z(heavy)
    return f"Light={lf:.5f}(MIBK {lz[0]:.4f})  Heavy={hf:.5f}(MIBK {hz[0]:.4f})"


def main():
    best = None
    for tag in TAG_CANDIDATES:
        try:
            res = _run(tag)
        except Exception as e:
            res = f"ERR {type(e).__name__}"
        print(f"tag={tag:34s} -> {res}")
        if res.startswith("Light="):
            # check if split (both phases nonzero)
            parts = res.split()
            try:
                lf = float(parts[0].split("=")[1])
                hf = float(parts[2].split("=")[1])
            except Exception:
                continue
            if lf > 1e-6 and hf > 1e-6:
                best = tag
                print(f"    *** SPLIT FOUND with {tag} ***")
                break
    if best is None:
        print("\nNo tag produced a two-liquid split through the native Vessel in this DWSIM build.")


if __name__ == "__main__":
    main()
