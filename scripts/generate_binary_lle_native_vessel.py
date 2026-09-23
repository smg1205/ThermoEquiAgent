"""Native-Vessel liquid-liquid split for MIBK/water (no CustomUO, no IronPython).

Uses DWSIM's built-in Vessel with its two native liquid outlets:
    Outlet Port #1 = Light Liquid Outlet   (organic, MIBK-rich)
    Outlet Port #2 = Heavy Liquid Outlet   (aqueous, water-rich)
plus a VLLE-capable flash algorithm (Nested Loops (VLLE)) and explicit
FlashTemperature/FlashPressure.  This is the pure-native way to force a
two-liquid-phase split without a scripted CustomUO.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]  # x_MIBK, x_water
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def _values(stream):
    t = stream.GetType()
    return {
        "flow": float(t.GetMethod("GetMolarFlow").Invoke(stream, None)),
        "z": [float(v) for v in t.GetMethod("GetOverallComposition").Invoke(stream, None)],
        "T": float(t.GetMethod("GetTemperature").Invoke(stream, None)),
    }


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()

    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand)
            break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIQUAC")

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LL")
    light_go = fs.AddObject(object_type.MaterialStream, 700, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 700, 70, "Heavy_Liquid")

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T)
    feed.SetPressure(P)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))

    vessel = ded._simulation_object(vessel_go)
    # Tell the vessel to flash at the feed's T/P and use the VLLE algorithm.
    try:
        vessel.FlashTemperature = T
        vessel.FlashPressure = P
        vessel.PreferredFlashAlgorithmTag = "Nested Loops (VLLE)"
        vessel.OverrideT = True
        vessel.OverrideP = True
        print("[OK] vessel configured: VLLE flash @", T, "K")
    except Exception as e:
        print("[WARN] vessel config partial:", type(e).__name__, e)

    # Connect feed -> inlet 0; vessel light-liquid outlet(1) -> Light_Liquid;
    # vessel heavy-liquid outlet(2) -> Heavy_Liquid.
    for label, fn in (
        ("feed->vessel", lambda: vessel.ConnectFeedMaterialStream(feed, 0)),
        ("vessel->light", lambda: vessel.ConnectProductMaterialStream(
            ded._simulation_object(light_go), 1)),
        ("vessel->heavy", lambda: vessel.ConnectProductMaterialStream(
            ded._simulation_object(heavy_go), 2)),
    ):
        try:
            fn()
            print("[OK]", label)
        except Exception as e:
            print("[ERR]", label, type(e).__name__, e)

    # graphical connections.  Vessel graphic: 7 input pts, 4 output pts
    # (0=vapor, 1=light liquid, 2=heavy liquid, 3=relief).  ConnectObjects(from,to,
    # fromidx,toidx): for vessel->stream use fromidx=vessel outlet index.
    for from_go, to_go, fidx, tidx, label in (
        (feed_go, vessel_go, 0, 0, "feed graphic"),
        (vessel_go, light_go, 1, 0, "light graphic (outlet 1)"),
        (vessel_go, heavy_go, 2, 0, "heavy graphic (outlet 2)"),
    ):
        try:
            fs.ConnectObjects(from_go.GraphicObject, to_go.GraphicObject, fidx, tidx)
            print("[OK] graphic", label)
        except Exception as e:
            print("[WARN] graphic", label, type(e).__name__, e)

    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("solve errors:")
        for i in range(errors.Count):
            print("  ", errors[i])

    light = ded._simulation_object(light_go)
    heavy = ded._simulation_object(heavy_go)
    lv = _values(light)
    hv = _values(heavy)
    print("\n===== native Vessel LLE result =====")
    print("Light_Liquid (organic): flow=%.5f  x=[MIBK %.5f, water %.5f]" % (lv["flow"], lv["z"][0], lv["z"][1]))
    print("Heavy_Liquid (aqueous): flow=%.5f  x=[MIBK %.5f, water %.5f]" % (hv["flow"], hv["z"][0], hv["z"][1]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
