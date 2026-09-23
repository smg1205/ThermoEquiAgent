"""Try DWSIM's native TPVessel (T-P flash vessel) for MIBK/water LLE with UNIFAC-LL.

TPVessel performs a Temperature-Pressure flash -- exactly the TP path that the
NestedLoopsImmiscible kernel needs, unlike the plain Vessel whose outlet streams
force a PH flash.  This is a fully native unit (no CustomUO script).
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
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_tpvessel_lle.dwxmz"


def _vals(s):
    return {
        "flow": float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None)),
        "z": [float(v) for v in s.GetType().GetMethod("GetOverallComposition").Invoke(s, None)],
    }


def main():
    factory, object_type = ded._automation_factory()
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

    # inspect TPVessel ports first
    tp = getattr(object_type, "TPVessel", None)
    if tp is None:
        print("[FATAL] TPVessel not in ObjectType")
        return

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    tp_go = fs.AddObject(tp, 350, 0, "TPVessel_LL")
    tp_obj = ded._simulation_object(tp_go)
    print("TPVessel ports:")
    for p in tp_obj.GetConnectionPortsList():
        print("   ", p)

    # create one product stream per outlet port
    outs = []
    for i in range(4):
        go = fs.AddObject(object_type.MaterialStream, 750, -100 + i * 70, f"Out_{i}")
        outs.append(go)

    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    try:
        tp_obj.FlashTemperature = T
        tp_obj.FlashPressure = P
    except Exception as e:
        print("[WARN] T/P:", type(e).__name__)

    # connect feed -> inlet 0
    try:
        fs.ConnectObjects(feed_go.GraphicObject, tp_go.GraphicObject, 0, 0)
        print("[OK] feed -> TPVessel")
    except Exception as e:
        print("[WARN] feed conn:", type(e).__name__)

    # try each outlet index 0..3
    for i, go in enumerate(outs):
        try:
            fs.ConnectObjects(tp_go.GraphicObject, go.GraphicObject, i, 0)
            print(f"[OK] TPVessel outlet {i} -> Out_{i}")
        except Exception as e:
            print(f"[WARN] outlet {i}: {type(e).__name__}")

    e = automation.CalculateFlowsheet4(fs)
    if e and e.Count:
        print("calc errors:", str(e[0])[:200])

    print("\n===== TPVessel results =====")
    for i, go in enumerate(outs):
        v = _vals(ded._simulation_object(go))
        print(f"  Out_{i}: flow={v['flow']:.5f} z=[MIBK {v['z'][0]:.5f}, water {v['z'][1]:.5f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
