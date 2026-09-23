"""Direct PT flash with UNIFAC-LL for MIBK/water -- does the group method predict a split?"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.4, 0.6]


def dump(r, tag):
    print(f"--- {tag} ---")
    for i in range(len(r)):
        v = r[i]
        try:
            s = f"[{', '.join(f'{float(x):.5f}' for x in v)}]" if hasattr(v, "__iter__") and not isinstance(v, str) else repr(v)
        except Exception:
            s = repr(v)
        print(f"  result[{i}] = {s}")


def main():
    factory, object_type = ded._automation_factory()
    from System import Activator, Array, Double, Enum  # noqa: E402
    from DWSIM.Interfaces.Enums import FlashSetting  # noqa: E402
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
            ded._add_property_package(fs, p)
            pkg = p; break
        except Exception:
            continue
    print("[OK] package:", pkg)
    pp = list(fs.PropertyPackages.Values)[0]

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    pp.GetType().GetProperty("CurrentMaterialStream").SetValue(pp, feed, None)

    asm = pp.GetType().Assembly
    for cls in ("NestedLoops3PV3", "GibbsMinimizationMulti", "NestedLoopsImmiscible", "SimpleLLE"):
        try:
            ft = asm.GetType(f"DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.{cls}")
            if ft is None:
                print(f"[skip] {cls} not found"); continue
            fl = Activator.CreateInstance(ft)
            try:
                fl.StabSearchSeverity = 3
            except Exception:
                pass
            r = fl.Flash_PT(Array[Double](Z), P, T, pp, False, None)
            dump(r, f"UNIFAC-LL {cls}.Flash_PT @298.15K")
        except Exception as e:
            print(f"[ERR] {cls}: {type(e).__name__} {str(e)[:120]}")


if __name__ == "__main__":
    main()
