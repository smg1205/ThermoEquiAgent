"""Probe which flash kernel works for ethanol/ethyl acetate/water ternary LLE."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
Z = [0.20, 0.30, 0.50]


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
    from System import Activator, Array, Double  # noqa: E402
    automation = factory()
    fs = automation.CreateFlowsheet()
    fs.AddCompound("Ethanol")
    fs.AddCompound("Ethyl acetate")
    fs.AddCompound("Water")
    ded._add_property_package(fs, "NRTL")
    pp = list(fs.PropertyPackages.Values)[0]

    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    feed = ded._simulation_object(feed_go)
    feed.SetTemperature(T); feed.SetPressure(P); feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(ded._composition_argument(Z))
    pp.GetType().GetProperty("CurrentMaterialStream").SetValue(pp, feed, None)

    asm = pp.GetType().Assembly
    for cls in ("NestedLoops3PV3", "GibbsMinimizationMulti", "NestedLoopsImmiscible", "SimpleLLE", "NestedLoopsSVLLE"):
        ft = asm.GetType(f"DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.{cls}")
        if ft is None:
            print(f"[skip] {cls}"); continue
        try:
            fl = Activator.CreateInstance(ft)
            try:
                fl.StabSearchSeverity = 3
            except Exception:
                pass
            r = fl.Flash_PT(Array[Double](Z), P, T, pp, False, None)
            dump(r, f"NRTL {cls}.Flash_PT")
        except Exception as e:
            print(f"[ERR] {cls}: {type(e).__name__} {str(e)[:100]}")

    # also try just running the stream's own Calculate (UniversalFlash path)
    print("\n--- stream Calculate (default path) ---")
    try:
        feed.Calculate(True, True)
        for nm in ("Liquid1", "Liquid2"):
            p = feed.GetType().GetProperty(nm).GetValue(feed, None)
            if p is not None:
                print(f"  {nm}: molarfraction={p.Properties.molarfraction}")
        print("  PhaseIds:", list(feed.PhaseIds) if feed.PhaseIds else None)
        print("  z:", [float(v) for v in feed.GetOverallComposition()])
    except Exception as e:
        print("  Calculate err:", type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    main()
