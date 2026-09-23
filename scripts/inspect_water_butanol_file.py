"""Inspect the saved water/butanol file: two-liquid split, compositions, stability."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "butanol_test" / "water_butanol_NRTL_default.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))

    def find(tag):
        for it in fs.SimulationObjects.Values:
            if str(it.GraphicObject.Tag) == tag:
                return it
        return None

    def v(tag):
        o = find(tag)
        if o is None:
            return None
        t = o.GetType()
        return (float(t.GetMethod("GetMolarFlow").Invoke(o, None)),
                [float(x) for x in t.GetMethod("GetOverallComposition").Invoke(o, None)])

    print("=== first calculation on load ===")
    automation.CalculateFlowsheet4(fs)

    feed = find("Feed")
    print("\nFeed phase state:")
    for nm in ("Liquid1", "Liquid2", "Vapor"):
        try:
            p = feed.GetType().GetProperty(nm).GetValue(feed, None)
        except Exception:
            p = None
        if p is None:
            print(f"  {nm}: absent")
            continue
        try:
            fr = float(p.Properties.molarfraction)
        except Exception:
            fr = None
        comp = None
        try:
            comp = [float(c.MoleFraction) for c in p.Compounds.Values]
        except Exception:
            pass
        print(f"  {nm}: fraction={fr}  composition(butanol, water)={comp}")
    try:
        print("  PhaseIds:", list(feed.PhaseIds) if feed.PhaseIds else None)
    except Exception:
        print("  PhaseIds: unreadable")

    print("\nOutlets:")
    for tag in ("Light_Liquid", "Heavy_Liquid", "Vapor"):
        r = v(tag)
        if r:
            print(f"  {tag:14s} flow={r[0]:.5f} z=[butanol {r[1][0]:.5f}, water {r[1][1]:.5f}]")

    print("\n--- mass balance ---")
    fr = v("Feed")
    tot_b = fr[0] * fr[1][0]
    tot_w = fr[0] * fr[1][1]
    ob = ow = 0.0
    for tag in ("Light_Liquid", "Heavy_Liquid", "Vapor"):
        r = v(tag)
        ob += r[0] * r[1][0]
        ow += r[0] * r[1][1]
    print(f"  butanol: in={tot_b:.5f} out={ob:.5f} diff={tot_b-ob:+.2e}")
    print(f"  water  : in={tot_w:.5f} out={ow:.5f} diff={tot_w-ow:+.2e}")

    print("\n--- recalculation stability ---")
    e = automation.CalculateFlowsheet4(fs)
    print("  recalc errors:", (str(e[0])[:80] if e and e.Count else "none"))
    for tag in ("Light_Liquid", "Heavy_Liquid"):
        r = v(tag)
        print(f"  {tag:14s} flow={r[0]:.5f} z=[butanol {r[1][0]:.5f}, water {r[1][1]:.5f}]")


if __name__ == "__main__":
    main()
