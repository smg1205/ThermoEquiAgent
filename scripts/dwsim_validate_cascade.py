"""Validate the 2-stage native Vessel cascade: mass balance + try to clear the
Recycle PH-flash warning by seeding the recycle stream."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

T = 298.15
P = 101325.0
FEED_Z = [0.4, 0.6]
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_vessel_cascade_2stage.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(OUT))
    pp = list(fs.PropertyPackages.Values)[0]

    def find(tag):
        for it in fs.SimulationObjects.Values:
            if str(it.GraphicObject.Tag) == tag:
                return it
        return None

    def vals(o):
        t = o.GetType()
        return (float(t.GetMethod("GetMolarFlow").Invoke(o, None)),
                [float(v) for v in t.GetMethod("GetOverallComposition").Invoke(o, None)])

    # re-run once and report
    e = automation.CalculateFlowsheet4(fs)
    print("recalc errors:", (str(e[0])[:120] if e and e.Count else "none"))

    print("\n--- streams ---")
    for tag in ("Feed", "Solvent", "Stage_1_light", "Stage_1_heavy",
                "Stage_2_light", "Stage_2_heavy"):
        o = find(tag)
        if o is None:
            print(f"  {tag}: NOT FOUND"); continue
        f, z = vals(o)
        print(f"  {tag:16s} flow={f:.5f} z=[MIBK {z[0]:.5f}, water {z[1]:.5f}]")

    print("\n--- mass balance check ---")
    fF, zF = vals(find("Feed"))
    fS, zS = vals(find("Solvent"))
    in_mibk = fF * zF[0] + fS * zS[0]
    in_water = fF * zF[1] + fS * zS[1]
    # products: stage2 light (organic product) + stage1 heavy (aqueous product)
    fL, zL = vals(find("Stage_2_light"))
    fH, zH = vals(find("Stage_1_heavy"))
    out_mibk = fL * zL[0] + fH * zH[0]
    out_water = fL * zL[1] + fH * zH[1]
    print(f"  MIBK  in={in_mibk:.5f}  out={out_mibk:.5f}  diff={in_mibk-out_mibk:+.5f}")
    print(f"  water in={in_water:.5f}  out={out_water:.5f}  diff={in_water-out_water:+.5f}")


if __name__ == "__main__":
    main()
