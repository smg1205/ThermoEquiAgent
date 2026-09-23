"""Per-stage mass balance audit for the open cascade, to locate the closure gap."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_cascade_open.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))
    automation.CalculateFlowsheet4(fs)

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

    print("--- per-stage component flows (mol/s) ---")
    names = ("MIBK", "water")
    for i in range(1, 4):
        feed_in = v("Feed") if i == 1 else v(f"Organic_Product_{i-1}")
        sol = v(f"Solvent_{i}")
        org = v(f"Organic_Product_{i}")
        aq = v(f"Aqueous_Extract_{i}")
        print(f"\nStage {i}:")
        for j, nm in enumerate(names):
            fin = (feed_in[0] * feed_in[1][j]) if feed_in else 0.0
            sin = sol[0] * sol[1][j]
            fout = org[0] * org[1][j] + aq[0] * aq[1][j]
            print(f"  {nm:5s} in = {fin:.5f} (organic feed) + {sin:.5f} (solvent) = {fin+sin:.5f}"
                  f"   out = {fout:.5f}   diff = {fin+sin-fout:+.5f}")


if __name__ == "__main__":
    main()
