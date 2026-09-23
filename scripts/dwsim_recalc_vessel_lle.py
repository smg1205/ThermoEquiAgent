"""Reproduce the PH-flash failure by reloading and recalculating the saved Vessel LLE file."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))
    print("loaded and recalculating...")
    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("RECALC ERRORS:")
        for i in range(errors.Count):
            print("  ", str(errors[i])[:300])
    else:
        print("recalc OK (no errors)")

    # read the outlet flows after recalc
    for tag in ("Light_Liquid", "Heavy_Liquid", "Vapor"):
        for item in fs.SimulationObjects.Values:
            if str(item.GraphicObject.Tag) == tag:
                f = float(item.GetType().GetMethod("GetMolarFlow").Invoke(item, None))
                print(f"  {tag}: flow={f:.6f}")


if __name__ == "__main__":
    main()
