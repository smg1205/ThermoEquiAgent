"""Diagnose why the saved Vessel LLE file goes through Flash_PH (PH flash) and fails.

Loads the generated .dwxmz and prints the Vessel's CalculationMode + each stream's
SpecType, so we can see whether a PH-flash path is being taken instead of TP.
"""
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
    print("loaded:", FILE)

    for item in fs.SimulationObjects.Values:
        tag = str(item.GraphicObject.Tag)
        try:
            calc_mode = item.GetType().GetProperty("CalculationMode")
            cm = calc_mode.GetValue(item, None) if calc_mode else None
        except Exception:
            cm = "?"
        try:
            spec = item.GetType().GetProperty("SpecType")
            st = spec.GetValue(item, None) if spec else None
        except Exception:
            st = "?"
        print(f"  obj={tag:16s} class={item.GetType().Name:18s} CalcMode={cm!r} SpecType={st!r}")


if __name__ == "__main__":
    main()
