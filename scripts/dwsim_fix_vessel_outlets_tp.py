"""Fix: force the Vessel outlet streams to Temperature_and_Pressure (TP) spec,
so re-calculation avoids the unstable GibbsMinimizationMulti Flash_PH path.

Keeps GibbsMinimization (which enables the liquid-liquid split), but the outlet
streams are re-set to TP-flash so they don't run a PH flash on recalc.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

FILE = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle.dwxmz"
OUT = ROOT / "lunwen" / "dwsim_demonstration" / "water_mibk_native_vessel_lle_fixed.dwxmz"


def _find(fs, tag):
    for item in fs.SimulationObjects.Values:
        if str(item.GraphicObject.Tag) == tag:
            return item
    raise KeyError(tag)


def main():
    factory, object_type = ded._automation_factory()
    from System import Enum  # noqa: E402
    automation = factory()
    fs = automation.LoadFlowsheet2(str(FILE))

    # force outlet streams to TP spec
    for tag in ("Light_Liquid", "Heavy_Liquid", "Vapor"):
        s = _find(fs, tag)
        try:
            sp = s.GetType().GetProperty("SpecType")
            enum_type = sp.PropertyType
            sp.SetValue(s, Enum.Parse(enum_type, "Temperature_and_Pressure"), None)
            print(f"[OK] {tag} SpecType -> Temperature_and_Pressure")
        except Exception as e:
            print(f"[WARN] {tag} SpecType: {type(e).__name__} {e}")

    print("recalculating after forcing TP...")
    errors = automation.CalculateFlowsheet4(fs)
    if errors and errors.Count:
        print("RECALC ERRORS:")
        for i in range(errors.Count):
            print("  ", str(errors[i])[:200])
    else:
        print("recalc OK")

    for tag in ("Light_Liquid", "Heavy_Liquid", "Vapor"):
        s = _find(fs, tag)
        f = float(s.GetType().GetMethod("GetMolarFlow").Invoke(s, None))
        print(f"  {tag}: flow={f:.6f}")

    ded._save_flowsheet_via_temp(automation, fs, OUT)
    print("\n[OK] saved:", OUT)


if __name__ == "__main__":
    main()
