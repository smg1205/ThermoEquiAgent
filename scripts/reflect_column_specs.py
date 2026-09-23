"""Reflect the ColumnSpec objects so the reboiler (bottom) spec can be set properly."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

OUT = ROOT / "report" / "dwsim"
NAME = "ipa_water_binary_column_x0p5_unifac.dwxmz"


def dump(obj, label, depth=0):
    ind = "  " * depth
    print(f"{ind}{label}: {obj.GetType().FullName}")
    for p in obj.GetType().GetProperties():
        try:
            v = p.GetValue(obj, None)
        except Exception:
            v = "<err>"
        if p.Name in ("TypeId", "SpecType", "SpecValue", "Name", "IsActive", "Units", "Compound",
                      "TotalOrPartial", "SpecValue2"):
            print(f"{ind}  .{p.Name} = {v!r}  ({p.PropertyType.Name})")


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(OUT / NAME))
    column = None
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
            column = item.GetAsObject() if hasattr(item, "GetAsObject") else item
    print("=== available spec types on this column ===")
    # what enum does SpecType use?
    specs = column.Specs
    for k in list(specs.Keys):
        s = specs[k]
        dump(s, f"Spec[{k}]")
        print()

    # enumerate the specification type enum
    try:
        p = None
        for k in list(specs.Keys):
            p = specs[k].GetType().GetProperty("SpecType")
            break
        if p is not None:
            import System
            et = p.PropertyType
            print("=== SpecType enum values ===")
            for n in System.Enum.GetNames(et):
                print("  ", n)
    except Exception as e:
        print("enum err:", type(e).__name__, e)

    print("\n=== SetReboilerSpec / SetCondenserSpec signatures ===")
    for m in column.GetType().GetMethods():
        if "Spec" in m.Name and "Set" in m.Name:
            print(f"  {m.Name}({', '.join(pp.Name + ':' + pp.ParameterType.Name for pp in m.GetParameters())})")


if __name__ == "__main__":
    main()
