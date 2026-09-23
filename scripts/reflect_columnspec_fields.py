"""List every field of a ColumnSpec so we can pick a valid spectype string."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory  # noqa: E402

OUT = ROOT / "report" / "dwsim"
NAME = "ipa_water_binary_column_x0p5_unifac.dwxmz"


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.LoadFlowsheet2(str(OUT / NAME))
    column = None
    for item in fs.SimulationObjects.Values:
        if item.GetType().Name in ("DistillationColumn", "AbsorptionColumn"):
            column = item.GetAsObject() if hasattr(item, "GetAsObject") else item

    specs = column.Specs
    s = specs["R"]
    print("=== ColumnSpec full member list ===")
    for p in s.GetType().GetProperties():
        try:
            v = p.GetValue(s, None)
        except Exception as e:
            v = f"<{type(e).__name__}>"
        print(f"  PROP  {p.Name:24s} : {p.PropertyType.Name:20s} = {v!r}")

    print("\n=== ColumnSpec methods ===")
    for m in s.GetType().GetMethods():
        if not m.Name.startswith("get_") and not m.Name.startswith("set_"):
            print(f"  {m.Name}({', '.join(pp.ParameterType.Name + ' ' + pp.Name for pp in m.GetParameters())})")


if __name__ == "__main__":
    main()
