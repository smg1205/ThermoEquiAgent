"""Reflect how a DWSIM DistillationColumn accepts feed-stage information."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thermo_engine.dwsim_export import _automation_factory, _add_property_package, _simulation_object  # noqa: E402


def main():
    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    fs.AddCompound("Isopropanol")
    fs.AddCompound("Water")
    _add_property_package(fs, "UNIQUAC")

    col_go = fs.AddObject(object_type.DistillationColumn, 250, 0, "Col")
    col = _simulation_object(col_go)

    print("=== methods mentioning Feed/Stage/Connect ===")
    for m in col.GetType().GetMethods():
        n = m.Name
        if any(k in n for k in ("Feed", "Stage", "Connect", "Spec")):
            try:
                params = ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in m.GetParameters())
            except Exception:
                params = "?"
            print(f"  {n}({params})")

    print("\n=== properties mentioning Feed/Stage/Stages/Port ===")
    for pr in col.GetType().GetProperties():
        n = pr.Name
        if any(k in n for k in ("Feed", "Stage", "Port", "Spec")):
            try:
                w = pr.CanWrite
            except Exception:
                w = "?"
            print(f"  {n} : {pr.PropertyType.Name} (w={w})")

    print("\n=== Stages list element type / fields ===")
    try:
        stages = col.GetType().GetProperty("Stages").GetValue(col, None)
        print("  Stages count:", len(stages))
        if len(stages):
            st = stages[0]
            print("  stage type:", st.GetType().FullName)
            for f in st.GetType().GetFields():
                print(f"    FIELD {f.Name} : {f.FieldType.Name}")
            for p in st.GetType().GetProperties():
                print(f"    PROP  {p.Name} : {p.PropertyType.Name}")
    except Exception as e:
        print("  err:", type(e).__name__, e)


if __name__ == "__main__":
    main()
