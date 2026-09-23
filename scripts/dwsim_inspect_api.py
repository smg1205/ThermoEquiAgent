"""Introspect the installed DWSIM Automation API so we can call it correctly.

Prints the public methods available on:
  - the Automation entry,
  - a created Flowsheet,
  - a Material Stream simulation object.

This lets us find the *actual* solve / flash / read method names for the
installed DWSIM version instead of guessing.

Run (real terminal):
    conda activate thermo
    python scripts/dwsim_inspect_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))


def _methods(obj, prefix=""):
    names = sorted({n for n in dir(obj) if not n.startswith("_")})
    return [n for n in names]


def main():
    from thermo_engine import dwsim_export as ded

    factory, object_type = ded._automation_factory()
    automation = factory()

    print("=== Automation object method count ===")
    am = _methods(automation)
    print(f"{len(am)} non-underscore members")
    for n in am:
        print(f"  {n}")

    print("\n=== ObjectType members (unit ops) ===")
    otm = _methods(object_type)
    interesting = [n for n in otm if any(k in n for k in
        ("Stream", "Vessel", "Flash", "Separator", "RTD", "Converter", "CapHeatExchanger", "Material"))]
    print("filtered:", interesting)

    fs = automation.CreateFlowsheet()
    print("\n=== Flowsheet members ===")
    fm = _methods(fs)
    for n in fm:
        if any(k in n.lower() for k in ("calcul", "solve", "run", "save", "object", "compound", "simulation", "refresh")):
            print(f"  {n}")
    print(f"(total {len(fm)} members)")

    # try to create a stream and inspect it
    try:
        obj = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    except Exception as e:
        print("AddObject(MaterialStream) failed:", e)
        return
    print("\n=== MaterialStream graphic object ===")
    print("str:", str(obj)[:200])
    for attr in ("GetAsObject", "GetSimulationObject"):
        if hasattr(obj, attr):
            try:
                inner = getattr(obj, attr)()
                print(f"  via {attr}(): members:")
                for n in _methods(inner):
                    if any(k in n.lower() for k in ("temperat", "press", "compos", "fraction", "molar", "enthal", "flash")):
                        print(f"     {n}")
            except Exception as e:
                print(f"  {attr}() failed: {e}")


if __name__ == "__main__":
    main()
