"""Inspect the CreateFlowsheet overloads on the installed DWSIM Automation3."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))


def main():
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    m = getattr(automation, "CreateFlowsheet")
    print("CreateFlowsheet:", m)
    # pythonnet exposes .Overloads on the method object
    try:
        ov = m.Overloads
        print("has Overloads attr:", ov)
    except Exception as e:
        print("no .Overloads:", e)
    # Try a couple of DWSIM-common signatures
    tries = [
        ("CreateFlowsheet()", lambda: automation.CreateFlowsheet()),
        ("CreateFlowsheet(True)", lambda: automation.CreateFlowsheet(True)),
        ("CreateFlowsheet(False)", lambda: automation.CreateFlowsheet(False)),
        ("CreateFlowsheet2()", lambda: automation.CreateFlowsheet2()),
    ]
    for label, fn in tries:
        try:
            r = fn()
            print(f"  OK  {label} -> {type(r).__name__}")
        except Exception as e:
            print(f"  FAIL {label}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
