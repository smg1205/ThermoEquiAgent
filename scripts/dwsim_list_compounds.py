"""List DWSIM AvailableCompounds names to find ethyl propionate's real key."""
from __future__ import annotations

import sys, os
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))


def main():
    from dotenv import load_dotenv
    load_dotenv()
    import os as _os
    from thermo_engine import dwsim_export as ded
    from dotenv import load_dotenv as ld
    ld()
    factory, object_type = ded._automation_factory()
    automation = factory()
    # AvailableCompounds is a property returning a collection
    try:
        comps = automation.AvailableCompounds
    except Exception as e:
        print("AvailableCompounds access failed:", e)
        return
    names = []
    try:
        for c in comps:
            names.append(str(c))
    except Exception as e:
        print("iter failed:", repr(e))
    print("total compounds:", len(names))
    print("=== names containing 'propionate' / 'Ethyl' / 'ethanol' / 'cyclohexane' ===")
    for n in names:
        low = n.lower()
        if any(k in low for k in ("propion", "ethyl", "ethanol", "cyclohexane", "acetate")):
            print("  ", n)


if __name__ == "__main__":
    main()
