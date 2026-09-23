"""Determine which candidate components DWSIM can actually AddCompound.

Uses the authoritative test: try AddCompound(name) on a fresh flowsheet and see
whether it throws. Also substring-searches AvailableCompounds for clues.
"""
from __future__ import annotations

import sys, os
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))

CANDIDATES = [
    # (try_name, display_label)
    ("Ethanol", "Ethanol"),
    ("Cyclohexane", "Cyclohexane"),
    ("Ethyl propionate", "Ethyl propionate"),
    ("Ethyl propanoate", "Ethyl propanoate"),
    ("Propionic acid", "Propionic acid"),
    ("Propyl acetate", "Propyl acetate"),
    ("Methyl propionate", "Methyl propionate"),
    ("Acetone", "Acetone"),
    ("Isooctane", "Isooctane"),
    ("Tetrahydropyran", "Tetrahydropyran"),
    ("Isopropanol", "Isopropanol"),
    ("Hexane", "Hexane"),
    ("Methanol", "Methanol"),
    ("n-propanol", "n-propanol"),
    ("Water", "Water"),
    ("Ethyl acetate", "Ethyl acetate"),
    ("Acetonitrile", "Acetonitrile"),
    ("Benzene", "Benzene"),
    ("1,4-dioxane", "1,4-dioxane"),
    ("N-propyl acetate", "N-propyl acetate"),
    ("Cis-1,2-dimethylcyclopentane", "1,2-DMCP"),
]


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from thermo_engine import dwsim_export as ded
    factory, object_type = ded._automation_factory()
    automation = factory()

    # gather available name strings (substring clues)
    comps = automation.AvailableCompounds
    allstrs = [str(c) for c in comps]
    print("== AddCompound probe ==")
    for name, label in CANDIDATES:
        fs = automation.CreateFlowsheet()
        try:
            fs.AddCompound(name)
            # confirm via check
            try:
                fs.AddCompound(name); second = "dup-ok"
            except Exception:
                second = "dup-rejected"
            print(f"  {label:22s} '{name}': ADD-OK ({second})")
        except Exception as e:
            print(f"  {label:22s} '{name}': FAIL ({type(e).__name__})")
        # free
    print("\n== substring hints in AvailableCompounds str() ==")
    for key in ["Ethanol", "Cyclohexane", "Propionate", "Propanoate", "Propionic", "Propanol", "Acetate", "Acetone", "Hexane"]:
        hits = [s for s in allstrs if key in s]
        print(f"  contains '{key}': {len(hits)} hit(s)")
        for h in hits[:3]:
            print(f"      {h[:90]}")


if __name__ == "__main__":
    main()
