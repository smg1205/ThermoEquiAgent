"""Locate how DWSIM exposes Vapor Fraction / phase fraction on a material stream.

Reflects over the DWSIM Thermodynamics/UnitOperations assemblies to find the
exact property name and (where possible) the result-panel context for the liquid
/vapor molar fractions, so we can say where Vapor Fraction appears in the GUI.
"""
from __future__ import annotations

import sys, os
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))


def main():
    from dotenv import load_dotenv
    load_dotenv()
    home = Path(os.getenv("DWSIM_HOME") or r"C:\Users\34861\AppData\Local\DWSIM")
    import clr, System
    for dll in ("DWSIM.Thermodynamics", "DWSIM.SharedClasses", "DWSIM.UnitOperations",
                "DWSIM.UI.Desktop.Forms", "DWSIM.UI.Desktop", "DWSIM.UI.Desktop.Shared"):
        p = home / (dll + ".dll")
        if p.is_file():
            try:
                clr.AddReference(str(p))
            except Exception as e:
                print(f"{dll}: {e}")
    print("=== Members containing 'VaporFraction' / 'PhaseFraction' / 'OverallVaporFraction' ===")
    seen = set()
    for asm in System.AppDomain.CurrentDomain.GetAssemblies():
        name = str(asm.GetName().Name)
        if "DWSIM" not in name:
            continue
        try:
            for t in asm.GetTypes():
                fn = str(t.FullName)
                tf = t.GetTypeInfo() if hasattr(t, "GetTypeInfo") else None
                try:
                    props = t.GetProperties()
                except Exception:
                    continue
                for pr in props:
                    pn = pr.Name
                    if any(k in pn for k in ("VaporFraction", "PhaseFractions", "OverallVaporFraction",
                                             "GlobalVaporFraction", "VaporMoleFraction", "PhaseMoleFractions")):
                        key = (name, fn, pn)
                        if key not in seen:
                            seen.add(key)
                            print(f"  [{name}] {fn} . {pn}  (type={pr.PropertyType.Name})")
        except Exception:
            pass
    print("\n=== MaterialStream result enum / flash spec names ===")
    for asm in System.AppDomain.CurrentDomain.GetAssemblies():
        name = str(asm.GetName().Name)
        if "DWSIM" not in name:
            continue
        try:
            for t in asm.GetTypes():
                fn = str(t.FullName)
                low = fn.lower()
                if any(k in low for k in ("flashtype", "flashspec", "phasefraction", "state", "varfraction")):
                    print(f"  [{name}] {fn}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
