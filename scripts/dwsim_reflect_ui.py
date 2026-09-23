"""Locate the exact DWSIM GUI container that hosts bubble-point / phase-envelope.

Uses python reflection over the DWSIM UI/thermo assemblies (without needing the
GUI to be open) to find which form/menu/panel owns the Phase Envelope and Bubble
Temperature utilities, so we can give a definitive GUI path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(r"E:\PythonProject\ThermoEqui-Agent-main-3")
sys.path.insert(0, str(_REPO))


def main():
    from thermo_engine import dwsim_export as ded
    from dotenv import load_dotenv
    load_dotenv()
    import os
    home = Path(os.getenv("DWSIM_HOME") or r"C:\Users\34861\AppData\Local\DWSIM")
    import clr
    for dll in ("DWSIM.UI.Desktop", "DWSIM.UI.Desktop.Forms", "DWSIM.UI.Desktop.Shared", "DWSIM.SharedClasses"):
        p = home / (dll + ".dll")
        if p.is_file():
            try:
                clr.AddReference(str(p))
                print(f"loaded {dll}")
            except Exception as e:
                print(f"loaded {dll}: {e}")
    import System
    # enumerate types whose FullName mentions forms / envelopes / utilities
    found = []
    for asm in AppDomain_CurrentDomain().GetAssemblies() if False else System.AppDomain.CurrentDomain.GetAssemblies():
        name = str(asm.GetName().Name)
        if not any(k in name for k in ("DWSIM",)):
            continue
        try:
            for t in asm.GetTypes():
                fn = str(t.FullName)
                low = fn.lower()
                if any(k in low for k in ("phaseenvelope", "envelope", "bubble", "utilities", "utilityform", "utility", "preferenceswindow")):
                    found.append((name, fn))
        except Exception:
            pass
    print(f"\n=== types mentioning envelope/bubble/utility ({len(found)}) ===")
    for name, fn in found[:60]:
        print(f"  [{name}] {fn}")


def AppDomain_CurrentDomain():
    import System
    return System.AppDomain.CurrentDomain


if __name__ == "__main__":
    main()
