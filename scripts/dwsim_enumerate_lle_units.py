"""Enumerate DWSIM's native unit operations that can do LLE, with UNIFAC-LL.

Goal: find a NATIVE (non-CustomUO) unit that splits MIBK/water into two liquid
phases without a script.  Candidates:
  - AbsorptionColumn with OperationMode = Extractor (GUI: Liquid-Liquid Extractor)
  - Vessel (Light/Heavy liquid outlets) with every flash-approach setting
  - 3-phase capable separators if present
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded


def main():
    factory, object_type = ded._automation_factory()
    import System
    print("=== ObjectType enum names (all) ===")
    for name in System.Enum.GetNames(object_type):
        print("  ", name)

    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIFAC-LL")

    # probe candidate native units and their connection ports
    for cand in ("AbsorptionColumn", "Vessel", "ComponentSeparator", "Splitter", "Heater", "Mixer"):
        member = getattr(object_type, cand, None)
        if member is None:
            print(f"\n[NO-ENUM] {cand}")
            continue
        try:
            go = fs.AddObject(member, 100, 100, cand)
            sim = go.GetAsObject() if hasattr(go, "GetAsObject") else go
            print(f"\n[OK] {cand} -> {sim.GetType().Name}")
            ports = sim.GetConnectionPortsList()
            for p in ports:
                print("     ", p)
        except Exception as e:
            print(f"\n[ERR] {cand}: {type(e).__name__} {str(e)[:90]}")


if __name__ == "__main__":
    main()
