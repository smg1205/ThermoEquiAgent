"""Reflect the rigorous AbsorptionColumn's flash-kernel + solver configuration members."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded


def main():
    factory, object_type = ded._automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
        try:
            fs.AddCompound(cand); break
        except Exception:
            continue
    fs.AddCompound("Water")
    ded._add_property_package(fs, "UNIFAC-LL")

    col_go = fs.AddObject(object_type.AbsorptionColumn, 400, 0, "Col")
    col = ded._simulation_object(col_go)
    print("column class:", col.GetType().FullName)

    print("\n=== properties with flash/solver/estimate/tolerance/stage ===")
    for pr in col.GetType().GetProperties():
        n = pr.Name.lower()
        if any(k in n for k in ("flash", "solver", "estimate", "toleran", "stage", "iteration",
                                "method", "mode", "initial", "spec")):
            try:
                w = pr.CanWrite
            except Exception:
                w = "?"
            print(f"  {pr.Name} : {pr.PropertyType.Name} (w={w})")

    print("\n=== methods with flash/solver/estimate/column ===")
    for m in col.GetType().GetMethods():
        n = m.Name.lower()
        if any(k in n for k in ("flash", "solver", "estimate", "column", "stage")):
            print(f"  {m.Name}({', '.join(p.Name + ':' + p.ParameterType.Name for p in m.GetParameters())})")

    # available solving-method classes
    print("\n=== available column solving methods (DWSIM.UnitOperations) ===")
    import System
    for asm in System.AppDomain.CurrentDomain.GetAssemblies():
        nm = str(asm.GetName().Name)
        if "DWSIM.UnitOperations" not in nm:
            continue
        try:
            for t in asm.GetTypes():
                fn = t.FullName or ""
                if "SepOps.SolvingMethods" in fn and t.IsClass:
                    print("   ", fn)
        except Exception:
            pass


if __name__ == "__main__":
    main()
