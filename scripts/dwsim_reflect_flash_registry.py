"""Find how DWSIM resolves a flash-algorithm 'tag' to a real algorithm object.

The Vessel.PreferredFlashAlgorithmTag is a String, but a plain enum NAME may not
be the right key.  We reflect the property-package / calculator for the method
that maps a string tag -> FlashAlgorithm instance, and dump the actual registered
algorithm key->name mapping.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
automation = factory()
fs = automation.CreateFlowsheet()
for cand in ("Methyl isobutyl ketone", "4-Methyl-2-pentanone", "MIBK"):
    try:
        fs.AddCompound(cand); break
    except Exception:
        continue
fs.AddCompound("Water")
ded._add_property_package(fs, "UNIQUAC")

pp = list(fs.PropertyPackages.Values)[0]

# 1) how does Vessel resolve PreferredFlashAlgorithmTag? find any method with 'Flash'/tag/algorithm
print("=== PropertyPackage members mentioning 'FlashAlgorithm'/'Tag'/'GetFlash' ===")
for m in pp.GetType().GetMethods():
    n = m.Name
    if any(k in n for k in ("Flash", "Tag", "Algorithm", "Method")):
        print(f"  METHOD {n}({', '.join(p.Name + ':' + p.ParameterType.Name for p in m.GetParameters())})")

print("\n=== FlashBase (FlashAlgorithm) members ===")
try:
    fb = pp.GetType().GetProperty("FlashBase").GetValue(pp, None)
    print("  FlashBase type:", fb.GetType().FullName)
    for pr in fb.GetType().GetProperties():
        print(f"    .{pr.Name} : {pr.PropertyType.Name}")
except Exception as e:
    print("  FlashBase err:", type(e).__name__, e)

# 2) search assemblies for a registry/dictionary of flash algorithm names
import System
print("\n=== types whose name mentions 'FlashAlgorithms'/registry/enum map ===")
seen = set()
for asm in System.AppDomain.CurrentDomain.GetAssemblies():
    nm = str(asm.GetName().Name)
    if "DWSIM" not in nm:
        continue
    try:
        ts = asm.GetTypes()
    except Exception:
        continue
    for t in ts:
        fn = t.FullName or ""
        if any(k in fn.lower() for k in ("flashalgorithm", "flashmethod", "flashbase")):
            if fn not in seen:
                seen.add(fn)
                print(f"  [{nm}] {fn}")
