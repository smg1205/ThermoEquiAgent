"""Inspect UniversalFlash: its AlgoType (FlashMethod enum) and how to route to NestedLoopsImmiscible."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermo_engine import dwsim_export as ded

factory, object_type = ded._automation_factory()
import System  # noqa: E402  (pythonnet injects System after the assembly is loaded)
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

fb = pp.GetType().GetProperty("FlashBase").GetValue(pp, None)
print("FlashBase:", fb.GetType().FullName)

print("\n=== UniversalFlash members (name : type) ===")
for pr in fb.GetType().GetProperties():
    print(f"  {pr.Name} : {pr.PropertyType.Name}")

print("\n=== UniversalFlash methods mentioning algo/route/dispatch ===")
for m in fb.GetType().GetMethods():
    n = m.Name
    if any(k in n.lower() for k in ("algo", "route", "dispatch", "select", "switch", "method", "list", "available")):
        print(f"  {n}({', '.join(p.Name for p in m.GetParameters())})")

# AlgoType is FlashMethod (Thermodynamics enum). dump its values
print("\n=== FlashMethod (Thermodynamics) enum values ===")
atype = fb.GetType().GetProperty("AlgoType")
t = atype.PropertyType
print("  AlgoType property type:", t.FullName)
if t.IsEnum:
    for name in System.Enum.GetNames(t):
        print("    ", name)

# what is the current AlgoType / Name / Tag?
print("\n=== current FlashBase state ===")
for attr in ("AlgoType", "Name", "Tag", "Description", "Order"):
    try:
        v = getattr(fb, attr)
        print(f"  {attr} = {v!r}")
    except Exception as e:
        print(f"  {attr} = <err {type(e).__name__}>")
