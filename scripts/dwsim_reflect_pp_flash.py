"""Reflect the UNIQUAC property package's flash-algorithm-setting members."""
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

pps = list(fs.PropertyPackages.Values)
pp = pps[0]
print("PP type:", pp.GetType().FullName)

print("\n=== PP properties whose name has 'flash'/'method'/'algorithm'/'approach' ===")
for pr in pp.GetType().GetProperties():
    n = pr.Name.lower()
    if any(k in n for k in ("flash", "method", "algorithm", "approach", "immiscib", "stab")):
        try:
            val = pr.GetValue(pp, None)
            print(f"  {pr.Name} : {pr.PropertyType.Name} = {repr(val)[:80]}")
        except Exception as e:
            print(f"  {pr.Name} : {pr.PropertyType.Name} = <err {type(e).__name__}>")

print("\n=== FlashSettings dict current contents ===")
try:
    sp = pp.GetType().GetProperty("FlashSettings")
    settings = sp.GetValue(pp, None)
    print("  FlashSettings type:", type(settings).__name__)
    for k in list(settings.Keys):
        print(f"    {k} = {settings[k]!r}")
except Exception as e:
    print("  FlashSettings err:", type(e).__name__, e)
