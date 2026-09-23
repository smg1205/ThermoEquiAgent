"""Reflect ForcedPhase enum values and how to force two liquid phases on a stream."""
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
        fs.AddCompound(cand)
        break
    except Exception:
        continue
fs.AddCompound("Water")
ded._add_property_package(fs, "UNIQUAC")
feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
feed = ded._simulation_object(feed_go)

# ForcedPhase enum
print("=== ForcedPhase enum values ===")
try:
    fp = feed.GetType().GetProperty("ForcePhase")
    enum_type = fp.PropertyType
    import System
    print("enum type:", enum_type.FullName)
    for name in System.Enum.GetNames(enum_type):
        print("  ", name)
except Exception as e:
    print("ForcePhase err:", type(e).__name__, e)

# how to write phase estimates / force two liquids -- look for estimates members
print("\n=== members mentioning 'estimate'/'initial'/'liquid' on MaterialStream ===")
for pr in feed.GetType().GetProperties():
    n = pr.Name.lower()
    if any(k in n for k in ("estimate", "initial", "liquid2", "liquid1", "forcephase")):
        try:
            print(f"  {pr.Name} ({pr.PropertyType.Name}) write={pr.CanWrite}")
        except Exception:
            pass
for m in feed.GetType().GetMethods():
    n = m.Name.lower()
    if any(k in n for k in ("estimate", "initial", "liquid", "phase", "force")):
        try:
            print(f"  METHOD {m.Name}({', '.join(p.Name for p in m.GetParameters())})")
        except Exception:
            pass

# PhaseProperties members (what fraction/flow fields exist)
print("\n=== IPhase.Properties (PhaseProperties) members ===")
try:
    liq1 = feed.GetType().GetProperty("Liquid1").GetValue(feed, None)
    props = liq1.Properties
    for pr in props.GetType().GetProperties():
        print(f"  {pr.Name} ({pr.PropertyType.Name})")
except Exception as e:
    print("PhaseProperties err:", type(e).__name__, e)
