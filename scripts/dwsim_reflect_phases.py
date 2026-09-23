"""Reflect the MaterialStream.Phases structure to learn the authoritative multi-phase result."""
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
feed.SetTemperature(298.15)
feed.SetPressure(101325.0)
feed.SetMolarFlow(1.0)
feed.SetOverallComposition(ded._composition_argument([0.4, 0.6]))

print("=== MaterialStream properties with 'phase'/'liquid' in name ===")
for pr in feed.GetType().GetProperties():
    n = pr.Name.lower()
    if any(k in n for k in ("phase", "liquid", "vapor", "fraction", "amount")):
        try:
            val = pr.GetValue(feed, None)
            print(f"  {pr.Name} ({pr.PropertyType.Name}) = {type(val).__name__}")
        except Exception as e:
            print(f"  {pr.Name} ({pr.PropertyType.Name}) = <err {type(e).__name__}>")

print("\n=== Phases property detail ===")
try:
    phases = feed.GetType().GetProperty("Phases").GetValue(feed, None)
    print("Phases type:", type(phases).__name__)
    if phases is not None and hasattr(phases, "Keys"):
        for k in list(phases.Keys):
            v = phases[k]
            print(f"  key={k!r}  value type={type(v).__name__}")
            # introspect the phase object
            for pr in v.GetType().GetProperties():
                print(f"      .{pr.Name} ({pr.PropertyType.Name})")
except Exception as e:
    print("Phases read err:", type(e).__name__, e)

print("\n=== Phase enum / Phase names in Interfaces ===")
try:
    from DWSIM.Interfaces.Enums import Phase as PhaseEnum
    import System
    for name in System.Enum.GetNames(PhaseEnum):
        print("  Phase.", name)
except Exception as e:
    print("Phase enum err:", type(e).__name__, e)
