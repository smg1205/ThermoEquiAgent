"""Print exact ConnectorName order for Vessel Input/Output connectors."""
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

vessel_go = fs.AddObject(object_type.Vessel, 300, 0, "Vessel")
go = vessel_go.GraphicObject

print("=== Output connectors (name : direction : type) ===")
for i, c in enumerate(go.OutputConnectors):
    print(f"  [{i}] {c.ConnectorName!r} dir={c.Direction} type={c.Type} active={c.Active}")

print("\n=== Input connectors ===")
for i, c in enumerate(go.InputConnectors):
    print(f"  [{i}] {c.ConnectorName!r} dir={c.Direction} type={c.Type} active={c.Active}")

# Also check what the feed stream graphic exposes as its input/ouput connectors
feed_go = fs.AddObject(object_type.MaterialStream, 0, -100, "Feed")
fgo = feed_go.GraphicObject
print("\n=== Feed stream Input/Output connectors ===")
for i, c in enumerate(fgo.OutputConnectors):
    print(f"  OUT[{i}] {c.ConnectorName!r}")
for i, c in enumerate(fgo.InputConnectors):
    print(f"  IN[{i}] {c.ConnectorName!r}")

vapor_go = fs.AddObject(object_type.MaterialStream, 600, -80, "Light")
print("\n=== Light stream (product) connectors ===")
for i, c in enumerate(vapor_go.GraphicObject.InputConnectors):
    print(f"  IN[{i}] {c.ConnectorName!r}")
for i, c in enumerate(vapor_go.GraphicObject.OutputConnectors):
    print(f"  OUT[{i}] {c.ConnectorName!r}")
