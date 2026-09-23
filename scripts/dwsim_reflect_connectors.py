"""Dump VesselGraphic Input/Output connectors (names + indices) to fix ConnectObjects."""
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

def dump_connectors(listattr, label):
    try:
        conns = getattr(go, listattr)
    except Exception as e:
        print(f"{label}: err {type(e).__name__}")
        return
    print(f"{label}: {len(conns)} connectors")
    for i, c in enumerate(conns):
        name = getattr(c, "Name", getattr(c, "Tag", "?"))
        connected = getattr(c, "IsAttached", getattr(c, "Connected", "?"))
        print(f"   [{i}] name={name!r} type={c.GetType().Name} attached={connected}")

dump_connectors("InputConnectors", "InputConnectors")
dump_connectors("OutputConnectors", "OutputConnectors")
try:
    e = go.EnergyConnector
    print(f"EnergyConnector: {e!r} type={type(e).__name__}")
except Exception as ex:
    print("EnergyConnector err:", type(ex).__name__)

# raw ConnectObjects signature on flowsheet
print("\n=== flowsheet ConnectObjects signatures ===")
for m in fs.GetType().GetMethods():
    if m.Name == "ConnectObjects":
        print("  ConnectObjects(", ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in m.GetParameters()), ")")
