
"""Reflect DWSIM assemblies for real liquid-liquid extraction unit operations.

Run from ThermoAgent root:
    python scripts/reflect_dwsim_lle_units.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
dwsim_home = os.getenv("DWSIM_HOME")
if not dwsim_home:
    raise SystemExit("DWSIM_HOME is not set")
install_dir = Path(dwsim_home).expanduser().resolve()
os.environ["TEMP"] = str((Path.cwd() / ".tmp" / "dwsim-reflect").resolve())
os.environ["TMP"] = os.environ["TEMP"]
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
sys.path.append(str(install_dir))

import clr  # type: ignore
from System import AppDomain, Type, Activator, Enum  # type: ignore

# Load the usual DWSIM assemblies plus common unit-op assemblies if present.
for dll_name in [
    "DWSIM.Automation.dll",
    "DWSIM.Interfaces.dll",
    "DWSIM.UnitOperations.dll",
    "DWSIM.UnitOperations.Auxiliary.dll",
    "DWSIM.Thermodynamics.dll",
    "DWSIM.SharedClasses.dll",
]:
    dll = install_dir / dll_name
    if dll.exists():
        try:
            clr.AddReference(str(dll))
            print(f"LOADED {dll_name}")
        except Exception as exc:
            print(f"LOAD-ERR {dll_name}: {type(exc).__name__}: {exc}")
    else:
        print(f"MISSING {dll_name}")

from DWSIM.Automation import Automation3  # type: ignore
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType  # type: ignore

print("\n== ObjectType enum names containing extraction-ish terms ==")
terms = ["extract", "decan", "liquid", "separator", "split", "column", "absor"]
for name in Enum.GetNames(ObjectType):
    if any(t in name.lower() for t in terms):
        print(name)

print("\n== Loaded DWSIM types containing extraction-ish terms ==")
type_terms = [
    "extract", "extraction", "extractor", "decanter", "liquidliquid",
    "liquid_liquid", "ll", "separator", "componentseparator", "splitter",
]
all_matches = []
for asm in AppDomain.CurrentDomain.GetAssemblies():
    try:
        types = asm.GetTypes()
    except Exception:
        continue
    for typ in types:
        full = typ.FullName or ""
        low = full.lower()
        if typ.IsClass and any(term in low for term in type_terms):
            all_matches.append(full)
for full in sorted(set(all_matches)):
    print(full)

print("\n== UnitOperations namespace classes, selected constructors/properties ==")
selected = [
    full for full in sorted(set(all_matches))
    if "DWSIM.UnitOperations" in full or "DWSIM.UnitOperations.UnitOperations" in full
]
for full in selected:
    typ = Type.GetType(full)
    if typ is None:
        for asm in AppDomain.CurrentDomain.GetAssemblies():
            typ = asm.GetType(full)
            if typ is not None:
                break
    if typ is None:
        continue
    print(f"\nTYPE {full}")
    try:
        ctors = typ.GetConstructors()
        for ctor in ctors[:5]:
            params = ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in ctor.GetParameters())
            print(f"  CTOR({params})")
    except Exception as exc:
        print(f"  CTOR-ERR {type(exc).__name__}: {exc}")
    try:
        props = []
        for prop in typ.GetProperties():
            n = prop.Name
            if any(t in n.lower() for t in ["phase", "liquid", "split", "sep", "stage", "feed", "outlet", "stream", "spec", "component"]):
                props.append(f"{n}:{prop.PropertyType.Name}:write={prop.CanWrite}")
        for item in props[:40]:
            print(f"  PROP {item}")
    except Exception as exc:
        print(f"  PROP-ERR {type(exc).__name__}: {exc}")

print("\n== Try Automation AddObject for plausible ObjectType names ==")
automation = Automation3()
fs = automation.CreateFlowsheet()
for candidate in [
    "LiquidLiquidExtractor", "LiquidLiquidExtractionColumn", "ExtractionColumn",
    "LiquidLiquidColumn", "Decanter", "Separator3Phase", "Separator", "ComponentSeparator",
    "Splitter", "Vessel", "TPVessel", "AbsorptionColumn",
]:
    member = getattr(ObjectType, candidate, None)
    if member is None:
        print(f"NO-ENUM {candidate}")
        continue
    try:
        obj = fs.AddObject(member, 100, 100, candidate)
        sim = obj.GetAsObject() if hasattr(obj, "GetAsObject") else obj
        print(f"ADD-OK {candidate}: {sim.GetType().FullName}")
    except Exception as exc:
        print(f"ADD-ERR {candidate}: {type(exc).__name__}: {exc}")
