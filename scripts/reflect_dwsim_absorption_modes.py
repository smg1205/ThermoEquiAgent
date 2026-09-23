"""Reflect DWSIM AbsorptionColumn operation modes.

Usage:
    python scripts/reflect_dwsim_absorption_modes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine.dwsim_export import _automation_factory  # noqa: E402


def _interesting(name: str) -> bool:
    lowered = name.lower()
    return any(
        term in lowered
        for term in (
            "mode",
            "type",
            "operation",
            "extract",
            "liquid",
            "absor",
            "column",
            "stage",
            "phase",
            "spec",
        )
    )


def main() -> None:
    os.environ.setdefault("DWSIM_SILENT", "1")
    factory, object_type = _automation_factory()
    automation = factory()
    flowsheet = automation.CreateFlowsheet()
    column = flowsheet.AddObject(object_type.AbsorptionColumn, 250, 0, "Probe AbsorptionColumn")
    sim = getattr(column, "GetAsObject", lambda: column)()

    print("OBJECT:", sim.GetType().FullName)
    try:
        from System import Enum  # type: ignore[import-not-found]

        op_type = sim.GetType().GetProperty("OperationMode").PropertyType
        extractor_mode = Enum.Parse(op_type, "Extractor")
        sim.OperationMode = extractor_mode
        print("SET OperationMode:", sim.OperationMode)
    except Exception as exc:  # noqa: BLE001
        print("SET OperationMode failed:", type(exc).__name__, exc)

    print("\n== Calculation modes ==")
    try:
        for mode in sim.GetCalculationModes():
            print("CALC-MODE", mode)
    except Exception as exc:  # noqa: BLE001
        print("CALC-MODE failed:", type(exc).__name__, exc)

    print("\n== Equipment types ==")
    try:
        for eq_type in sim.EquipmentTypes:
            print("EQUIP-TYPE", eq_type)
    except Exception as exc:  # noqa: BLE001
        print("EQUIP-TYPE failed:", type(exc).__name__, exc)
    print("\n== Public properties containing mode/type/extract/liquid/etc. ==")
    for prop in sim.GetType().GetProperties():
        name = prop.Name
        if not _interesting(name):
            continue
        try:
            value = prop.GetValue(sim, None)
            value_text = str(value)
        except Exception as exc:  # noqa: BLE001
            value_text = f"<{type(exc).__name__}: {exc}>"
        print(f"PROP {name}: {prop.PropertyType.FullName} = {value_text}")

    print("\n== Public fields containing mode/type/extract/liquid/etc. ==")
    for field in sim.GetType().GetFields():
        name = field.Name
        if not _interesting(name):
            continue
        try:
            value = field.GetValue(sim)
            value_text = str(value)
        except Exception as exc:  # noqa: BLE001
            value_text = f"<{type(exc).__name__}: {exc}>"
        print(f"FIELD {name}: {field.FieldType.FullName} = {value_text}")

    print("\n== Methods containing mode/type/extract/liquid/etc. ==")
    for method in sim.GetType().GetMethods():
        name = method.Name
        if not _interesting(name):
            continue
        params = ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in method.GetParameters())
        print(f"METHOD {name}({params}) -> {method.ReturnType.FullName}")

    print("\n== Nested types/enums near AbsorptionColumn ==")
    asm = sim.GetType().Assembly
    for typ in asm.GetTypes():
        full = typ.FullName or typ.Name
        if not _interesting(full):
            continue
        if typ.IsEnum or "Absorption" in full or "Column" in full or "Extract" in full:
            print("TYPE", full, "ENUM" if typ.IsEnum else "")
            if typ.IsEnum:
                try:
                    for value_name in Enum.GetNames(typ):
                        print("  -", value_name)
                except Exception as exc:  # noqa: BLE001
                    print("  <enum read failed>", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
