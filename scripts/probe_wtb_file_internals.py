"""Probe the generated column file: dump real object names and stream internals.

The generic wrappers in ``flowsheet.SimulationObjects`` hide the properties, and
``MaterialStream`` attributes vary by DWSIM build.  This dumps what is actually
reachable so the verifier can assert against the right fields.

Run with full process access (pythonnet needs OpenProcess).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = (
    ROOT / "data" / "exports" / "flow_examples"
    / "water_toluene_butanol_extractive_thermoformer.dwxmz"
)


def unwrap(obj: object) -> object:
    get_as_object = getattr(obj, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else obj


def main() -> int:
    from thermo_engine.dwsim_export import _automation_factory

    factory, _object_type = _automation_factory()
    automation = factory()
    flowsheet = automation.LoadFlowsheet(str(TARGET))

    sim_objs = flowsheet.SimulationObjects
    for key in [str(k) for k in sim_objs.Keys]:
        wrapper = sim_objs[key]
        inner = unwrap(wrapper)
        type_name = str(inner.GetType().Name)
        print(f"\n=== {key} ===")
        print(f"  wrapper : {wrapper.GetType().Name}")
        print(f"  inner   : {type_name}")

        for attr in ("Name", "Tag", "GraphicObject"):
            try:
                print(f"  {attr:22} = {getattr(inner, attr)!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {attr:22} ! {type(exc).__name__}: {str(exc)[:90]}")

        if type_name != "MaterialStream":
            continue

        print("  --- stream state ---")
        for attr in (
            "Temperature",
            "Pressure",
            "MolarFlow",
            "MassFlow",
            "VolumetricFlow",
            "ConnectedFeedStage",
            "FeedStage",
        ):
            try:
                print(f"  {attr:22} = {getattr(inner, attr)!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {attr:22} ! {type(exc).__name__}")

        for attr in ("Phases", "OverallComposition"):
            try:
                value = getattr(inner, attr)
                print(f"  {attr:22} = {value}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {attr:22} ! {type(exc).__name__}: {str(exc)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
