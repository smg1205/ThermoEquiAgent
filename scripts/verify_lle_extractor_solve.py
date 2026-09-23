"""Verify that a generated DWSIM LLE extractor actually solves without the
trivial-solution error, and report the two outlet streams' real values.

This directly calls the rigorous column's ``Calculate()`` (not the whole-
flowsheet shortcuts that can swallow the solver exception) and then reads the
raffinate/extract outlet molar flows and compositions.  It answers the question
"did it really solve?" with numbers, not log keywords.

Usage (from the repo root):

    python scripts/verify_lle_extractor_solve.py exports/test/ethanol_eac_water_lle_twophase.dwxmz
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _unpack_xml(obj: Any) -> Any:
    get_as_object = getattr(obj, "GetAsObject", None)
    return get_as_object() if callable(get_as_object) else obj


def _stream_values(stream: Any) -> dict[str, object]:
    z = list(stream.GetOverallComposition())
    return {
        "flow_mol_s": float(stream.GetMolarFlow()),
        "temperature_K": float(stream.GetTemperature()),
        "pressure_Pa": float(stream.GetPressure()),
        "z": [float(v) for v in z],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the .dwxmz file to verify.")
    args = parser.parse_args()

    load_dotenv()
    dwsim_home = os.getenv("DWSIM_HOME")
    if not dwsim_home:
        print("DWSIM_HOME is not set")
        return 2
    install_dir = Path(dwsim_home).expanduser().resolve()
    os.environ["TEMP"] = str(Path(tempfile.mkdtemp(prefix="dwsim-verify-")))
    os.environ["TMP"] = os.environ["TEMP"]
    sys.path.append(str(install_dir))

    import clr  # type: ignore[import-not-found]

    clr.AddReference(str(install_dir / "DWSIM.Automation.dll"))
    from DWSIM.Automation import Automation3  # type: ignore[import-not-found]

    automation = Automation3()
    dst = Path(args.path).resolve()
    if not dst.is_file():
        print(f"file not found: {dst}")
        return 2

    flowsheet = automation.LoadFlowsheet2(str(dst))

    # Identify objects by tag.
    col = None
    feeds: list[Any] = []
    products: list[Any] = []
    for item in flowsheet.SimulationObjects.Values:
        sim = _unpack_xml(item)
        name = sim.GetType().FullName or ""
        tag = getattr(item.GraphicObject, "Tag", "?")
        if "AbsorptionColumn" in name:
            col = sim
        elif "MaterialStream" in name:
            t = str(tag).lower()
            if "feed" in t or "solvent" in t:
                feeds.append(sim)
            else:
                products.append(sim)

    if col is None:
        print("ERROR: no AbsorptionColumn found in the flowsheet")
        return 1

    print(f"column: {col.GetType().FullName}")
    try:
        print(f"OperationMode: {col.OperationMode}")
    except Exception as exc:  # noqa: BLE001
        print(f"OperationMode: <{type(exc).__name__}>")
    try:
        print(f"NumberOfStages: {col.NumberOfStages}")
    except Exception as exc:  # noqa: BLE001
        print(f"NumberOfStages: <{type(exc).__name__}>")

    print("\n== calling column.Calculate() ==")
    try:
        col.Calculate(None)
    except Exception as exc:  # noqa: BLE001
        print(f"SOLVE FAILED: {type(exc).__name__}: {exc}")
        # Re-raise so the exit code is non-zero and the trace is visible.
        return 3

    print("SOLVE OK (no exception raised)")
    print("\n== outlet streams ==")
    for p in products:
        tag = getattr(p, "GraphicObject", None)
        tag = getattr(tag, "Tag", "?")
        try:
            vals = _stream_values(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: <read error {type(exc).__name__}: {exc}>")
            continue
        print(
            f"  {tag}: flow={vals['flow_mol_s']:.6g} mol/s, "
            f"z=[EtOH {vals['z'][0]:.4f}, EtAc {vals['z'][1]:.4f}, H2O {vals['z'][2]:.4f}]"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"unexpected: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
