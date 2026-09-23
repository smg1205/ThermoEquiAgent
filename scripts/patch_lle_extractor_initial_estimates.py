"""Patch a DWSIM LLE extractor .dwxmz with rigorous-column initial estimates."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _set_value(parent: ET.Element, path: str, value: str) -> None:
    node = parent.find(path)
    if node is None:
        raise RuntimeError(f"missing XML node: {path}")
    node.text = value


def _set_optional(parent: ET.Element, path: str, value: str) -> None:
    node = parent.find(path)
    if node is not None:
        node.text = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=str(ROOT / ".pytest-tmp" / "ethanol_eac_water_lle_absorption_extractor.dwxmz"),
    )
    parser.add_argument(
        "--dst",
        default=str(ROOT / ".pytest-tmp" / "ethanol_eac_water_lle_absorption_extractor_with_estimates.dwxmz"),
    )
    parser.add_argument(
        "--property-package",
        default=None,
        choices=("NRTL", "UNIFAC-LL"),
        help="Optionally rewrite the single property package in the DWSIM file.",
    )
    parser.add_argument(
        "--strong-lle-seed",
        action="store_true",
        help="Use strongly split water-rich / ester-rich initial estimates and nonzero internal feed rates.",
    )
    parser.add_argument("--feed-flow", type=float, default=1.0)
    parser.add_argument("--solvent-flow", type=float, default=1.5)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    compounds = ["Ethanol", "Ethyl acetate", "Water"]

    with zipfile.ZipFile(src, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    xml_name = next(name for name in entries if name.lower().endswith(".xml"))
    root = ET.fromstring(entries[xml_name])

    column: ET.Element | None = None
    for sim in root.iter("SimulationObject"):
        if sim.findtext("Type") == "DWSIM.UnitOperations.UnitOperations.AbsorptionColumn":
            column = sim
            break
    if column is None:
        raise RuntimeError("no AbsorptionColumn SimulationObject found")

    _set_value(column, "OperationMode", "Extractor")
    _set_value(column, "AutoUpdateInitialEstimates", "false")
    _set_value(column, "UseTemperatureEstimates", "true")
    _set_value(column, "UseVaporFlowEstimates", "true")
    _set_value(column, "UseLiquidFlowEstimates", "true")
    _set_value(column, "UseCompositionEstimates", "true")
    _set_value(column, "InitialEstimatesProvider", "Internal 2 (Experimental)")
    _set_value(column, "SolvingMethodName", "Burningham-Otto (Sum Rates)")
    _set_value(column, "MaxIterations", "500")
    _set_value(column, "InternalLoopTolerance", "0.0001")
    _set_value(column, "ExternalLoopTolerance", "0.0001")
    _set_value(column, "ColumnPressureDrop", "1000")
    _set_optional(column, "PreferredFlashAlgorithmTag", "Nested Loops (VLLE)")

    # DWSIM may serialize fixed initial values with zero bounds.  That is
    # harmless in the editor but can make the rigorous solver reject the seed
    # after a stage/feed edit, so give every estimate parameter usable bounds.
    for parameter in column.findall(".//Parameter"):
        _set_optional(parameter, "MinVal", "-1e6")
        _set_optional(parameter, "MaxVal", "1e6")

    package_container = root.find("PropertyPackages")
    packages = [] if package_container is None else package_container.findall("PropertyPackage")
    if args.property_package == "NRTL":
        for package in packages:
            _set_value(package, "Type", "DWSIM.Thermodynamics.PropertyPackages.NRTLPropertyPackage")
            _set_value(package, "ComponentName", "NRTL")
            _set_value(package, "Tag", "NRTL")
    elif args.property_package == "UNIFAC-LL":
        for package in packages:
            _set_value(package, "Type", "DWSIM.Thermodynamics.PropertyPackages.UNIFACLLPropertyPackage")
            _set_value(package, "ComponentName", "UNIFAC-LL")
            _set_value(package, "Tag", "UNIFAC-LL")

    estimates = column.find("InitialEstimates")
    if estimates is None:
        raise RuntimeError("missing InitialEstimates")

    liquid_rows = estimates.find("LiquidCompositions").findall("LiquidComposition")
    stages = len(liquid_rows)
    if args.strong_lle_seed:
        top_liquid = [0.01, 0.01, 0.98]
        bottom_liquid = [0.12, 0.04, 0.84]
    else:
        top_liquid = [0.03, 0.02, 0.95]
        bottom_liquid = [0.10, 0.05, 0.85]

    for index, row in enumerate(liquid_rows):
        fraction = index / max(stages - 1, 1)
        values = [
            (1.0 - fraction) * top_liquid[i] + fraction * bottom_liquid[i]
            for i in range(len(compounds))
        ]
        total = sum(values)
        values = [value / total for value in values]
        for compound, value in zip(compounds, values, strict=True):
            _set_value(row, f"Compound[@ID='{compound}']/Value", f"{value:.12g}")

    if args.strong_lle_seed:
        top_vapor = [0.25, 0.72, 0.03]
        bottom_vapor = [0.40, 0.59, 0.01]
    else:
        top_vapor = [0.22, 0.73, 0.05]
        bottom_vapor = [0.38, 0.60, 0.02]
    vapor_rows = estimates.find("VaporCompositions").findall("VaporComposition")
    for index, row in enumerate(vapor_rows):
        fraction = index / max(stages - 1, 1)
        values = [
            (1.0 - fraction) * top_vapor[i] + fraction * bottom_vapor[i]
            for i in range(len(compounds))
        ]
        total = sum(values)
        values = [value / total for value in values]
        for compound, value in zip(compounds, values, strict=True):
            _set_value(row, f"Compound[@ID='{compound}']/Value", f"{value:.12g}")

    for row in estimates.find("StageTemps").findall("StageTemp"):
        _set_value(row, "Value", "298.15")
    for row in estimates.find("LiqMoleFlows").findall("LiqMoleFlow"):
        _set_value(row, "Value", f"{args.solvent_flow:.12g}")
    for row in estimates.find("VapMoleFlows").findall("VapMoleFlow"):
        _set_value(row, "Value", f"{args.feed_flow:.12g}")

    if args.strong_lle_seed:
        material_streams = column.find("MaterialStreams")
        if material_streams is not None:
            feed_rates = (f"{args.feed_flow:.12g}", f"{args.solvent_flow:.12g}")
            feeds = [
                stream
                for stream in material_streams.findall("MaterialStream")
                if stream.findtext("StreamBehavior") == "Feed"
            ]
            for stream, rate in zip(feeds, feed_rates, strict=False):
                _set_value(stream, "FlowRate/Value", rate)
                _set_optional(stream, "FlowRate/MinVal", "0")
                _set_optional(stream, "FlowRate/MaxVal", "100")

    # Apply the LLE-capable flash route to every material stream in the file,
    # including products that are not yet calculated.
    for stream in root.iter("SimulationObject"):
        if stream.findtext("Type") and "MaterialStream" in stream.findtext("Type"):
            _set_optional(stream, "PreferredFlashAlgorithmTag", "Nested Loops (VLLE)")

    entries[xml_name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, payload in entries.items():
            zout.writestr(name, payload)

    print(dst)
    print(dst.stat().st_size)


if __name__ == "__main__":
    sys.exit(main())
