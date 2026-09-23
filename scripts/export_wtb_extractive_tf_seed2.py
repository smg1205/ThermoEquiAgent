"""Export the DWSIM extractive column for §1.5 case 2 using the ThermoFormer design.

System: water (light key) / toluene (heavy key) + 1-butanol (entrainer), the
same ternary system as report §2.2 and the **ThermoFormer** arm of table 1.5-2.

Design source
-------------
``report/dwsim/design_1p5_wtb_thermoformer_seed2.json``, produced by
``scripts/design_1p5_wtb_thermoformer_seed2.py`` from
:func:`thermo_engine.column_design.design_ternary_extractive_column` with
``alpha_source="thermoformer"`` on the ``vle_overall_ternary/seed_2`` checkpoint.
No design number is hard-coded here.  The JSON hop exists because pythonnet
(``clr``) and torch cannot coexist in one process: the design runs under torch,
the export runs under the CLR.

Design values consumed (report table 1.5-2, ThermoFormer column):
    alpha_base 2.3501, alpha_ext 1.9256, selectivity 0.8193
    N = 18 stages, R = 2.095, feed stage 8, entrainer stage 2
    condenser 325.63 K, reboiler 381.28 K, feed 327.09 K

Material balance (entrainer assumed to leave entirely in the bottoms):
    feed       water 0.500 + toluene 0.500 mol/s
    entrainer  1-butanol 2.000 mol/s
    distillate (0.473684 mol/s): water 0.450000, toluene 0.023684 -> x_water = 0.95
    bottoms    (2.526316 mol/s): water 0.050000, toluene 0.476316, 1-butanol 2.000000
    closure    D + B = 3.000000 = feed + entrainer

Property package: UNIQUAC (matches the archived
``water_toluene_butanol_extractive_*.dwxmz`` files so old and new are comparable).

Output: ``data/exports/flow_examples/water_toluene_butanol_extractive_thermoformer.dwxmz``

Run with full process access (pythonnet needs OpenProcess); a restricted sandbox
rejects the CLR initialisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DESIGN_JSON = ROOT / "report" / "dwsim" / "design_1p5_wtb_thermoformer_seed2.json"
OUTDIR = ROOT / "data" / "exports" / "flow_examples"

PROPERTY_PACKAGE = "UNIQUAC"
DP_KPA = 5.0

#: Archived file this run refreshes.  Written in place so the report's filename
#: reference stays valid; the previous copy is preserved with a ``_prev`` suffix.
DEST = OUTDIR / "water_toluene_butanol_extractive_thermoformer.dwxmz"


def main() -> int:
    if not DESIGN_JSON.is_file():
        raise SystemExit(
            f"design JSON not found: {DESIGN_JSON}\n"
            "run: python scripts/design_1p5_wtb_thermoformer_seed2.py"
        )

    design = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))

    if design["alpha_source"] != "thermoformer":
        raise SystemExit(f"expected a thermoformer design, got {design['alpha_source']!r}")

    light = design["light"]
    heavy = design["heavy"]
    entrainer = design["entrainer"]
    feed = [float(v) for v in design["feed_composition"]]

    print("=== 设计值（ThermoFormer seed_2，来自 design JSON）===")
    for key in (
        "alpha_base",
        "alpha_ext",
        "selectivity",
        "alpha_avg",
        "theoretical_stages",
        "minimum_stages",
        "reflux_ratio",
        "minimum_reflux_ratio",
        "feed_stage",
        "entrainer_stage",
        "condenser_temperature_K",
        "reboiler_temperature_K",
        "feed_temperature_K",
        "distillate_flow_mol_s",
        "bottoms_flow_mol_s",
    ):
        print(f"  {key:30} {design[key]}")

    # Imported here, after the design is read: importing the exporter is what
    # pulls in pythonnet, and torch must already be out of the picture.
    from thermo_engine.dwsim_export import export_generic_extractive_column

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if DEST.is_file():
        prev = DEST.with_name(DEST.stem + "_prev" + DEST.suffix)
        prev.write_bytes(DEST.read_bytes())
        print(f"\nprevious file preserved: {prev}")

    export_generic_extractive_column(
        light=light,
        heavy=heavy,
        entrainer=entrainer,
        feed_composition=feed,
        feed_flow_mol_s=float(design["feed_flow_mol_s"]),
        feed_temperature_K=float(design["feed_temperature_K"]),
        feed_pressure_kPa=float(design["operating_pressure_kPa"]),
        stages=int(design["theoretical_stages"]),
        reflux_ratio=float(design["reflux_ratio"]),
        feed_stage=int(design["feed_stage"]),
        entrainer_stage=int(design["entrainer_stage"]),
        entrainer_ratio=float(design["entrainer_ratio"]),
        condenser_temperature_K=float(design["condenser_temperature_K"]),
        reboiler_temperature_K=float(design["reboiler_temperature_K"]),
        property_package=PROPERTY_PACKAGE,
        destination=DEST,
        pressure_drop_kPa=DP_KPA,
        distillate_purity_mole_fraction=float(design["distillate_purity_mole_fraction"]),
        # Pass the design's own product flows so the column specs cannot drift
        # from the short-cut design (the recovery-aware single source of truth).
        recovery=float(design["recovery"]),
        distillate_flow_mol_s=float(design["distillate_flow_mol_s"]),
        bottoms_flow_mol_s=float(design["bottoms_flow_mol_s"]),
    )

    if not DEST.is_file():
        raise SystemExit("export reported success but no file was written")

    print(f"\nexported: {DEST}")
    print(f"size:     {DEST.stat().st_size} bytes")
    print("\n=== DWSIM GUI 规格 ===")
    print(f"  物性包: {PROPERTY_PACKAGE}")
    print(f"  组分: {light} / {heavy} / {entrainer}")
    print(f"  {design['theoretical_stages']} 级, 进料板 {design['feed_stage']}, "
          f"萃取剂板 {design['entrainer_stage']}, R = {design['reflux_ratio']}")
    print(f"  进料 {design['feed_flow_mol_s']} mol/s (水:甲苯 = 0.5:0.5) @ "
          f"{design['feed_temperature_K']} K")
    print(f"  萃取剂 1-丁醇 {design['entrainer_ratio']} mol/s @ "
          f"{design['condenser_temperature_K']} K")
    print(f"  冷凝器 {design['condenser_temperature_K']} K / "
          f"再沸器 {design['reboiler_temperature_K']} K")
    print(f"  塔顶 {design['distillate_flow_mol_s']:.6f} mol/s (x_water = 0.95)")
    print(f"  塔釜 {design['bottoms_flow_mol_s']:.6f} mol/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
