"""Export two DWSIM ``.dwxmz`` column flowsheets matching §1.6 of the v5 report.

* Case 1 — 2-propanol / water, plain binary distillation column.
* Case 2 — ethyl acetate / n-propyl acetate separated with dimethyl sulfoxide
  (DMSO) as an extractive-distillation column.

Both towers are designed by the deterministic engine so the DWSIM flowsheets
carry real, reproducible design values:

* binary:      ``thermo_engine.column_design.design_binary_distillation_column``
* extractive:  ``thermo_engine.column_design.design_ternary_extractive_column``

Then they are sent to DWSIM through:

* ``dwsim_export.export_dwsim_binary_column``            (case 1)
* ``dwsim_export.export_generic_extractive_column``      (case 2)

Prerequsite: a real DWSIM install reachable via ``DWSIM_HOME`` (the exporter
loads ``DWSIM.Automation.dll`` through pythonnet). Run with DWSIM present:

    conda activate <env-with-dwsim-and-pythonnet>
    python scripts/export_flow_examples.py --outdir data/exports/flow_examples

With ``--dry-run`` it prints the design inputs/outputs without contacting DWSIM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the deterministic design engine.
from thermo_engine.column_design import (
    bubble_temperature,
    design_binary_distillation_column,
    design_ternary_extractive_column,
)
from thermo_engine.dwsim_export import (
    export_dwsim_binary_column,
    export_generic_extractive_column,
)

P_KPA = 101.325
PURITY = 0.995
RECOVERY = 0.98


def case1_binary() -> dict[str, object]:
    """2-propanol / water, feed x_IPA = 0.5 (same operating point as §1.6)."""
    light, heavy = "isopropanol", "water"
    x = 0.5
    feed = [x, 1.0 - x]
    # Use the engine's isobaric bubble at the feed as the feed-stream temperature.
    feed_t = bubble_temperature([light, heavy], feed, P_KPA, alpha_source="unifac")
    d = design_binary_distillation_column(
        [light, heavy], feed,
        feed_flow_mol_s=1.0, feed_temperature_K=feed_t,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY, recovery=RECOVERY,
        alpha_source="unifac",
    )
    return {
        "kind": "binary",
        "components": [light, heavy],
        "feed_composition": feed,
        "feed_flow_mol_s": d.feed_flow_mol_s,
        "feed_temperature_K": feed_t,
        "operating_pressure_kPa": d.operating_pressure_kPa,
        "stages": d.theoretical_stages,
        "condenser_temperature_K": d.condenser_temperature_K,
        "reboiler_temperature_K": d.reboiler_temperature_K,
        "reflux_ratio": d.reflux_ratio,
        "feed_stage": d.feed_stage,
        # for print-out
        "alpha": d.relative_volatility,
        "n_min": d.minimum_stages,
        "r_min": d.minimum_reflux_ratio,
    }


def case2_extractive(*, purity: float = 0.95, recovery: float = 0.90,
                     entrainer_ratio: float = 2.0) -> dict[str, object]:
    """ethyl acetate / n-propyl acetate + DMSO, feed E:P = 0.5 : 0.5, entrainer_ratio.

    ``purity``/``recovery`` are the light-key (ethyl acetate) tower targets.  The
    default here is deliberately gentler than the §1.6 design (0.995 / 0.98):
    a lower top purity and recovery keep the DWSIM rigorous column well inside a
    feasible operating envelope so Naphtali-Sandholm cold-starts converge.  Re-run
    the script with ``--case2-purity 0.995 --case2-recovery 0.98`` to go back to
    the strict report specification.
    """
    light, heavy, ent = "ethyl acetate", "n-propyl acetate", "dimethyl sulfoxide"
    feed = [0.5, 0.5]
    feed_t = bubble_temperature([light, heavy], feed, P_KPA, alpha_source="unifac")
    des = design_ternary_extractive_column(
        light, heavy, ent,
        feed_composition=feed, feed_flow_mol_s=1.0,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=purity, recovery=recovery,
        entrainer_ratio=entrainer_ratio, alpha_source="unifac",
    )
    return {
        "kind": "extractive",
        "light": light,
        "heavy": heavy,
        "entrainer": ent,
        "feed_composition": feed,
        "feed_flow_mol_s": 1.0,
        "feed_temperature_K": float(des["feed_temperature_K"]),
        "operating_pressure_kPa": float(des["operating_pressure_kPa"]),
        "entrainer_ratio": entrainer_ratio,
        "stages": int(des["theoretical_stages"]),
        "feed_stage": int(des["feed_stage"]),
        "entrainer_stage": int(des["entrainer_stage"]),
        "reflux_ratio": float(des["reflux_ratio"]),
        "condenser_temperature_K": float(des["condenser_temperature_K"]),
        "reboiler_temperature_K": float(des["reboiler_temperature_K"]),
        "property_package": "UNIQUAC",
        "purity": purity,
        "recovery": recovery,
        # for print-out
        "alpha_base": des["alpha_base"],
        "alpha_ext": des["alpha_ext"],
        "selectivity": des["selectivity"],
        "alpha_avg": des["alpha_avg"],
        "n_min": des["minimum_stages"],
        "r_min": des["minimum_reflux_ratio"],
    }


def describe(case: dict[str, object]) -> str:
    if case["kind"] == "binary":
        comps = "/".join(case["components"]) + f" (x light key = {case['feed_composition'][0]})"
        return (
            f"binary  {comps}\n"
            f"  alpha={case['alpha']:.3f}  N={case['stages']}  Nmin={case['n_min']:.3f}\n"
            f"  R={case['reflux_ratio']:.3f}  Rmin={case['r_min']:.3f}  feed_stage={case['feed_stage']}\n"
            f"  Tcond={case['condenser_temperature_K']:.2f} K  Treb={case['reboiler_temperature_K']:.2f} K\n"
        )
    return (
        f"extractive  ({case['light']}, {case['heavy']}) + {case['entrainer']}\n"
        f"  alpha_base={case['alpha_base']:.3f} alpha_ext={case['alpha_ext']:.3f} "
        f"sel={case['selectivity']:.3f} alpha_avg={case['alpha_avg']:.3f}\n"
        f"  N={case['stages']}  Nmin={case['n_min']:.3f}  R={case['reflux_ratio']:.3f}  "
        f"Rmin={case['r_min']:.3f}\n"
        f"  feed_stage={case['feed_stage']} entrainer_stage={case['entrainer_stage']}\n"
        f"  Tcond={case['condenser_temperature_K']:.2f} K  Treb={case['reboiler_temperature_K']:.2f} K\n"
    )


def export_case1(case: dict[str, object], outdir: Path) -> Path:
    dest = outdir / "ipa_water_binary_column.dwxmz"
    return export_dwsim_binary_column(
        components=list(case["components"]),
        feed_composition=[float(v) for v in case["feed_composition"]],
        feed_flow_mol_s=case["feed_flow_mol_s"],
        feed_temperature_K=case["feed_temperature_K"],
        operating_pressure_kPa=case["operating_pressure_kPa"],
        stages=case["stages"],
        minimum_stages=case["n_min"],
        reflux_ratio=case["reflux_ratio"],
        minimum_reflux_ratio=case["r_min"],
        feed_stage=case["feed_stage"],
        condenser_temperature_K=case["condenser_temperature_K"],
        reboiler_temperature_K=case["reboiler_temperature_K"],
        destination=dest,
    )


def export_case2(case: dict[str, object], outdir: Path) -> Path:
    dest = outdir / "eac_npac_dmso_extractive_column.dwxmz"
    return export_generic_extractive_column(
        light=case["light"],
        heavy=case["heavy"],
        entrainer=case["entrainer"],
        feed_composition=[float(v) for v in case["feed_composition"]],
        feed_flow_mol_s=case["feed_flow_mol_s"],
        feed_temperature_K=case["feed_temperature_K"],
        feed_pressure_kPa=case["operating_pressure_kPa"],
        stages=case["stages"],
        reflux_ratio=case["reflux_ratio"],
        feed_stage=case["feed_stage"],
        entrainer_stage=case["entrainer_stage"],
        entrainer_ratio=case["entrainer_ratio"],
        condenser_temperature_K=case["condenser_temperature_K"],
        reboiler_temperature_K=case["reboiler_temperature_K"],
        property_package=case["property_package"],
        destination=dest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/exports/flow_examples"),
        help="Directory where the .dwxmz files are written (created if absent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute the two designs and print them without contacting DWSIM.",
    )
    parser.add_argument(
        "--case2-property-package",
        default="UNIQUAC",
        choices=("UNIQUAC", "NRTL", "Wilson"),
        help="Property package written into the case-2 extractive flowsheet. "
             "UNIQUAC is more stable for the ester/DMSO system; NRTL was the original.",
    )
    parser.add_argument(
        "--case2-purity",
        type=float,
        default=0.95,
        help="Case-2 overhead light-key (ethyl acetate) purity target. "
             "Lower = much easier for the DWSIM rigorous column to converge.",
    )
    parser.add_argument(
        "--case2-recovery",
        type=float,
        default=0.90,
        help="Case-2 overhead light-key recovery fraction. Lower keeps the column "
             "inside a feasible envelope for cold-start convergence.",
    )
    parser.add_argument(
        "--case2-entrainer-ratio",
        type=float,
        default=2.0,
        help="DMSO : feed molar ratio for the case-2 extractive column. "
             "Lower ratios (e.g. 1.5) ease DWSIM convergence, at the cost of less "
             "entrainment effect.",
    )
    args = parser.parse_args()

    case1 = case1_binary()
    case2 = case2_extractive(
        purity=args.case2_purity,
        recovery=args.case2_recovery,
        entrainer_ratio=args.case2_entrainer_ratio,
    )
    case2["property_package"] = args.case2_property_package
    print("== Designs (deterministic engine) ==")
    print(describe(case1))
    print(describe(case2))

    if args.dry_run:
        print("dry-run: no DWSIM export attempted.")
        return

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print("Writing DWSIM flowsheets into:", outdir)
    a = export_case1(case1, outdir)
    print("  wrote", a.name)
    c2 = case2
    b = export_case2(c2, outdir)
    print("  wrote", b.name)
    _print_case2_gui(c2, b)
    print("Done.")


def _print_case2_gui(case: dict[str, object], path: Path) -> None:
    """Print, for the exported case-2 file, the condenser/reboiler specs a user
    still has to enter in the DWSIM GUI to make the rigorous column converge."""
    feed_flow = float(case["feed_flow_mol_s"])            # 1.0 E/P feed
    ratio = float(case["entrainer_ratio"])                # DMSO/feed
    recovery = float(case["recovery"])
    dist_flow = feed_flow * recovery * 0.5                # ethyl acetate to top (share 0.5 in feed)
    # overhead also carries the small heavy impurity; total D close to dist_flow
    total_in = feed_flow + ratio * feed_flow              # feed + DMSO entrainer
    bottoms_flow = total_in - dist_flow
    print(
        "\n[GUI 规格提示 - case2]: open {0} and set (Naphtali-Sandholm solver):\n"
        "  Condenser spec : Product flow = {1:.3f} mol/s (ethyl-acetate overhead)\n"
        "  Reboiler spec  : Bottoms flow  = {2:.3f} mol/s   (feed {3:.2f} + DMSO {4:.2f} - overhead)\n"
        "  initial tray T  : from ~77 C (top) to ~126 C (bottom)\n"
        .format(path.name, dist_flow, bottoms_flow, feed_flow, ratio * feed_flow)
    )



if __name__ == "__main__":
    main()
