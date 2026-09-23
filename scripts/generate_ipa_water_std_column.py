"""Generate a DWSIM ``.dwxmz`` for the **2-propanol / water ordinary (direct)
binary-distillation column** described in report §1.6.2.

Design dossier (from §1.6.2 of the v5 report):
    system        | 2-propanol (light / overhead key) + water (heavy)
    mode          | direct, plain binary distillation (no extractant)
    P             | 101.325 kPa
    feed          | 1.0 mol/s, liquid, x_IPA target (default 0.3)
    product spec  | distillate 2-propanol purity = 0.995 (mole frac), recovery 0.98
    method        | deterministic short-cut Fenske-Underwood-Gilliland-Eduljee
                 |   ``thermo_engine.column_design.design_binary_distillation_column``
                 |   (alpha and bubble temperatures from UNIFAC by default, or
                 |   ThermoFormer bubble-point y when ``--alpha-source tf``).

It then writes the design verbatim into DWSIM via
``thermo_engine.dwsim_export.export_dwsim_binary_column`` (rigorous column; the
same exporter used for ``data/exports/flow_examples/ipa_water_binary_column.dwxmz``).

Prerequsite: a real DWSIM install reachable through ``DWSIM_HOME`` and
``pythonnet`` working in the *unconfined* terminal (pythonnet needs full process
access; it fails inside a file/read-only sandbox with ``拒绝访问``).

Run:
    python scripts/generate_ipa_water_std_column.py                      # x_IPA=0.3, UNIFAC -> .dwxmz
    python scripts/generate_ipa_water_std_column.py --dry-run            # print design only, no DWSIM
    python scripts/generate_ipa_water_std_column.py -x 0.5 --outdir data/exports/flow_examples
    python scripts/generate_ipa_water_std_column.py --alpha-source tf    # ThermoFormer-vs-UNIFAC variant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the deterministic short-cut design engine and the DWSIM exporter.
from thermo_engine.column_design import (
    BinaryDistillationDesign,
    bubble_temperature,
    design_binary_distillation_column,
)
from thermo_engine.dwsim_export import export_dwsim_binary_column

LIGHT = "isopropanol"          # overhead-concentrated (more volatile)
HEAVY = "water"                # bottoms-concentrated
DEFAULT_X_IPA = 0.3            # feed mole fraction of 2-propanol (§1.6.2 operating point)
P_KPA = 101.325                # report operating pressure
PURITY = 0.995                 # distillate 2-propanol purity (mole fraction)
RECOVERY = 0.98                # 2-propanol recovery to the distillate


def design_binary(
    x_ipa: float, alpha_source: str, feed_flow_mol_s: float
) -> tuple[BinaryDistillationDesign, float]:
    """Run the deterministic binary-column design used to dimension the DWSIM column.

    Returns ``(design, feed_temperature_K)``.  ``feed_temperature_K`` is the
    deterministic isobaric bubble temperature of the saturated-liquid feed (also
    the reference temperature the short-cut method uses for the relative
    volatility); the ``BinaryDistillationDesign`` object does not store it, so we
    re-derive it here with the same source to set the DWSIM feed stream on its
    bubble point (consistent with how ``scripts/export_flow_examples.py`` works).
    """
    feed = [x_ipa, 1.0 - x_ipa]
    feed_temperature_K = bubble_temperature(
        [LIGHT, HEAVY], feed, P_KPA, alpha_source=alpha_source
    )
    d = design_binary_distillation_column(
        [LIGHT, HEAVY],
        feed,
        feed_flow_mol_s=feed_flow_mol_s,
        feed_temperature_K=feed_temperature_K,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        alpha_source=alpha_source,
    )
    return d, feed_temperature_K


def describe(d: BinaryDistillationDesign, feed_temperature_K: float) -> str:
    return (
        "2-propanol / water \u2014 direct binary distillation (deterministic short-cut design)\n"
        f"  feed x_IPA={d.feed_composition[0]:.3f}, F={d.feed_flow_mol_s:.3f} mol/s, "
        f"P={d.operating_pressure_kPa:.3f} kPa, alpha_source={d.alpha_source}\n"
        f"  light key = 2-propanol (overhead): dist purity={d.distillate_purity_mole_fraction:.3f}, "
        f"recovery ~ distillate_flow={d.distillate_flow_mol_s:.4f} mol/s\n"
        f"  alpha={d.relative_volatility:.3f}\n"
        f"  Nmin (Fenske)={d.minimum_stages:.3f}, N={d.theoretical_stages}, "
        f"feed_stage={d.feed_stage}\n"
        f"  r_min (Underwood)={d.minimum_reflux_ratio:.3f}, R (={1.4:.1f}*r_min)={d.reflux_ratio:.3f}\n"
        f"  T_cond (bubble)={d.condenser_temperature_K:.2f} K "
        f"({d.condenser_temperature_K - 273.15:.2f} \u00b0C)\n"
        f"  T_reb  (bubble)={d.reboiler_temperature_K:.2f} K "
        f"({d.reboiler_temperature_K - 273.15:.2f} \u00b0C)\n"
        f"  T_feed (bubble)={feed_temperature_K:.2f} K "
        f"({feed_temperature_K - 273.15:.2f} \u00b0C)"
    )


def export_flowsheet(
    d: BinaryDistillationDesign, feed_temperature_K: float, outdir: Path, filename: str
) -> Path:
    """Write the design into a DWSIM rigorous binary column and save the ``.dwxmz``."""
    destination = outdir / filename
    return export_dwsim_binary_column(
        components=[LIGHT, HEAVY],
        feed_composition=list(d.feed_composition),
        feed_flow_mol_s=d.feed_flow_mol_s,
        feed_temperature_K=feed_temperature_K,
        operating_pressure_kPa=d.operating_pressure_kPa,
        stages=d.theoretical_stages,
        minimum_stages=d.minimum_stages,
        reflux_ratio=d.reflux_ratio,
        minimum_reflux_ratio=d.minimum_reflux_ratio,
        feed_stage=d.feed_stage,
        condenser_temperature_K=d.condenser_temperature_K,
        reboiler_temperature_K=d.reboiler_temperature_K,
        destination=destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-x", "--x-ipa", type=float, default=[DEFAULT_X_IPA],
        help="Feed mole fraction of 2-propanol (default 0.3, the report §1.6.2 point). "
             "Must be in (0, 1) and you may pass multiple (e.g. -x 0.3 -x 0.5) to batch.",
        action="append", dest="x_list", metavar="X",
    )
    parser.add_argument(
        "--alpha-source", default="unifac", choices=("unifac", "tf"),
        help="Where the relative volatility and bubble temperatures come from. "
             "unifac = engine UNIFAC (report left column); tf = ThermoFormer "
             "bubble-point y -> alpha (report right column).",
    )
    parser.add_argument(
        "--feed-flow", type=float, default=1.0,
        help="Feed molar flow in mol/s (report uses 1.0).",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=_REPO_ROOT / "data" / "exports" / "flow_examples",
        help="Directory where the .dwxmz files are written (created if absent).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Recompute the design(s) and print them without contacting DWSIM.",
    )
    args = parser.parse_args()

    x_values = sorted({float(x) for x in args.x_list})
    for x in x_values:
        if not 0.0 < x < 1.0:
            parser.error(f"--x-ipa must be strictly inside (0,1), got {x}")

    # Deterministic design (no DWSIM needed) always runs first and is printed.
    designs: list[tuple[BinaryDistillationDesign, float]] = []
    print("== Deterministic binary-column designs (Fenske/Underwood/Gilliland) ==")
    for x in x_values:
        d = design_binary(x, args.alpha_source, args.feed_flow)
        designs.append(d)
        print(f"[x_IPA={x:.3f} alpha_source={args.alpha_source}]")
        print(describe(d[0], d[1]))
        print()

    if args.dry_run:
        print("dry-run: no DWSIM export attempted.")
        return

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for d, feed_temperature_K in designs:
        x = d.feed_composition[0]
        label = f"ipa_water_binary_column_x{x:g}"
        if args.alpha_source != "unifac":
            label = f"{label}_tf"
        path = export_flowsheet(d, feed_temperature_K, outdir, f"{label}.dwxmz")
        written.append(path)
        print(f"wrote {path.name}  ->  {path}")

    print("\nDone.  Open the file(s) in DWSIM GUI and hit Calculate to inspect the "
          "rigorous-column profile.  A distillate purity close to 0.995 may need the "
          "GUI condenser / reboiler specifications to reproduce exactly (see report "
          "§1.6 notes on how the DWSIM column operating point is recorded).")


if __name__ == "__main__":
    main()
