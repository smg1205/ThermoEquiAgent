"""Generate a DWSIM ``.dwxmz`` for the **ethyl-acetate / n-propyl-acetate + DMSO
extractive-distillation tower** dimensioned by the **ThermoFormer short-cut column**
of report §1.6.3 / the extractive design in §1.6.

This is the extractive sibling of :file:`scripts/generate_ipa_water_std_column.py`:
we write the **ThermoFormer column** numbers from the §1.6.3 comparison table
verbatim into a DWSIM rigorous ``DistillationColumn`` instead of re-deriving them,
so the exported file faithfully reproduces the reported ThermoFormer reference
(rather than silently falling back to UNIFAC when a ThermoFormer checkpoint is not
present in the local env).

Reported ThermoFormer short-cut design (§1.6.3 comparison table)
    system            | ethyl acetate (light/top) + n-propyl acetate (heavy)
    entrainer         | dimethyl sulfoxide (DMSO), ratio 2.0 (pure), fed above stage 2
    feed              | E : n-PAC = 0.5 : 0.5, 1.0 mol/s, @101.325 kPa
    alpha_base/ext    | 2.240 / 1.943     selectivity 0.931     alpha_avg 2.086
    N (theoretical)   | 18                 Nmin (Fenske) 9.310
    R / r_min         | 2.179 / 1.557
    T_cond / T_reb    | 349.73 / 391.20 K

Run (in an unconfined terminal with DWSIM + pythonnet):
    python scripts/generate_eac_npac_dmso_extractive_tf.py
    python scripts/generate_eac_npac_dmso_extractive_tf.py --dry-run     # no DWSIM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermo_engine.column_design import bubble_temperature
from thermo_engine.dwsim_export import export_generic_extractive_column

LIGHT = "ethyl acetate"
HEAVY = "n-propyl acetate"
ENTRAINER = "dimethyl sulfoxide"
FEED_COMP = [0.5, 0.5]
FEED_FLOW_MOL_S = 1.0
P_KPA = 101.325
ENTRAINER_RATIO = 2.0

# ---- ThermoFormer short-cut design (verbatim from report §1.6.3 table) ------
TF_STAGES = 18
TF_REFLUX = 2.179
TF_MIN_STAGES = 9.310
TF_MIN_REFLUX = 1.557
TF_FEED_STAGE = 8          # the design's feed plate (0.45*(18+1) rounding)
TF_ENTRAINER_STAGE = 2     # DMSO fed above stage 2
TF_T_COND = 349.73         # K, distillate bubble
TF_T_REB = 391.20          # K, bottoms/reboiler bubble
TF_ALPHA_AVG = 2.086
TF_ALPHA_BASE = 2.240
TF_ALPHA_EXT = 1.943
TF_SELECTIVITY = 0.931

PROPERTY_PACKAGE = "UNIQUAC"


def feed_bubble_temperature_K() -> float:
    """Bubble temperature of the saturated-liquid E : n-PAC binary feed.

    The report files feed a liquid E/P mixture; we place the feed on its
    (deterministic) bubble point using the same engine ``export_flow_examples.py``
    relies on, so the DWSIM feed stream is a genuine saturated liquid.
    """
    return bubble_temperature([LIGHT, HEAVY], FEED_COMP, P_KPA, alpha_source="unifac")


def describe() -> str:
    return (
        "Ethyl acetate / n-propyl acetate + DMSO extractive-distillation tower\n"
        "  (dimensional by the ThermoFormer short-cut column of report §1.6.3)\n"
        f"  feed E:n-PAC = {FEED_COMP[0]:.1f}:{1 - FEED_COMP[0]:.1f}, "
        f"{FEED_FLOW_MOL_S:.1f} mol/s @ {P_KPA:.3f} kPa, sat-liquid T="
        f"{feed_bubble_temperature_K():.2f} K\n"
        f"  entrainer DMSO ratio {ENTRAINER_RATIO:.1f} (pure), fed above stage "
        f"{TF_ENTRAINER_STAGE}\n"
        f"  property package = {PROPERTY_PACKAGE}   |   tower stages N={TF_STAGES}, "
        f"feed_stage={TF_FEED_STAGE}, entrainer_stage={TF_ENTRAINER_STAGE}\n"
        f"  reflux R={TF_REFLUX:.3f}, Nmin={TF_MIN_STAGES:.3f}, Rmin={TF_MIN_REFLUX:.3f}\n"
        f"  alpha_base/ext = {TF_ALPHA_BASE:.3f} / {TF_ALPHA_EXT:.3f} "
        f"(sel={TF_SELECTIVITY:.3f}, alpha_avg={TF_ALPHA_AVG:.3f})\n"
        f"  T_cond={TF_T_COND:.2f} K ({TF_T_COND - 273.15:.2f} C)  "
        f"T_reb={TF_T_REB:.2f} K ({TF_T_REB - 273.15:.2f} C)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path,
        default=_REPO_ROOT / "data" / "exports" / "flow_examples",
        help="Output directory for the .dwxmz (created if absent).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the design without contacting DWSIM.",
    )
    parser.add_argument(
        "--filename", default="eac_npac_dmso_extractive_tf_column.dwxmz",
        help="Name of the written .dwxmz.",
    )
    args = parser.parse_args()

    print("== Extractive tower (ThermoFormer short-cut column, report §1.6.3) ==")
    print(describe())
    print()

    if args.dry_run:
        print("dry-run: no DWSIM export attempted.")
        return

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / args.filename

    path = export_generic_extractive_column(
        light=LIGHT,
        heavy=HEAVY,
        entrainer=ENTRAINER,
        feed_composition=list(FEED_COMP),
        feed_flow_mol_s=FEED_FLOW_MOL_S,
        feed_temperature_K=feed_bubble_temperature_K(),
        feed_pressure_kPa=P_KPA,
        stages=TF_STAGES,
        reflux_ratio=TF_REFLUX,
        feed_stage=TF_FEED_STAGE,
        entrainer_stage=TF_ENTRAINER_STAGE,
        entrainer_ratio=ENTRAINER_RATIO,
        condenser_temperature_K=TF_T_COND,
        reboiler_temperature_K=TF_T_REB,
        property_package=PROPERTY_PACKAGE,
        destination=destination,
    )
    print(f"wrote {path.name}  ->  {path}")
    print(
        "\nDone.  Open the file in DWSIM GUI and run Calculate (Naphtali-Sandholm solver). "
        "The exported file is the rigorous 18-stage column built from the ThermoFormer "
        "short-cut numbers.  To drive the overhead toward high ethyl-acetate purity you "
        "will usually set a Condenser spec (e.g. reflux ~2.18) / Reboiler spec in the GUI; "
        "see report §1.6.3 for how the DWSIM operating point is recorded and verified."
    )


if __name__ == "__main__":
    main()
