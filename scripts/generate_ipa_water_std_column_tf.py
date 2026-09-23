"""Generate the **2-propanol / water ordinary (direct) distillation column** `.dwxmz`
for the **x_IPA = 0.3 (ThermoFormer)** design in report §1.6.1 (表 1.6-1).

This reproduces the ordinary binary column dimensioned by the *ThermoFormer*
short-cut design (the report's "TF" column of that table), as opposed to the
default UNIFAC run in `generate_ipa_water_std_column.py`.

It writes the reported ThermoFormer design numbers verbatim into a DWSIM rigorous
``DistillationColumn`` (the same exporter that produced
`data/exports/flow_examples/ipa_water_binary_column_x0.3.dwxmz`, but with the TF
stage/reflux/temperature values):

    system            | 2-propanol (light / overhead) + water (heavy), direct distillation
    P / feed          | 101.325 kPa, 1.0 mol/s, x_IPA = 0.3 (sat-liquid at feed bubble)
    design targets    | distillate 2-propanol purity 0.995, recovery 0.98

ThermoFormer design numbers (§1.6.1 / 表 1.6-1, TF column):
    N=17  theoretical stages     Nmin=8.795 (Fenske)
    R=2.16 operating             rmin=1.543
    T_cond (bubble) 355.25 K     T_reb (bubble) 371.72 K

Run (in an unconfined terminal with DWSIM + pythonnet):
    python scripts/generate_ipa_water_std_column_tf.py
    python scripts/generate_ipa_water_std_column_tf.py --dry-run   # print only, no DWSIM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermo_engine.column_design import bubble_temperature
from thermo_engine.dwsim_export import export_dwsim_binary_column

LIGHT = "isopropanol"     # 2-propanol, overhead-concentrated
HEAVY = "water"
X_IPA = 0.3
P_KPA = 101.325
FEED_FLOW_MOL_S = 1.0

# ---- ThermoFormer design (report §1.6.1, 表 1.6-1 TF 列) --------------------
TF_STAGES = 17
TF_MIN_STAGES = 8.795
TF_REFLUX = 2.16
TF_MIN_REFLUX = 1.543
TF_FEED_STAGE = 8        # TF feed plate (8; 2-propanol/water ordinary tower)
TF_T_COND = 355.25       # K, distillate bubble temperature
TF_T_REB = 371.72        # K, bottoms/reboiler bubble temperature
TF_ALPHA = 3.135


def feed_sat_temp_K() -> float:
    """Saturate the 2-propanol/water feed on its bubble point (liquid feed)."""
    return bubble_temperature(
        [LIGHT, HEAVY], [X_IPA, 1.0 - X_IPA], P_KPA, alpha_source="unifac"
    )


def describe() -> str:
    return (
        "2-propanol / water \u2014 ordinary binary distillation "
        "(ThermoFormer design, report \u00a71.6.1 / table 1.6-1 TF column)\n"
        f"  feed x_IPA={X_IPA:.3f}, F={FEED_FLOW_MOL_S:.1f} mol/s @ {P_KPA:.3f} kPa\n"
        f"  alpha={TF_ALPHA:.3f}\n"
        f"  Nmin={TF_MIN_STAGES:.3f}, N={TF_STAGES}, feed_stage={TF_FEED_STAGE}\n"
        f"  r_min={TF_MIN_REFLUX:.3f}, R={TF_REFLUX:.3f}\n"
        f"  T_cond={TF_T_COND:.2f} K ({TF_T_COND - 273.15:.2f} C)   "
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
        "--filename", default="ipa_water_binary_column_x0p3_tf.dwxmz",
        help="Name of the written .dwxmz.",
    )
    args = parser.parse_args()

    print("== 2-propanol / water ordinary tower \u2014 ThermoFormer design ==")
    print(describe())
    feed_t = feed_sat_temp_K()
    print(f"  feed sat-liquid T={feed_t:.2f} K ({feed_t - 273.15:.2f} C)\n")

    if args.dry_run:
        print("dry-run: no DWSIM export attempted.")
        return

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / args.filename

    path = export_dwsim_binary_column(
        components=[LIGHT, HEAVY],
        feed_composition=[X_IPA, 1.0 - X_IPA],
        feed_flow_mol_s=FEED_FLOW_MOL_S,
        feed_temperature_K=feed_t,
        operating_pressure_kPa=P_KPA,
        stages=TF_STAGES,
        minimum_stages=TF_MIN_STAGES,
        reflux_ratio=TF_REFLUX,
        minimum_reflux_ratio=TF_MIN_REFLUX,
        feed_stage=TF_FEED_STAGE,
        condenser_temperature_K=TF_T_COND,
        reboiler_temperature_K=TF_T_REB,
        destination=destination,
    )
    print(f"wrote {path.name}  ->  {path}")
    print("\nDone.  Open in DWSIM GUI and hit Calculate to inspect the column. "
          "This is the rigorous (shortcut-capable) tower built from the report "
          "ThermoFormer design numbers; report \u00a71.6.1 lists the design only and "
          "gives no DWSIM-fitted overhead purity, so treat top/bottom reach in GUI "
          "results as the source of truth rather than assuming 0.995 is reproduced "
          "automatically.")


if __name__ == "__main__":
    main()
