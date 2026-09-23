"""IPA / water (x_IPA = 0.5) binary column: UNIFAC vs ThermoFormer short-cut design.

v4-report style: both designs use the same deterministic Fenske-Underwood-Gilliland
short-cut; only the thermodynamic source of the relative volatility (and the
bubble temperatures) differs:

    alpha_source="unifac"        -> mechanistic UNIFAC activity coefficients
    alpha_source="thermoformer"  -> ThermoFormer isobaric bubble prediction

The ThermoFormer-source design is then exported to DWSIM, so the generated column
carries ThermoFormer-derived stage/reflux numbers.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine.column_design import (  # noqa: E402
    design_binary_distillation_column,
)
from thermo_engine.dwsim_export import export_dwsim_binary_column  # noqa: E402

P_KPA = 101.325
DP_KPA = 5.0
X_IPA = 0.5
PURITY = 0.995
RECOVERY = 0.98
# Feed temperature handed to the DWSIM feed stream: the UNIFAC isobaric bubble
# point of x_IPA = 0.5 at 101.325 kPa.
FEED_TEMPERATURE_K = 353.74
COMPONENTS = ["isopropanol", "water"]
FEED = [X_IPA, 1.0 - X_IPA]
OUTDIR = ROOT / "report" / "dwsim"


def run_design(alpha_source: str) -> dict:
    """Run the deterministic binary short-cut design with the given alpha source.

    feed_temperature_K is left unset so the engine evaluates the feed bubble
    temperature with the SAME source that supplies alpha.
    """
    d = design_binary_distillation_column(
        COMPONENTS,
        FEED,
        feed_flow_mol_s=1.0,
        feed_temperature_K=None,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        alpha_source=alpha_source,
    )
    return {
        "source": alpha_source,
        "alpha": d.relative_volatility,
        "N": d.theoretical_stages,
        "N_min": d.minimum_stages,
        "R": d.reflux_ratio,
        "R_min": d.minimum_reflux_ratio,
        "feed_stage": d.feed_stage,
        "T_cond": d.condenser_temperature_K,
        "T_reb": d.reboiler_temperature_K,
        "D": d.distillate_flow_mol_s,
        "B": d.bottoms_flow_mol_s,
        "design": d,
    }


def main():
    results = {}
    for source in ("unifac", "thermoformer"):
        print(f"--- designing with alpha_source = {source} ---")
        try:
            results[source] = run_design(source)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[source] = None

    u = results.get("unifac")
    t = results.get("thermoformer")

    print("\n=== 2-propanol / water  x_IPA = 0.5  binary distillation, 101.325 kPa ===")
    header = f"{'quantity':<26}{'UNIFAC':>14}{'ThermoFormer':>16}"
    print(header)
    print("-" * len(header))

    def row(label, key, fmt="{:.4f}"):
        uv = fmt.format(u[key]) if u else "n/a"
        tv = fmt.format(t[key]) if t else "n/a"
        print(f"{label:<26}{uv:>14}{tv:>16}")

    if u or t:
        row("relative volatility", "alpha")
        row("theoretical stages N", "N", "{:d}")
        row("minimum stages N_min", "N_min")
        row("reflux ratio R", "R")
        row("minimum reflux R_min", "R_min")
        row("feed stage", "feed_stage", "{:d}")
        row("condenser T (K)", "T_cond", "{:.2f}")
        row("reboiler T (K)", "T_reb", "{:.2f}")

    # write a CSV of the comparison
    csv_path = OUTDIR / "ipa_water_x0p5_source_comparison.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "unifac", "thermoformer"])
        for label, key, fmt in (
            ("alpha", "alpha", "{}"),
            ("N", "N", "{}"),
            ("N_min", "N_min", "{}"),
            ("R", "R", "{}"),
            ("R_min", "R_min", "{}"),
            ("feed_stage", "feed_stage", "{}"),
            ("condenser_T_K", "T_cond", "{}"),
            ("reboiler_T_K", "T_reb", "{}"),
        ):
            w.writerow([label, u[key] if u else "", t[key] if t else ""])
    print(f"\nwrote: {csv_path}")

    # export the ThermoFormer-source column (this is the requested "TF" .dwxmz)
    if t is not None:
        d = t["design"]
        dest = OUTDIR / "ipa_water_binary_column_x0p5_tf.dwxmz"
        export_dwsim_binary_column(
            components=COMPONENTS,
            feed_composition=FEED,
            feed_flow_mol_s=1.0,
            feed_temperature_K=FEED_TEMPERATURE_K,
            operating_pressure_kPa=d.operating_pressure_kPa,
            stages=d.theoretical_stages,
            minimum_stages=d.minimum_stages,
            reflux_ratio=d.reflux_ratio,
            minimum_reflux_ratio=d.minimum_reflux_ratio,
            feed_stage=d.feed_stage,
            condenser_temperature_K=d.condenser_temperature_K,
            reboiler_temperature_K=d.reboiler_temperature_K,
            destination=dest,
            distillate_flow_mol_s=d.distillate_flow_mol_s,
            bottoms_flow_mol_s=d.bottoms_flow_mol_s,
            pressure_drop_kPa=DP_KPA,
            solving_method="Naphtali-Sandholm",
            max_iterations=500,
        )
        print(f"[OK] wrote ThermoFormer-source column: {dest}")
    if u is not None:
        d = u["design"]
        dest = OUTDIR / "ipa_water_binary_column_x0p5_unifac.dwxmz"
        export_dwsim_binary_column(
            components=COMPONENTS,
            feed_composition=FEED,
            feed_flow_mol_s=1.0,
            feed_temperature_K=FEED_TEMPERATURE_K,
            operating_pressure_kPa=d.operating_pressure_kPa,
            stages=d.theoretical_stages,
            minimum_stages=d.minimum_stages,
            reflux_ratio=d.reflux_ratio,
            minimum_reflux_ratio=d.minimum_reflux_ratio,
            feed_stage=d.feed_stage,
            condenser_temperature_K=d.condenser_temperature_K,
            reboiler_temperature_K=d.reboiler_temperature_K,
            destination=dest,
            distillate_flow_mol_s=d.distillate_flow_mol_s,
            bottoms_flow_mol_s=d.bottoms_flow_mol_s,
            pressure_drop_kPa=DP_KPA,
            solving_method="Naphtali-Sandholm",
            max_iterations=500,
        )
        print(f"[OK] wrote UNIFAC-source column: {dest}")


if __name__ == "__main__":
    main()
