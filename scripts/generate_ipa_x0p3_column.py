"""2-propanol / water, x_IPA = 0.3: dual-source binary column design + DWSIM export.

Reuses the v4-report case-1 operating point (x_IPA = 0.3, 101.325 kPa) but exports
the rigorous DWSIM column with:
  * explicit column pressure drop,
  * feed bound to its designated tray,
  * both mandatory specifications (condenser reflux + reboiler bottoms flow),
  * Naphtali-Sandholm solver with seeded temperature/flow estimates.

One .dwxmz is written per thermodynamic source (UNIFAC and ThermoFormer).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine.column_design import design_binary_distillation_column  # noqa: E402
from thermo_engine.dwsim_export import export_dwsim_binary_column  # noqa: E402

P_KPA = 101.325
DP_KPA = 5.0
X_IPA = 0.3
PURITY = 0.995
RECOVERY = 0.98
COMPONENTS = ["isopropanol", "water"]
FEED = [X_IPA, 1.0 - X_IPA]
OUTDIR = ROOT / "report" / "dwsim"


def design(source: str):
    d = design_binary_distillation_column(
        COMPONENTS,
        FEED,
        feed_flow_mol_s=1.0,
        feed_temperature_K=None,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        alpha_source=source,
    )
    return d


def main():
    designs = {}
    for source in ("unifac", "thermoformer"):
        print(f"--- designing with alpha_source = {source} ---")
        try:
            designs[source] = design(source)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            designs[source] = None

    u, t = designs.get("unifac"), designs.get("thermoformer")

    print("\n=== 2-propanol / water   x_IPA = 0.3   binary distillation, 101.325 kPa ===")
    hdr = f"{'quantity':<26}{'UNIFAC':>14}{'ThermoFormer':>16}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, attr, fmt="{:.4f}"):
        uv = fmt.format(getattr(u, attr)) if u else "n/a"
        tv = fmt.format(getattr(t, attr)) if t else "n/a"
        print(f"{label:<26}{uv:>14}{tv:>16}")

    row("relative volatility", "relative_volatility")
    row("theoretical stages N", "theoretical_stages", "{:d}")
    row("minimum stages N_min", "minimum_stages")
    row("reflux ratio R", "reflux_ratio")
    row("minimum reflux R_min", "minimum_reflux_ratio")
    row("feed stage", "feed_stage", "{:d}")
    row("condenser T (K)", "condenser_temperature_K", "{:.2f}")
    row("reboiler T (K)", "reboiler_temperature_K", "{:.2f}")
    row("distillate D (mol/s)", "distillate_flow_mol_s", "{:.4f}")
    row("bottoms B (mol/s)", "bottoms_flow_mol_s", "{:.4f}")

    csv_path = OUTDIR / "ipa_water_x0p3_source_comparison.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "unifac", "thermoformer"])
        for attr in ("relative_volatility", "theoretical_stages", "minimum_stages",
                     "reflux_ratio", "minimum_reflux_ratio", "feed_stage",
                     "condenser_temperature_K", "reboiler_temperature_K",
                     "distillate_flow_mol_s", "bottoms_flow_mol_s"):
            w.writerow([attr, getattr(u, attr) if u else "", getattr(t, attr) if t else ""])
    print(f"\nwrote: {csv_path}")

    # export both columns
    for source, d, suffix in (("unifac", u, "unifac"), ("thermoformer", t, "tf")):
        if d is None:
            print(f"[skip] {source}: design unavailable")
            continue
        dest = OUTDIR / f"ipa_water_binary_column_x0p3_{suffix}.dwxmz"
        export_dwsim_binary_column(
            components=COMPONENTS,
            feed_composition=FEED,
            feed_flow_mol_s=1.0,
            feed_temperature_K=float(d.condenser_temperature_K) + 1.0,
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
        print(f"[OK] wrote {source}-source column: {dest}")


if __name__ == "__main__":
    main()
