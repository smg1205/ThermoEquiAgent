"""Generate a 2-propanol / water binary distillation column (x_IPA = 0.5) for testing.

Uses the deterministic short-cut design (Fenske-Underwood-Gilliland) and exports a
native DWSIM rigorous column with an explicit column pressure drop and the feed
bound to its designated tray.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine.column_design import (  # noqa: E402
    bubble_temperature,
    design_binary_distillation_column,
)
from thermo_engine.dwsim_export import export_dwsim_binary_column  # noqa: E402

P_KPA = 101.325
DP_KPA = 5.0          # explicit request-supplied total column pressure drop
X_IPA = 0.5
PURITY = 0.995
RECOVERY = 0.98
OUT = ROOT / "report" / "dwsim" / "ipa_water_binary_column_x0p5.dwxmz"


def main():
    components = ["isopropanol", "water"]
    feed = [X_IPA, 1.0 - X_IPA]

    feed_t = bubble_temperature(components, feed, P_KPA, alpha_source="unifac")
    design = design_binary_distillation_column(
        components,
        feed,
        feed_flow_mol_s=1.0,
        feed_temperature_K=feed_t,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY,
        recovery=RECOVERY,
        alpha_source="unifac",
    )

    print("=== 2-propanol / water binary distillation design (x_IPA = 0.5) ===")
    print(f"  feed            : {design.feed_composition}  flow 1.0 mol/s  T = {feed_t:.2f} K")
    print(f"  pressure        : {design.operating_pressure_kPa} kPa")
    print(f"  pressure drop   : {DP_KPA} kPa")
    print(f"  alpha           : {design.relative_volatility:.4f}")
    print(f"  stages / feed   : {design.theoretical_stages} / {design.feed_stage}")
    print(f"  N_min           : {design.minimum_stages:.4f}")
    print(f"  reflux ratio    : {design.reflux_ratio:.4f}  (min {design.minimum_reflux_ratio:.4f})")
    print(f"  T condenser     : {design.condenser_temperature_K:.2f} K")
    print(f"  T reboiler      : {design.reboiler_temperature_K:.2f} K")
    print(f"  D / B           : {design.distillate_flow_mol_s:.4f} / {design.bottoms_flow_mol_s:.4f} mol/s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    export_dwsim_binary_column(
        components=components,
        feed_composition=feed,
        feed_flow_mol_s=1.0,
        feed_temperature_K=feed_t,
        operating_pressure_kPa=design.operating_pressure_kPa,
        stages=design.theoretical_stages,
        minimum_stages=design.minimum_stages,
        reflux_ratio=design.reflux_ratio,
        minimum_reflux_ratio=design.minimum_reflux_ratio,
        feed_stage=design.feed_stage,
        condenser_temperature_K=design.condenser_temperature_K,
        reboiler_temperature_K=design.reboiler_temperature_K,
        destination=OUT,
        pressure_drop_kPa=DP_KPA,
    )
    print(f"\n[OK] wrote: {OUT}")


if __name__ == "__main__":
    main()
