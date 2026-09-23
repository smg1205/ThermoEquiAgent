"""Verify the pressure drop and feed-stage bindings land in the generated files."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_engine.column_design import (  # noqa: E402
    bubble_temperature,
    design_binary_distillation_column,
    design_ternary_extractive_column,
)
from thermo_engine.dwsim_export import (  # noqa: E402
    export_dwsim_binary_column,
    export_generic_extractive_column,
)

OUT = ROOT / "report" / "dwsim"
P_KPA = 101.325
DP_KPA = 5.0  # explicit request-supplied total column pressure drop


def main():
    # ---------------- binary VLE ----------------
    light, heavy = "isopropanol", "water"
    feed = [0.5, 0.5]
    feed_t = bubble_temperature([light, heavy], feed, P_KPA, alpha_source="unifac")
    d = design_binary_distillation_column(
        [light, heavy], feed,
        feed_flow_mol_s=1.0, feed_temperature_K=feed_t,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=0.995, recovery=0.98,
        alpha_source="unifac",
    )
    print(f"[binary] N={d.theoretical_stages} R={d.reflux_ratio:.3f} feed_stage={d.feed_stage} "
          f"dP={DP_KPA} kPa")
    p1 = OUT / "verify_binary_ipa_water_dp.dwxmz"
    export_dwsim_binary_column(
        components=[light, heavy],
        feed_composition=feed,
        feed_flow_mol_s=1.0,
        feed_temperature_K=feed_t,
        operating_pressure_kPa=P_KPA,
        stages=d.theoretical_stages,
        minimum_stages=d.minimum_stages,
        reflux_ratio=d.reflux_ratio,
        minimum_reflux_ratio=d.minimum_reflux_ratio,
        feed_stage=d.feed_stage,
        condenser_temperature_K=d.condenser_temperature_K,
        reboiler_temperature_K=d.reboiler_temperature_K,
        destination=p1,
        pressure_drop_kPa=DP_KPA,
    )
    print("  wrote:", p1.name)

    # ---------------- ternary VLE (extractive) ----------------
    lt, hv, ent = "ethyl acetate", "n-propyl acetate", "dimethyl sulfoxide"
    tfeed = [0.5, 0.5]
    des = design_ternary_extractive_column(
        lt, hv, ent,
        feed_composition=tfeed, feed_flow_mol_s=1.0,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=0.95, recovery=0.90,
        entrainer_ratio=2.0, alpha_source="unifac",
    )
    print(f"[ternary] N={des['theoretical_stages']} R={des['reflux_ratio']:.3f} "
          f"feed_stage={des['feed_stage']} entrainer_stage={des['entrainer_stage']} dP={DP_KPA} kPa")
    p2 = OUT / "verify_extractive_eac_npac_dmso_dp.dwxmz"
    export_generic_extractive_column(
        light=lt, heavy=hv, entrainer=ent,
        feed_composition=tfeed,
        feed_flow_mol_s=1.0,
        feed_temperature_K=float(des["feed_temperature_K"]),
        feed_pressure_kPa=float(des["operating_pressure_kPa"]),
        stages=int(des["theoretical_stages"]),
        reflux_ratio=float(des["reflux_ratio"]),
        feed_stage=int(des["feed_stage"]),
        entrainer_stage=int(des["entrainer_stage"]),
        entrainer_ratio=2.0,
        condenser_temperature_K=float(des["condenser_temperature_K"]),
        reboiler_temperature_K=float(des["reboiler_temperature_K"]),
        property_package="UNIQUAC",
        destination=p2,
        pressure_drop_kPa=DP_KPA,
    )
    print("  wrote:", p2.name)


if __name__ == "__main__":
    main()
