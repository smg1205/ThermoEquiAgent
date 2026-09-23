"""Export DWSIM column .dwxmz files from the real UNIFAC and ThermoFormer designs.

Unlike ``generate_ipa_water_std_column_tf.py`` / ``generate_eac_npac_dmso_extractive_tf.py``
-- which carry the "TF" design numbers as hard-coded literals and only take the feed
temperature from UNIFAC -- this script runs the deterministic short-cut design with
``alpha_source="thermoformer"`` (registry default seed_0 checkpoint) and exports the
resulting stage/reflux/temperature numbers to DWSIM.

Exports, per case:
    <case>_unifac.dwxmz   alpha_source = "unifac"
    <case>_tf.dwxmz       alpha_source = "thermoformer"

All equilibrium/design values come from ``thermo_engine``; this script only forwards
them to the DWSIM exporter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.pop("THERMOFORMER_CHECKPOINT", None)
os.environ["THERMOFORMER_SRC"] = r"E:\codex\ThermoAgent\ThermoFormer\ThermoAgent\lab_models\ThermoFormer\src"
os.environ["THERMOFORMER_USE_CUDA"] = "0"

from thermo_engine.column_design import (  # noqa: E402
    bubble_temperature,
    design_binary_distillation_column,
    design_ternary_extractive_column,
)
from thermo_engine.dwsim_export import (  # noqa: E402
    export_dwsim_binary_column,
    export_generic_extractive_column,
)

OUTDIR = ROOT / "data" / "exports" / "flow_examples"
P_KPA = 101.325
F_MOL_S = 1.0
SOURCES = ("unifac", "thermoformer")

IPA_X = 0.3
PURITY_C1 = 0.995
RECOVERY_C1 = 0.98

PURITY_C2 = 0.95
RECOVERY_C2 = 0.90
EAC = dict(light="ethyl acetate", heavy="n-propyl acetate", entrainer="dmso")
ENTRAINER_RATIO = 2.0


def case1(src: str) -> Path:
    d = design_binary_distillation_column(
        ["isopropanol", "water"], [IPA_X, 1.0 - IPA_X],
        feed_flow_mol_s=F_MOL_S, feed_temperature_K=None,
        operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY_C1, recovery=RECOVERY_C1,
        alpha_source=src,
    )
    # Liquid feed saturated on its bubble point; UNIFAC is the physical reference
    # for the feed thermal state (the design's own source supplies alpha and the
    # top/bottom bubble temperatures).
    feed_t = bubble_temperature(["isopropanol", "water"], [IPA_X, 1.0 - IPA_X], P_KPA, "unifac")
    dest = OUTDIR / f"ipa_water_binary_column_x0p3_{src}.dwxmz"
    export_dwsim_binary_column(
        components=["isopropanol", "water"],
        feed_composition=[IPA_X, 1.0 - IPA_X],
        feed_flow_mol_s=F_MOL_S,
        feed_temperature_K=feed_t,
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
        pressure_drop_kPa=5.0,
        solving_method="Naphtali-Sandholm",
        max_iterations=500,
    )
    print(f"  [{src}] alpha={d.relative_volatility:.4f} N={d.theoretical_stages} "
          f"Nmin={d.minimum_stages:.3f} R={d.reflux_ratio:.3f} feed={d.feed_stage} "
          f"Tcond={d.condenser_temperature_K:.2f} -> {dest.name}")
    return dest


def case2(src: str) -> Path:
    d = design_ternary_extractive_column(
        EAC["light"], EAC["heavy"], EAC["entrainer"], [0.5, 0.5],
        feed_flow_mol_s=F_MOL_S, operating_pressure_kPa=P_KPA,
        distillate_purity_mole_fraction=PURITY_C2, recovery=RECOVERY_C2,
        entrainer_ratio=ENTRAINER_RATIO, alpha_source=src,
    )
    dest = OUTDIR / f"eac_npac_dmso_extractive_{src}.dwxmz"
    export_generic_extractive_column(
        light=EAC["light"], heavy=EAC["heavy"], entrainer=EAC["entrainer"],
        feed_composition=[0.5, 0.5],
        feed_flow_mol_s=F_MOL_S,
        feed_temperature_K=d["feed_temperature_K"],
        feed_pressure_kPa=P_KPA,
        stages=d["theoretical_stages"],
        reflux_ratio=d["reflux_ratio"],
        feed_stage=d["feed_stage"],
        entrainer_stage=d["entrainer_stage"],
        entrainer_ratio=ENTRAINER_RATIO,
        condenser_temperature_K=d["condenser_temperature_K"],
        reboiler_temperature_K=d["reboiler_temperature_K"],
        property_package="UNIQUAC",
        destination=dest,
    )
    print(f"  [{src}] a_base={d['alpha_base']} a_ext={d['alpha_ext']} sel={d['selectivity']} "
          f"N={d['theoretical_stages']} R={d['reflux_ratio']} -> {dest.name}")
    return dest


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=== case 1: isopropanol / water, x_IPA = 0.3 ===")
    for src in SOURCES:
        try:
            case1(src)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{src}] FAILED {type(exc).__name__}: {exc}")
    print("=== case 2: ethyl acetate / n-propyl acetate + DMSO ===")
    for src in SOURCES:
        try:
            case2(src)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{src}] FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
