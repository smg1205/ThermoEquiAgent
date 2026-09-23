"""Create the DWSIM direct binary-distillation flowsheet for heptane/nonane.

The feed is an actual 101.325 kPa NIST ThermoML point, x(heptane)=0.466.
DWSIM UNIQUAC supplies the VLE values used for the design; the short-cut
Fenske/Underwood/Gilliland calculation writes a fully specified DWSIM binary
column (one feed, distillate and bottoms; no extraction solvent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_heptane_nonane_dwsim as flash_case
from scripts import generate_ternary_dwsim_demonstration as demo
from thermo_engine import dwsim_export as ded
from thermo_engine.column_design import BinaryDistillationDesign, design_binary_distillation_column
from thermo_engine.dwsim_export import export_dwsim_binary_column

LIGHT = "heptane"
HEAVY = "nonane"
P_KPA = 101.325
FEED_FLOW_MOL_S = 1.0
X_FEED_LIGHT = 0.466
DISTILLATE_PURITY = 0.995
LIGHT_RECOVERY = 0.98


def _setup_dwsim() -> None:
    factory, object_type = ded._automation_factory()
    demo._GLOBALS.automation = factory()
    demo._GLOBALS.object_type = object_type
    demo._GLOBALS.system = flash_case.SYSTEM
    demo._GLOBALS.property_package = "UNIQUAC"


def _bubble_temperature_and_y(x_light: float) -> tuple[float, list[float]]:
    if not 0.0 < x_light < 1.0:
        raise ValueError("bubble composition must lie strictly inside (0, 1)")
    t_c, y, _vapor_fraction = demo._bubble([x_light, 1.0 - x_light])
    if len(y) != 2 or any(value != value for value in y):
        raise RuntimeError("DWSIM returned an invalid binary vapor composition")
    return float(t_c) + 273.15, [float(y[0]), float(y[1])]


def _product_compositions() -> tuple[float, float, float, float]:
    n_light_feed = FEED_FLOW_MOL_S * X_FEED_LIGHT
    d_light = LIGHT_RECOVERY * n_light_feed
    d_heavy = (1.0 - DISTILLATE_PURITY) / DISTILLATE_PURITY * d_light
    distillate_flow = d_light + d_heavy
    bottoms_flow = FEED_FLOW_MOL_S - distillate_flow
    if bottoms_flow <= 0.0:
        raise ValueError("product specification produces a non-positive bottoms flow")
    return (
        d_light / distillate_flow,
        (n_light_feed - d_light) / bottoms_flow,
        distillate_flow,
        bottoms_flow,
    )


def _design() -> tuple[BinaryDistillationDesign, float, dict[str, object]]:
    _setup_dwsim()
    x_distillate, x_bottoms, d_balance, b_balance = _product_compositions()
    t_feed, y_feed = _bubble_temperature_and_y(X_FEED_LIGHT)
    t_cond, y_distillate = _bubble_temperature_and_y(x_distillate)
    t_reb, y_bottoms = _bubble_temperature_and_y(x_bottoms)
    alpha = (y_feed[0] / X_FEED_LIGHT) / (y_feed[1] / (1.0 - X_FEED_LIGHT))
    if alpha <= 1.0:
        raise RuntimeError(f"DWSIM UNIQUAC does not identify {LIGHT} as the light key: alpha={alpha:.6g}")

    design = design_binary_distillation_column(
        [LIGHT, HEAVY], [X_FEED_LIGHT, 1.0 - X_FEED_LIGHT],
        feed_flow_mol_s=FEED_FLOW_MOL_S, feed_temperature_K=t_feed,
        operating_pressure_kPa=P_KPA, distillate_purity_mole_fraction=DISTILLATE_PURITY,
        recovery=LIGHT_RECOVERY, relative_volatility_override=alpha,
        condenser_temperature_K_override=t_cond, reboiler_temperature_K_override=t_reb,
    )
    if abs(design.distillate_flow_mol_s - d_balance) > 1e-5 or abs(design.bottoms_flow_mol_s - b_balance) > 1e-5:
        raise RuntimeError("short-cut design product balance differs from the requested balance")

    provenance: dict[str, object] = {
        "system": "n-heptane / n-nonane",
        "experimental_source": "NIST ThermoML DOI 10.1016/j.fluid.2013.05.016",
        "mode": "direct binary VLE distillation; no extraction solvent",
        "property_package": "DWSIM UNIQUAC",
        "pressure_kPa": P_KPA,
        "feed": {"flow_mol_s": FEED_FLOW_MOL_S, "x_heptane": X_FEED_LIGHT},
        "product_targets": {"distillate_heptane_mole_fraction": DISTILLATE_PURITY, "heptane_recovery_to_distillate": LIGHT_RECOVERY},
        "dwsim_bubble_vle": {
            "feed": {"temperature_K": t_feed, "vapor_composition": y_feed, "relative_volatility": alpha},
            "distillate_target": {"x_heptane": x_distillate, "temperature_K": t_cond, "vapor_composition": y_distillate},
            "bottoms_target": {"x_heptane": x_bottoms, "temperature_K": t_reb, "vapor_composition": y_bottoms},
        },
        "shortcut_column_design": {
            "theoretical_stages": design.theoretical_stages, "minimum_stages": design.minimum_stages,
            "feed_stage": design.feed_stage, "minimum_reflux_ratio": design.minimum_reflux_ratio,
            "reflux_ratio": design.reflux_ratio, "distillate_flow_mol_s": design.distillate_flow_mol_s,
            "bottoms_flow_mol_s": design.bottoms_flow_mol_s, "assumptions": design.assumptions,
        },
    }
    return design, t_feed, provenance


def main() -> int:
    outdir = ROOT / "report" / "success" / "正庚烷-正壬烷"
    outdir.mkdir(parents=True, exist_ok=True)
    design, t_feed, provenance = _design()
    flowsheet = export_dwsim_binary_column(
        components=[LIGHT, HEAVY], feed_composition=list(design.feed_composition),
        feed_flow_mol_s=design.feed_flow_mol_s, feed_temperature_K=t_feed,
        operating_pressure_kPa=design.operating_pressure_kPa, stages=design.theoretical_stages,
        minimum_stages=design.minimum_stages, reflux_ratio=design.reflux_ratio,
        minimum_reflux_ratio=design.minimum_reflux_ratio, feed_stage=design.feed_stage,
        condenser_temperature_K=design.condenser_temperature_K, reboiler_temperature_K=design.reboiler_temperature_K,
        distillate_flow_mol_s=design.distillate_flow_mol_s, bottoms_flow_mol_s=design.bottoms_flow_mol_s,
        destination=outdir / "heptane_nonane_binary_distillation_x0p466.dwxmz",
    )
    metadata = outdir / "heptane_nonane_binary_distillation_x0p466_design.json"
    metadata.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {flowsheet}")
    print(f"wrote: {metadata}")
    print(f"DWSIM UNIQUAC alpha={design.relative_volatility:.6f}; N={design.theoretical_stages}; R={design.reflux_ratio:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
