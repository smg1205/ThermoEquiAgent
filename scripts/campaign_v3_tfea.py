"""ThermoFormer vs UNIFAC bubble prediction + extractive column design campaign
for the ternary 乙酸正丙酯 (n-propyl acetate) + 乙酸乙酯 (ethyl acetate) + 二甲基亚砜
(dimethyl sulfoxide, DMSO) system.

All equilibrium numbers come from the deterministic thermo_engine seams:
- ThermoFormer bubble predictions via thermo_engine.thermoformer_backend
  (ThermoFormerBackend.bubble_point, isothermal P-x-y / isobaric T-x-y).
- UNIFAC bubble temperature via thermo_engine.column_design.bubble_temperature.
- UNIFAC isothermal bubble pressure via unifac_bubble_pressure (UNIFAC gammas
  + the same vapor-pressure source as bubble_temperature).
- Column design via design_binary_distillation_column /
  design_ternary_extractive_column.

Prints a compact machine-readable summary (JSON) for the report authoring step.
"""
import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(WORKSPACE, "lab_models", "ThermoFormer", "src")
CKPT = os.path.join(
    WORKSPACE, "lab_models", "ThermoFormer", "checkpoints", "multiview",
    "chemical_attention", "formal", "c0_current_vanilla.on.overall_binary_ternary",
    "seed_0", "best_model.pt",
)
CACHE = os.path.join(WORKSPACE, ".thermoformer-cache")

os.environ.setdefault("THERMOFORMER_SRC", SRC)
os.environ.setdefault("THERMOFORMER_CHECKPOINT", CKPT)
os.environ.setdefault("THERMOFORMER_FEATURE_CACHE", CACHE)
os.environ.setdefault("THERMOFORMER_USE_CUDA", "0")
sys.path.insert(0, WORKSPACE)

from thermo import Chemical  # noqa: E402
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions  # noqa: E402
from thermo_engine.column_design import (  # noqa: E402
    bubble_temperature,
    design_binary_distillation_column,
    design_ternary_extractive_column,
    unifac_bubble_pressure,
)
from thermo_engine.thermoformer_backend import ThermoFormerBackend  # noqa: E402

LIGHT = "ethyl acetate"      # 乙酸乙酯 (more volatile, overhead)
HEAVY = "propyl acetate"     # 乙酸正丙酯 (bottoms)
ENT = "dimethyl sulfoxide"   # 二甲基亚砜 (extractive solvent)
ORDER = ["propyl acetate", "ethyl acetate", "dimethyl sulfoxide"]
SMILES = {"propyl acetate": "CCC(=O)OC", "ethyl acetate": "CC(=O)OCC",
          "dimethyl sulfoxide": "CS(=O)C"}
P_ATM = 101.325

backend = ThermoFormerBackend()


def ids(comps):
    return [ComponentIdentity(component_id=c, name=c, smiles=SMILES[c], aliases=[])
            for c in comps]


# --------------------------------------------------------------------------- #
# ThermoFormer bubble-point single calculation
# --------------------------------------------------------------------------- #
def tf_bubble(comps, x, T=None, P=None):
    cond = ThermodynamicConditions(
        temperature_K=T, pressure_kPa=P, liquid_composition=[float(v) for v in x]
    )
    req = TaskManifest(
        equilibrium_type="VLE", calculation_type="bubble_point",
        components=ids(comps), conditions=cond, model_name="ThermoFormer",
    )
    res = backend.bubble_point(req)
    pt = res.points[0]
    return {
        "T": pt.temperature_K,
        "P": pt.pressure_kPa,
        "y": [float(v) for v in pt.vapor_composition],
        "residual": pt.equilibrium_residual,
    }


# --------------------------------------------------------------------------- #
# UNIFAC reference: bubble temperature (isobaric) and bubble pressure (isothermal)
# --------------------------------------------------------------------------- #
def unifac_bubble_T(comps, x, P):
    try:
        return bubble_temperature(comps, x, P, "unifac"), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def unifac_bubble_P(comps, x, T):
    """P_bubble = sum_i x_i * gamma_i(T) * Psat_i(T)  (modified Raoult's law)."""
    try:
        P = unifac_bubble_pressure(comps, x, T)
        # Vapor composition at the bubble condition: y_i = x_i*gamma_i*Psat_i / P
        from thermo_engine.column_design import activity_coefficients
        gamma = activity_coefficients(comps, x, T)
        P_pa = P * 1000.0
        ys = [x[i] * gamma[i] * Chemical(comps[i]).VaporPressure(T) / P_pa
              for i in range(len(comps))]
        return P, ys, None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


if __name__ == "__main__":
    out = {}

    # ---- Isothermal ternary P-x-y at equimolar composition -----------------
    iso_P = []
    for T in (330.0, 340.0, 350.0, 360.0, 370.0):
        x = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        tf = tf_bubble(ORDER, x, T=T)
        pu, pu_y, err = unifac_bubble_P(ORDER, x, T)
        iso_P.append({
            "T": T, "P_TF": round(tf["P"], 4), "y_TF": [round(v, 4) for v in tf["y"]],
            "P_UNIFAC": None if pu is None else round(pu, 4),
            "y_UNIFAC": None if pu_y is None else [round(v, 4) for v in pu_y],
            "residual_kpa": round(tf["residual"], 4), "err": err,
        })
    out["isothermal_ternary_equimolar"] = iso_P

    # ---- Isobaric ternary T-x-y at nearest equimolar + extractive-rich ----
    iso_T = []
    for x in ([0.30, 0.30, 0.40], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
              [0.25, 0.25, 0.50], [0.20, 0.20, 0.60], [0.10, 0.10, 0.80]):
        tf = tf_bubble(ORDER, x, P=P_ATM)
        tu, err = unifac_bubble_T(ORDER, x, P_ATM)
        iso_T.append({
            "x": [round(v, 4) for v in x],
            "T_TF": round(tf["T"], 4), "y_TF": [round(v, 4) for v in tf["y"]],
            "T_UNIFAC": None if tu is None else round(tu, 4),
            "residual_kpa": round(tf["residual"], 4), "err": err,
        })
    out["isobaric_ternary"] = iso_T

    # ---- Binary reference (no entrainer) for context ------------------------
    bin_iso = []
    for T in (340.0, 350.0, 360.0, 370.0):
        x = [0.5, 0.5]
        comps = [LIGHT, HEAVY]
        tf = tf_bubble(comps, x, T=T)
        pu, pu_y, err = unifac_bubble_P(comps, x, T)
        bin_iso.append({
            "T": T, "P_TF": round(tf["P"], 4), "y_TF": [round(v, 4) for v in tf["y"]],
            "P_UNIFAC": None if pu is None else round(pu, 4),
            "y_UNIFAC": None if pu_y is None else [round(v, 4) for v in pu_y],
            "err": err,
        })
    out["isothermal_binary_ea_nproac"] = bin_iso

    # ---- Column design: plain binary (EA / n-PrOAc) -------------------------
    for src in ("unifac", "thermoformer"):
        try:
            d = design_binary_distillation_column(
                [LIGHT, HEAVY], [0.5, 0.5], feed_flow_mol_s=1.0,
                feed_temperature_K=None, operating_pressure_kPa=P_ATM,
                distillate_purity_mole_fraction=0.995, recovery=0.98,
                alpha_source=src,
            )
            out.setdefault("binary_column", {})[src] = {
                "relative_volatility": d.relative_volatility,
                "minimum_stages": d.minimum_stages,
                "minimum_reflux_ratio": d.minimum_reflux_ratio,
                "reflux_ratio": d.reflux_ratio,
                "theoretical_stages": d.theoretical_stages,
                "feed_stage": d.feed_stage,
                "condenser_temperature_K": d.condenser_temperature_K,
                "reboiler_temperature_K": d.reboiler_temperature_K,
            }
        except Exception as exc:  # noqa: BLE001
            out.setdefault("binary_column", {})[src] = {"error": repr(exc)}

    # ---- Column design: ternary extractive (EA / n-PrOAc + DMSO) -----------
    for src in ("unifac", "thermoformer"):
        try:
            d = design_ternary_extractive_column(
                LIGHT, HEAVY, ENT, [0.5, 0.5], feed_flow_mol_s=1.0,
                operating_pressure_kPa=P_ATM, distillate_purity_mole_fraction=0.995,
                recovery=0.98, entrainer_ratio=2.0, alpha_source=src,
            )
            out.setdefault("extractive_column", {})[src] = d
        except Exception as exc:  # noqa: BLE001
            out.setdefault("extractive_column", {})[src] = {"error": repr(exc)}

    # ---- Selectivity for the recommended extractive candidate --------------
    print(json.dumps(out, ensure_ascii=False, indent=2))
