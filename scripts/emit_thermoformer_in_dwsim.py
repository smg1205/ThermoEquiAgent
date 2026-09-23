"""Three ways to surface ThermoFormer predictions inside DWSIM ``.dwxmz`` files.

DWSIM knows only *property packages plus binary interaction parameters (BIPs)*.
It has no hook for calling an external ML model, so a ThermoFormer prediction
cannot be "executed" by DWSIM directly.  What CAN be done -- and what this script
demonstrates -- is three graded ways of carrying the prediction into the file:

  Route A  ``tf_fitted_bip``      Regress NRTL (A12, A21, alpha) from the
                                  ThermoFormer-predicted tie-lines and write those
                                  BIPs into the property package.  The DWSIM GUI
                                  then shows ThermoFormer-derived parameters in
                                  its binary-parameter table, and DWSIM solves
                                  the flash with them.

  Route B  ``tf_seeded_outlets``  Keep DWSIM's own parameters but ALSO pre-seed
                                  the two liquid outlets with the ThermoFormer
                                  coexistence compositions, so DWSIM starts from
                                  a good initial guess.

  Route C  ``tf_reference_stream`` Write the ThermoFormer prediction into the file
                                  as dedicated reference *material streams* (not
                                  connected to the solver) so the numbers travel
                                  with the document and can be screenshotted, while
                                  DWSIM still reports its own independent result.

Every number is sourced:

  * ThermoFormer tie-lines  <- ``report/dwsim/tf_rerun_water_butanol_lle.csv``
  * experimental tie-lines  <- ``report/dwsim/water_butanol_experimental.csv``
  * DWSIM's own parameters  <- DWSIM's built-in water/1-butanol NRTL pair

Nothing is invented here.  The regression in Route A fits *only* the
ThermoFormer-predicted tie-lines, never experiment and never hand-picked values.

Output: ``report/success/tf_<route>_<temperature>K.dwxmz`` plus a JSON summary.

Run with a DWSIM + pythonnet capable interpreter:

    python scripts/emit_thermoformer_in_dwsim.py
    python scripts/emit_thermoformer_in_dwsim.py --temperature 313.15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: DWSIM's internal pressure API takes Pa.
P_PA = 101325.0

#: Overall feed composition inside the two-phase region (1-butanol / water).
FEED_Z = [0.3, 0.7]

#: Feed molar flow, mol/s.
FEED_FLOW = 1.0

#: DWSIM compound keys, in the order used throughout this script.
COMPOUNDS = ["1-butanol", "Water"]

#: Gas constant, J/(mol K).
R = 8.314462618

TF_CSV = ROOT / "report" / "dwsim" / "tf_rerun_water_butanol_lle.csv"
EXP_CSV = ROOT / "report" / "dwsim" / "water_butanol_experimental.csv"
OUTDIR = ROOT / "report" / "success"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_thermoformer_tielines() -> list[dict[str, float]]:
    """ThermoFormer-predicted coexistence compositions, straight from the report CSV."""
    if not TF_CSV.is_file():
        raise SystemExit(f"ThermoFormer tie-line CSV not found: {TF_CSV}")
    rows: list[dict[str, float]] = []
    with open(TF_CSV, newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "T_K": float(raw["T_K"]),
                    "x_butanol_organic": float(raw["tf_x_butanol_organic"]),
                    "x_butanol_aqueous": float(raw["tf_x_butanol_aqueous"]),
                }
            )
    if not rows:
        raise SystemExit(f"ThermoFormer tie-line CSV is empty: {TF_CSV}")
    return rows


# --------------------------------------------------------------------------- #
# Route A -- regress NRTL from the ThermoFormer tie-lines
# --------------------------------------------------------------------------- #


def nrtl_ln_gamma(x1: float, a12: float, a21: float, alpha: float, t_k: float) -> tuple[float, float]:
    """ln(gamma_1), ln(gamma_2) for a symmetric-alpha binary NRTL.

    ``a12``/``a21`` are the energy terms in J/mol (tau_ij = A_ij / (R T)).
    """
    x2 = 1.0 - x1
    tau12 = a12 / (R * t_k)
    tau21 = a21 / (R * t_k)
    g12 = np.exp(-alpha * tau12)
    g21 = np.exp(-alpha * tau21)
    ln_g1 = x2 * x2 * (tau21 * (g21 / (x1 + x2 * g21)) ** 2 + tau12 * g12 / (x2 + x1 * g12) ** 2)
    ln_g2 = x1 * x1 * (tau12 * (g12 / (x2 + x1 * g12)) ** 2 + tau21 * g21 / (x1 + x2 * g21) ** 2)
    return float(ln_g1), float(ln_g2)


def fit_nrtl_from_tielines(tielines: list[dict[str, float]]) -> dict[str, float]:
    """Least-squares NRTL fit to the supplied tie-lines via equal-activity residuals.

    LLE requires ``x_i * gamma_i`` to match in both phases for every component;
    the residual is that mismatch in log space.  Fed only ThermoFormer predictions
    here, so the resulting parameters are a *representation* of the ML model.
    """

    def residual(params: np.ndarray) -> list[float]:
        a12, a21, alpha = params
        out: list[float] = []
        for row in tielines:
            t_k = row["T_K"]
            x_org = row["x_butanol_organic"]
            x_aq = row["x_butanol_aqueous"]
            lg1o, lg2o = nrtl_ln_gamma(x_org, a12, a21, alpha, t_k)
            lg1a, lg2a = nrtl_ln_gamma(x_aq, a12, a21, alpha, t_k)
            out.append((np.log(x_org) + lg1o) - (np.log(x_aq) + lg1a))
            out.append((np.log(1 - x_org) + lg2o) - (np.log(1 - x_aq) + lg2a))
        return out

    solution = least_squares(
        residual,
        np.array([5000.0, 5000.0, 0.3]),
        bounds=([-50000.0, -50000.0, 0.05], [50000.0, 50000.0, 0.9]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=200000,
    )
    a12, a21, alpha = (float(v) for v in solution.x)
    return {
        "A12_J_per_mol": a12,
        "A21_J_per_mol": a21,
        "alpha12": alpha,
        # DWSIM stores NRTL energies in cal/mol.
        "A12_cal_per_mol": a12 / 4.184,
        "A21_cal_per_mol": a21 / 4.184,
        "residual_norm": float(np.linalg.norm(solution.fun)),
    }


def write_nrtl_bips(flowsheet: object, property_package: object, fit: dict[str, float]) -> str:
    """Write fitted NRTL BIPs into DWSIM's ``NRTL_IPData`` in both directions.

    Returns a short status string.  Two extra fields matter:

    * ``AutoEstimateMissingNRTLUNIQUACParameters = False`` -- otherwise DWSIM
      re-estimates and silently discards what we wrote.
    * ``AreModelParametersDirty = True`` + ``ConfigParameters()`` -- forces the
      package to pick the new values up.
    """
    from System import Activator  # type: ignore[import-not-found]

    selected = flowsheet.SelectedCompounds
    compounds = {str(key): selected[key] for key in list(selected.Keys)}

    def resolve(fragment: str) -> str | None:
        for key in compounds:
            if fragment.casefold() in key.casefold():
                return key
        return None

    butanol_key = resolve("butanol")
    water_key = resolve("water")
    if butanol_key is None or water_key is None:
        raise RuntimeError(f"could not resolve the compound keys from {sorted(compounds)}")

    pkg_type = property_package.GetType()
    m_uni = pkg_type.GetProperty("m_uni").GetValue(property_package, None)
    interactions = m_uni.GetType().GetProperty("InteractionParameters").GetValue(m_uni, None)
    inner_type = interactions.GetType().GetGenericArguments()[1]
    data_type = m_uni.GetType().Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData"
    )
    if data_type is None:
        raise RuntimeError("DWSIM's NRTL_IPData type was not found in the loaded assemblies")

    a12 = fit["A12_cal_per_mol"]
    a21 = fit["A21_cal_per_mol"]
    alpha = fit["alpha12"]

    for first, second, a_ij, a_ji in (
        (butanol_key, water_key, a12, a21),
        (water_key, butanol_key, a21, a12),
    ):
        if not interactions.ContainsKey(first):
            interactions.Add(first, Activator.CreateInstance(inner_type))
        inner = interactions[first]
        if not inner.ContainsKey(second):
            inner.Add(second, Activator.CreateInstance(data_type))
        data = inner[second]
        data.ID1 = compounds[first]
        data.ID2 = compounds[second]
        data.A12 = float(a_ij)
        data.A21 = float(a_ji)
        data.alpha12 = float(alpha)
        data.B12 = 0.0
        data.B21 = 0.0
        data.C12 = 0.0
        data.C21 = 0.0
        data.comment = "NRTL regressed from ThermoFormer-predicted water/1-butanol tie-lines"

    pkg_type.GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(
        property_package, False, None
    )
    try:
        pkg_type.GetProperty("AreModelParametersDirty").SetValue(property_package, True, None)
    except Exception:  # noqa: BLE001 - property absent on some builds
        pass
    configure = pkg_type.GetMethod("ConfigParameters")
    if configure is not None:
        configure.Invoke(property_package, None)

    return (
        f"wrote NRTL BIPs A12={a12:.2f} A21={a21:.2f} alpha={alpha:.4f} cal/mol "
        f"(AutoEstimate disabled)"
    )


# --------------------------------------------------------------------------- #
# Flowsheet construction
# --------------------------------------------------------------------------- #


def build_lle_vessel(
    automation: object,
    object_type: object,
    temperature_k: float,
    *,
    feed_streams: list[tuple[str, list[float], float]] | None = None,
    seed_outlets: tuple[list[float], list[float]] | None = None,
):
    """Build ``Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid``.

    ``feed_streams`` overrides the single default feed; used by Route C to add
    extra, *unconnected* reference streams.  ``seed_outlets`` pre-seeds the two
    liquid outlets with (light, heavy) compositions -- Route B.
    """
    from thermo_engine.dwsim_export import _add_property_package, _composition_argument, _simulation_object

    flowsheet = automation.CreateFlowsheet()
    for name in COMPOUNDS:
        flowsheet.AddCompound(name)
    _add_property_package(flowsheet, "NRTL")

    streams = feed_streams or [("Feed", FEED_Z, FEED_FLOW)]
    feed_go = None
    for index, (tag, composition, flow) in enumerate(streams):
        handle = flowsheet.AddObject(object_type.MaterialStream, 0, 150 - 90 * index, tag)
        stream = _simulation_object(handle)
        stream.SetTemperature(temperature_k)
        stream.SetPressure(P_PA)
        stream.SetMolarFlow(flow)
        stream.SetOverallComposition(_composition_argument(list(composition)))
        if index == 0:
            feed_go = handle

    vessel_go = flowsheet.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vapor_go = flowsheet.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = flowsheet.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = flowsheet.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")

    vessel = _simulation_object(vessel_go)
    for attribute, value in (
        ("FlashTemperature", temperature_k),
        ("FlashPressure", P_PA),
        ("OverrideT", False),
        ("OverrideP", False),
        ("CalculationMode", "Legacy"),
        ("MinimumPressure", P_PA),
    ):
        try:
            setattr(vessel, attribute, value)
        except Exception:  # noqa: BLE001 - non-fatal on some DWSIM builds
            pass

    if seed_outlets is not None:
        light_x, heavy_x = seed_outlets
        for handle, composition in ((light_go, light_x), (heavy_go, heavy_x)):
            stream = _simulation_object(handle)
            try:
                stream.SetTemperature(temperature_k)
                stream.SetPressure(P_PA)
                stream.SetMolarFlow(FEED_FLOW * 0.5)
                stream.SetOverallComposition(_composition_argument(list(composition)))
            except Exception:  # noqa: BLE001 - non-fatal
                pass
    for handle in (vapor_go, light_go, heavy_go):
        try:
            _simulation_object(handle).SetTemperature(temperature_k)
        except Exception:  # noqa: BLE001 - non-fatal
            pass

    assert feed_go is not None
    for source, target, source_index, target_index in (
        (feed_go, vessel_go, 0, 0),
        (vessel_go, vapor_go, 0, 0),
        (vessel_go, light_go, 1, 0),
        (vessel_go, heavy_go, 2, 0),
    ):
        try:
            flowsheet.ConnectObjects(source.GraphicObject, target.GraphicObject, source_index, target_index)
        except Exception:  # noqa: BLE001 - surface, don't abort
            pass

    return flowsheet, light_go, heavy_go


def read_split(light_go: object, heavy_go: object) -> dict[str, object]:
    """DWSIM's own phase result for the two liquid outlets."""
    from thermo_engine.dwsim_export import _simulation_object

    result: dict[str, object] = {}
    for label, handle in (("light", light_go), ("heavy", heavy_go)):
        try:
            stream = _simulation_object(handle)
            stream_type = stream.GetType()
            result[f"{label}_flow_mol_s"] = float(
                stream_type.GetMethod("GetMolarFlow").Invoke(stream, None)
            )
            result[f"{label}_composition"] = [
                float(v) for v in stream_type.GetMethod("GetOverallComposition").Invoke(stream, None)
            ]
        except Exception:  # noqa: BLE001 - depends on DWSIM assemblies
            result[f"{label}_flow_mol_s"] = None
            result[f"{label}_composition"] = None
    heavy_flow = result.get("heavy_flow_mol_s")
    result["separated"] = isinstance(heavy_flow, float) and heavy_flow > 1e-10
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=298.15, help="flash temperature, K")
    parser.add_argument("--outdir", default=str(OUTDIR), help="output directory")
    args = parser.parse_args()

    temperature_k = float(args.temperature)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tielines = load_thermoformer_tielines()
    available = [row["T_K"] for row in tielines]
    if temperature_k not in available:
        raise SystemExit(f"no ThermoFormer tie-line at {temperature_k} K; available: {available}")
    point = next(row for row in tielines if row["T_K"] == temperature_k)

    tf_light = [point["x_butanol_organic"], 1.0 - point["x_butanol_organic"]]
    tf_heavy = [point["x_butanol_aqueous"], 1.0 - point["x_butanol_aqueous"]]

    print("=" * 74)
    print(f"ThermoFormer in DWSIM -- water / 1-butanol @ {temperature_k} K")
    print("=" * 74)
    print(f"ThermoFormer tie-line : Light(butanol-rich) x = {[round(v, 4) for v in tf_light]}")
    print(f"                        Heavy(water-rich)   x = {[round(v, 4) for v in tf_heavy]}")

    print("\n--- fitting NRTL to the ThermoFormer tie-lines (Route A) ---")
    fit = fit_nrtl_from_tielines(tielines)
    print(
        f"  A12 = {fit['A12_cal_per_mol']:.3f} cal/mol   A21 = {fit['A21_cal_per_mol']:.3f} cal/mol   "
        f"alpha = {fit['alpha12']:.4f}   residual = {fit['residual_norm']:.3e}"
    )

    from thermo_engine.dwsim_export import _automation_factory, _save_flowsheet_via_temp

    factory, object_type = _automation_factory()
    automation = factory()

    summary: dict[str, object] = {
        "temperature_K": temperature_k,
        "thermoformer_tie_line": {"light": tf_light, "heavy": tf_heavy},
        "nrtl_fit_from_thermoformer": fit,
        "source_csv": str(TF_CSV.relative_to(ROOT)),
        "generated_at": datetime.now().isoformat(),
        "routes": {},
    }

    # ----------------------------- Route A --------------------------------- #
    print("\n--- Route A: ThermoFormer-fitted NRTL BIPs written into the package ---")
    flowsheet, light_go, heavy_go = build_lle_vessel(automation, object_type, temperature_k)
    package = list(flowsheet.PropertyPackages.Values)[0]
    status = write_nrtl_bips(flowsheet, package, fit)
    print(f"  {status}")
    automation.CalculateFlowsheet4(flowsheet)
    split_a = read_split(light_go, heavy_go)
    name_a = f"tf_fitted_bip_{str(temperature_k).replace('.', 'p')}K.dwxmz"
    _save_flowsheet_via_temp(automation, flowsheet, outdir / name_a)
    print(f"  DWSIM split: light={split_a['light_composition']} heavy={split_a['heavy_composition']}")
    print(f"  separated: {split_a['separated']}   saved: {name_a}")
    summary["routes"]["A_tf_fitted_bip"] = {
        "file": name_a,
        "nrtl_bips_cal_per_mol": {
            "A12": fit["A12_cal_per_mol"],
            "A21": fit["A21_cal_per_mol"],
            "alpha12": fit["alpha12"],
        },
        "dwsim_split": split_a,
    }

    # ----------------------------- Route B --------------------------------- #
    print("\n--- Route B: DWSIM built-ins, outlets seeded from the ThermoFormer tie-line ---")
    flowsheet, light_go, heavy_go = build_lle_vessel(
        automation, object_type, temperature_k, seed_outlets=(tf_light, tf_heavy)
    )
    automation.CalculateFlowsheet4(flowsheet)
    split_b = read_split(light_go, heavy_go)
    name_b = f"tf_seeded_outlets_{str(temperature_k).replace('.', 'p')}K.dwxmz"
    _save_flowsheet_via_temp(automation, flowsheet, outdir / name_b)
    print(f"  DWSIM split: light={split_b['light_composition']} heavy={split_b['heavy_composition']}")
    print(f"  separated: {split_b['separated']}   saved: {name_b}")
    summary["routes"]["B_tf_seeded_outlets"] = {
        "file": name_b,
        "seed_light": tf_light,
        "seed_heavy": tf_heavy,
        "dwsim_split": split_b,
    }

    # ----------------------------- Route C --------------------------------- #
    print("\n--- Route C: ThermoFormer tie-line carried as reference streams ---")
    reference_streams = [
        ("Feed", FEED_Z, FEED_FLOW),
        ("TF_Light_Reference", tf_light, 0.5),
        ("TF_Heavy_Reference", tf_heavy, 0.5),
    ]
    flowsheet, light_go, heavy_go = build_lle_vessel(
        automation, object_type, temperature_k, feed_streams=reference_streams
    )
    automation.CalculateFlowsheet4(flowsheet)
    split_c = read_split(light_go, heavy_go)
    name_c = f"tf_reference_stream_{str(temperature_k).replace('.', 'p')}K.dwxmz"
    _save_flowsheet_via_temp(automation, flowsheet, outdir / name_c)
    print(f"  reference streams: TF_Light_Reference x={[round(v, 4) for v in tf_light]}")
    print(f"                     TF_Heavy_Reference x={[round(v, 4) for v in tf_heavy]}")
    print(f"  DWSIM split: light={split_c['light_composition']} heavy={split_c['heavy_composition']}")
    print(f"  separated: {split_c['separated']}   saved: {name_c}")
    summary["routes"]["C_tf_reference_stream"] = {
        "file": name_c,
        "reference_streams": {
            "TF_Light_Reference": tf_light,
            "TF_Heavy_Reference": tf_heavy,
        },
        "dwsim_split": split_c,
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = outdir / f"thermoformer_in_dwsim_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote: {json_path}")
    print(f"files: {name_a}, {name_b}, {name_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
