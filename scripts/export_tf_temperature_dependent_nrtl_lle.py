"""Export a DWSIM binary-LLE file from a temperature-dependent TF→NRTL fit.

This is deliberately a *surrogate export*, not a claim that DWSIM executes
ThermoFormer.  The binary ThermoFormer LLE checkpoint defines a composition- and
temperature-dependent RK excess-Gibbs-energy surface.  This program samples its
``ln(gamma)`` surface for 1-butanol/water, regresses DWSIM's temperature-dependent
NRTL interaction energies, and lets a native DWSIM Vessel perform the final LLE
flash.

The companion JSON records the TF samples, fitted BIPs and the DWSIM read-back.
Only use the resulting property package over the sampled temperature interval.

Run from ``ThermoAgent`` with the DWSIM/pythonnet-capable interpreter:

    python scripts/export_tf_temperature_dependent_nrtl_lle.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import least_squares, root

ROOT = Path(__file__).resolve().parents[1]
TF_ROOT = ROOT.parent
TF_SRC = TF_ROOT / "src"
CHECKPOINT = TF_ROOT / "models" / "lle" / "prediction" / "binary-system" / "seed_0" / "best.pt"
OUTDIR = ROOT / "report" / "success"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TF_SRC) not in sys.path:
    # Keep it after the agent root: ThermoFormer/src contains ``thermo.py``,
    # which must not shadow the installed ``thermo`` package used by the DWSIM
    # exporter.
    sys.path.append(str(TF_SRC))

# DWSIM component ordering.  x1 is 1-butanol throughout this file.
COMPONENTS = ["1-butanol", "Water"]
SMILES = ["CCCCO", "O"]
P_KPA = 101.325
P_PA = P_KPA * 1000.0
T_GRID = np.array([298.15, 313.15, 343.15, 353.15], dtype=float)
X_GRID = np.r_[np.geomspace(1e-5, 1e-2, 5), np.linspace(0.02, 0.98, 25), 1.0 - np.geomspace(1e-2, 1e-5, 5)]
X_GRID = np.unique(np.clip(X_GRID, 1e-8, 1.0 - 1e-8))
T_REF = float(T_GRID.mean())
R_CAL = 1.9858775


def _load_tf() -> tuple[Any, dict[str, np.ndarray]]:
    """Load the checked-in LLE checkpoint and its two known feature vectors."""
    from thermoformer.lle_tp.hybrid_binodal_model import HybridBinodalLLE

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # ``decoder`` identifies the saved architecture but is not an __init__
    # argument of HybridBinodalLLE.
    spec = {key: value for key, value in payload["spec"].items() if key != "decoder"}
    model = HybridBinodalLLE(**spec).double()
    model.load_state_dict(payload["model"])
    model.eval()
    features = {key: value.detach().cpu().numpy() for key, value in payload["feature_map"].items()}
    missing = [smiles for smiles in SMILES if smiles not in features]
    if missing:
        raise RuntimeError(f"checkpoint is missing required molecular features: {missing}")
    return model, features


def _tf_coefficients(model: Any, features: dict[str, np.ndarray], temperature_k: float) -> tuple[float, float]:
    """Return TF's dimensionless binary RK coefficients at one T/P state."""
    dimension = len(features[SMILES[0]])
    molecules = np.zeros((1, 3, dimension), dtype=np.float64)
    molecules[0, :2] = np.stack([features[smiles] for smiles in SMILES])
    mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    with torch.no_grad():
        context = model.context(
            torch.as_tensor(molecules, dtype=torch.float64),
            torch.as_tensor([[temperature_k]], dtype=torch.float64),
            torch.as_tensor([[P_KPA]], dtype=torch.float64),
            torch.as_tensor(mask, dtype=torch.float64),
        )
    # HybridBinodalLLE stores the binary RK coefficients immediately before
    # the component mask in its context tuple.
    coeff = context[-2][0].detach().cpu().numpy()
    return float(coeff[0]), float(coeff[1])


def _tf_ln_gamma(x1: np.ndarray, c0: float, c1: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ln(gamma) of TF's binary RK ``gE/(RT)`` decoder.

    The equation is exactly the binary branch used by ThermoFormer's LLE model:
    ``gE/RT = x1*(1-x1)*(c0 + c1*(2*x1 - 1))``.
    """
    x = np.asarray(x1, dtype=float)
    g = x * (1.0 - x) * (c0 + c1 * (2.0 * x - 1.0))
    dg_dx = (1.0 - 2.0 * x) * (c0 + c1 * (2.0 * x - 1.0)) + 2.0 * c1 * x * (1.0 - x)
    return g + (1.0 - x) * dg_dx, g - x * dg_dx


def _tf_binodal(c0: float, c1: float) -> tuple[float, float] | None:
    """Find the TF coexistence pair through equal chemical potentials."""
    def residual(values: np.ndarray) -> np.ndarray:
        a, b = values
        if not (1e-8 < a < b < 1.0 - 1e-8):
            return np.array([100.0, 100.0])
        l1a, l2a = _tf_ln_gamma(np.array([a]), c0, c1)
        l1b, l2b = _tf_ln_gamma(np.array([b]), c0, c1)
        return np.array([
            np.log(a) + l1a[0] - np.log(b) - l1b[0],
            np.log(1.0 - a) + l2a[0] - np.log(1.0 - b) - l2b[0],
        ])

    candidates: list[tuple[float, float, float]] = []
    for guess in ((0.01, 0.99), (0.03, 0.6), (0.05, 0.5), (0.1, 0.8), (0.2, 0.7)):
        solution = root(residual, guess)
        if not solution.success:
            continue
        a, b = (float(v) for v in solution.x)
        error = float(np.max(np.abs(residual(solution.x))))
        if 1e-7 < a < b < 1.0 - 1e-7 and b - a > 0.005 and error < 1e-7:
            candidates.append((a, b, error))
    if not candidates:
        return None
    a, b, _ = min(candidates, key=lambda item: item[2])
    return a, b


def _nrtl_params_at_temperature(params: np.ndarray, temperature_k: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, float]:
    """Map stable fitting coordinates to DWSIM's A+B*T NRTL energy form."""
    e12_ref, e21_ref, slope12, slope21, alpha = params
    t = np.asarray(temperature_k, dtype=float)
    # Eij(T) = Aij + Bij*T.  Cij remains zero unless validation demonstrates
    # that the five-parameter model is insufficient.
    return e12_ref + slope12 * (t - T_REF), e21_ref + slope21 * (t - T_REF), float(alpha)


def _nrtl_ln_gamma(x1: np.ndarray, temperature_k: np.ndarray | float, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DWSIM-compatible binary NRTL ln(gamma), x1 = 1-butanol."""
    x = np.asarray(x1, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    e12, e21, alpha = _nrtl_params_at_temperature(params, t)
    tau12 = e12 / (R_CAL * t)
    tau21 = e21 / (R_CAL * t)
    g12 = np.exp(np.clip(-alpha * tau12, -700.0, 700.0))
    g21 = np.exp(np.clip(-alpha * tau21, -700.0, 700.0))
    x2 = 1.0 - x
    ln1 = x2**2 * (tau21 * (g21 / (x + x2 * g21)) ** 2 + tau12 * g12 / (x2 + x * g12) ** 2)
    ln2 = x**2 * (tau12 * (g12 / (x2 + x * g12)) ** 2 + tau21 * g21 / (x + x2 * g21) ** 2)
    return ln1, ln2


def _nrtl_binodal(params: np.ndarray, temperature_k: float) -> tuple[float, float] | None:
    """Return the NRTL coexistence pair in increasing x1 order."""
    def residual(values: np.ndarray) -> np.ndarray:
        a, b = values
        if not (1e-8 < a < b < 1.0 - 1e-8):
            return np.array([100.0, 100.0])
        l1a, l2a = _nrtl_ln_gamma(np.array([a]), temperature_k, params)
        l1b, l2b = _nrtl_ln_gamma(np.array([b]), temperature_k, params)
        return np.array([
            np.log(a) + l1a[0] - np.log(b) - l1b[0],
            np.log(1.0 - a) + l2a[0] - np.log(1.0 - b) - l2b[0],
        ])

    candidates: list[tuple[float, float, float]] = []
    for guess in ((0.01, 0.99), (0.03, 0.6), (0.05, 0.5), (0.1, 0.8), (0.2, 0.7)):
        solution = root(residual, guess)
        if not solution.success:
            continue
        a, b = (float(v) for v in solution.x)
        error = float(np.max(np.abs(residual(solution.x))))
        if 1e-7 < a < b < 1.0 - 1e-7 and b - a > 0.005 and error < 1e-7:
            candidates.append((a, b, error))
    if not candidates:
        return None
    a, b, _ = min(candidates, key=lambda item: item[2])
    return a, b


def _fit(samples: list[dict[str, Any]]) -> np.ndarray:
    """Fit ln(gamma) grids plus equal-activity constraints at TF tie-lines."""
    def residual(params: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for sample in samples:
            x = sample["x"]
            l1_tf, l2_tf = sample["tf_ln_gamma"]
            l1_nrtl, l2_nrtl = _nrtl_ln_gamma(x, sample["temperature_K"], params)
            values.extend((l1_nrtl - l1_tf, l2_nrtl - l2_tf))
            endpoints = sample["tf_endpoints"]
            if endpoints is not None:
                xa, xb = endpoints
                l1a, l2a = _nrtl_ln_gamma(np.array([xa]), sample["temperature_K"], params)
                l1b, l2b = _nrtl_ln_gamma(np.array([xb]), sample["temperature_K"], params)
                # The factor prioritises retaining the LLE boundary while the
                # gamma-grid residual preserves the complete TF free-energy shape.
                values.extend((8.0 * np.array([np.log(xa) + l1a[0] - np.log(xb) - l1b[0]]),
                               8.0 * np.array([np.log(1.0 - xa) + l2a[0] - np.log(1.0 - xb) - l2b[0]])))
        return np.concatenate(values)

    starts = (
        [3000.0, 3000.0, 0.0, 0.0, 0.30],
        [6000.0, 3000.0, 0.0, 0.0, 0.35],
        [1000.0, 6000.0, 0.0, 0.0, 0.25],
        [5000.0, 5000.0, 4.0, -4.0, 0.40],
    )
    results = [
        least_squares(
            residual,
            start,
            bounds=([-30000.0, -30000.0, -100.0, -100.0, 0.05], [30000.0, 30000.0, 100.0, 100.0, 0.90]),
            max_nfev=100_000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        for start in starts
    ]
    return min(results, key=lambda item: float(np.mean(item.fun**2))).x


def _write_nrtl_bips(flowsheet: Any, package: Any, params: np.ndarray) -> dict[str, float]:
    """Write both directional temperature-dependent BIPs to DWSIM's NRTL package."""
    from System import Activator

    e12, e21, b12, b21, alpha = (float(v) for v in params)
    # Convert E(T)=Eref+B*(T-Tref) to DWSIM's A+B*T convention.
    a12, a21 = e12 - b12 * T_REF, e21 - b21 * T_REF
    selected = flowsheet.SelectedCompounds
    compounds = {str(key): selected[key] for key in list(selected.Keys)}
    butanol_key = next(key for key in compounds if "butanol" in key.casefold())
    water_key = next(key for key in compounds if "water" in key.casefold())
    model = package.GetType().GetProperty("m_uni").GetValue(package, None)
    interactions = model.GetType().GetProperty("InteractionParameters").GetValue(model, None)
    inner_type = interactions.GetType().GetGenericArguments()[1]
    data_type = model.GetType().Assembly.GetType("DWSIM.Thermodynamics.PropertyPackages.Auxiliary.NRTL_IPData")
    if data_type is None:
        raise RuntimeError("DWSIM NRTL_IPData type was not found")

    for first, second, direct_a, reverse_a, direct_b, reverse_b in (
        (butanol_key, water_key, a12, a21, b12, b21),
        (water_key, butanol_key, a21, a12, b21, b12),
    ):
        if not interactions.ContainsKey(first):
            interactions.Add(first, Activator.CreateInstance(inner_type))
        inner = interactions[first]
        if not inner.ContainsKey(second):
            inner.Add(second, Activator.CreateInstance(data_type))
        data = inner[second]
        data.ID1 = compounds[first]
        data.ID2 = compounds[second]
        data.A12, data.A21 = direct_a, reverse_a
        data.B12, data.B21 = direct_b, reverse_b
        data.C12, data.C21 = 0.0, 0.0
        data.alpha12 = alpha
        data.comment = "Temperature-dependent NRTL regression from ThermoFormer binary RK gE/RT"

    package.GetType().GetProperty("AutoEstimateMissingNRTLUNIQUACParameters").SetValue(package, False, None)
    try:
        package.GetType().GetProperty("AreModelParametersDirty").SetValue(package, True, None)
    except Exception:
        pass
    configure = package.GetType().GetMethod("ConfigParameters")
    if configure is not None:
        configure.Invoke(package, None)
    return {"A12_cal_per_mol": a12, "A21_cal_per_mol": a21, "B12_cal_per_mol_K": b12,
            "B21_cal_per_mol_K": b21, "C12_cal_per_mol_K2": 0.0, "C21_cal_per_mol_K2": 0.0,
            "alpha12": alpha}


def _build_and_solve(params: np.ndarray, temperature_k: float, destination: Path) -> dict[str, Any]:
    """Create and solve a DWSIM LLE Vessel with a feed inside TF's gap."""
    from thermo_engine.dwsim_export import _add_property_package, _automation_factory, _composition_argument, _save_flowsheet_via_temp, _simulation_object

    factory, object_type = _automation_factory()
    automation = factory()
    fs = automation.CreateFlowsheet()
    for component in COMPONENTS:
        fs.AddCompound(component)
    _add_property_package(fs, "NRTL")
    package = list(fs.PropertyPackages.Values)[0]
    bip = _write_nrtl_bips(fs, package, params)
    # Verify the exact DWSIM implementation at a composition in the predicted
    # two-liquid window before accepting the Vessel result.
    from System import Activator, Array, Double, Enum, String
    from DWSIM.Interfaces.Enums import FlashSetting
    selected = fs.SelectedCompounds
    dwsim_names = [next(str(key) for key in list(selected.Keys) if "butanol" in str(key).casefold()),
                   next(str(key) for key in list(selected.Keys) if "water" in str(key).casefold())]
    nrtl_model = package.GetType().GetProperty("m_uni").GetValue(package, None)
    gamma_readback = [float(value) for value in nrtl_model.GAMMA_MR(
        temperature_k, Array[Double]([0.30, 0.70]), Array[String](dwsim_names)
    )]
    # UniversalFlash defaults to a VLE-oriented path.  Install DWSIM's explicit
    # immiscible-LLE kernel rather than accepting the trivial one-liquid root.
    flash_type = package.GetType().Assembly.GetType(
        "DWSIM.Thermodynamics.PropertyPackages.Auxiliary.FlashAlgorithms.NestedLoops3PV3"
    )
    if flash_type is None:
        raise RuntimeError("DWSIM's NestedLoops3PV3 algorithm is unavailable")
    flash_algorithm = Activator.CreateInstance(flash_type)
    flash_algorithm.StabSearchSeverity = 3
    package.GetType().GetProperty("FlashAlgorithm").SetValue(package, flash_algorithm, None)
    settings_property = package.GetType().GetProperty("FlashSettings")
    settings = settings_property.GetValue(package, None)
    # Do not enable ``ImmiscibleWaterOption``: that heuristic forces pure
    # water/organic outlets and would override the fitted mutual solubilities.
    settings[FlashSetting.ImmiscibleWaterOption] = "False"
    settings[FlashSetting.ThreePhaseFlashStabTestSeverity] = "3"
    settings[FlashSetting.UsePhaseIdentificationAlgorithm] = "True"
    settings_property.SetValue(package, settings, None)
    feed_go = fs.AddObject(object_type.MaterialStream, 0, 0, "Feed")
    vessel_go = fs.AddObject(object_type.Vessel, 350, 0, "Vessel_LLE")
    vapor_go = fs.AddObject(object_type.MaterialStream, 750, -140, "Vapor")
    light_go = fs.AddObject(object_type.MaterialStream, 750, -70, "Light_Liquid")
    heavy_go = fs.AddObject(object_type.MaterialStream, 750, 70, "Heavy_Liquid")
    feed = _simulation_object(feed_go)
    feed.SetTemperature(temperature_k)
    feed.SetPressure(P_PA)
    feed.SetMolarFlow(1.0)
    feed.SetOverallComposition(_composition_argument([0.30, 0.70]))
    package.GetType().GetProperty("CurrentMaterialStream").SetValue(package, feed, None)
    direct_flash = flash_algorithm.Flash_PT(
        Array[Double]([0.30, 0.70]), P_PA, temperature_k, package, False, None
    )
    direct_flash_debug: list[str] = []
    for value in direct_flash:
        try:
            direct_flash_debug.append("[" + ", ".join(f"{float(v):.8g}" for v in value) + "]")
        except TypeError:
            direct_flash_debug.append(repr(value))
    vessel = _simulation_object(vessel_go)
    for field, value in (("FlashTemperature", temperature_k), ("FlashPressure", P_PA), ("OverrideT", False), ("OverrideP", False)):
        try:
            setattr(vessel, field, value)
        except Exception:
            pass
    vessel.PreferredFlashAlgorithmTag = "Nested_Loops_VLLE"
    for source, target, source_port, target_port in (
        (feed_go, vessel_go, 0, 0), (vessel_go, vapor_go, 0, 0),
        (vessel_go, light_go, 1, 0), (vessel_go, heavy_go, 2, 0),
    ):
        fs.ConnectObjects(source.GraphicObject, target.GraphicObject, source_port, target_port)
    errors = automation.CalculateFlowsheet4(fs)

    def stream_state(handle: Any) -> dict[str, Any]:
        stream = _simulation_object(handle)
        kind = stream.GetType()
        return {"flow_mol_s": float(kind.GetMethod("GetMolarFlow").Invoke(stream, None)),
                "composition": [float(v) for v in kind.GetMethod("GetOverallComposition").Invoke(stream, None)]}

    light, heavy = stream_state(light_go), stream_state(heavy_go)
    _save_flowsheet_via_temp(automation, fs, destination)
    error_text = [] if not errors or not errors.Count else [str(errors[index]) for index in range(errors.Count)]
    composition_gap = float(np.abs(np.asarray(light["composition"]) - np.asarray(heavy["composition"])).sum())
    physically_distinct = light["flow_mol_s"] > 1e-10 and heavy["flow_mol_s"] > 1e-10 and composition_gap > 0.01
    return {"file": destination.name, "temperature_K": temperature_k, "feed": [0.30, 0.70], "bips": bip,
            "dwsim_gamma_readback_at_feed": gamma_readback,
            "direct_tp_flash_raw": direct_flash_debug,
            "flash_algorithm": "NestedLoops3PV3 / VLLE with phase-stability switches",
            "dwsim": {"light": light, "heavy": heavy, "composition_L1_distance": composition_gap,
                      "separated": physically_distinct, "errors": error_text}}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model, features = _load_tf()
    samples: list[dict[str, Any]] = []
    for temperature_k in T_GRID:
        c0, c1 = _tf_coefficients(model, features, float(temperature_k))
        endpoints = _tf_binodal(c0, c1)
        samples.append({"temperature_K": float(temperature_k), "c0": c0, "c1": c1, "x": X_GRID,
                        "tf_ln_gamma": _tf_ln_gamma(X_GRID, c0, c1), "tf_endpoints": endpoints})
    params = _fit(samples)

    validation: list[dict[str, Any]] = []
    for sample in samples:
        nrtl_pair = _nrtl_binodal(params, sample["temperature_K"])
        l1_tf, l2_tf = sample["tf_ln_gamma"]
        l1_nrtl, l2_nrtl = _nrtl_ln_gamma(sample["x"], sample["temperature_K"], params)
        validation.append({"temperature_K": sample["temperature_K"], "tf_endpoints": sample["tf_endpoints"],
                           "nrtl_endpoints": nrtl_pair,
                           "ln_gamma_rmse": float(np.sqrt(np.mean(np.r_[(l1_nrtl-l1_tf)**2, (l2_nrtl-l2_tf)**2])) )})

    target_temperature = 298.15
    flowsheet = OUTDIR / "water_butanol_tf_temperature_dependent_nrtl_298p15K.dwxmz"
    dwsim = _build_and_solve(params, target_temperature, flowsheet)
    expected_ln_gamma = _nrtl_ln_gamma(np.array([0.30]), target_temperature, params)
    expected_gamma = [float(np.exp(expected_ln_gamma[0][0])), float(np.exp(expected_ln_gamma[1][0]))]
    dwsim["expected_nrtl_gamma_at_feed"] = expected_gamma
    e12, e21, slope12, slope21, alpha = (float(v) for v in params)
    report = {
        "status": "generated" if dwsim["dwsim"]["separated"] else "generated_flash_validation_failed",
        "method": "ThermoFormer binary RK gE/RT -> temperature-dependent NRTL -> DWSIM Vessel LLE flash",
        "scope": {"components": COMPONENTS, "pressure_kPa": P_KPA, "sampled_temperature_K": T_GRID.tolist(),
                  "do_not_extrapolate_outside_sampled_temperature_range": True},
        "fit_coordinates": {"E12_at_Tref_cal_per_mol": e12, "E21_at_Tref_cal_per_mol": e21,
                            "Tref_K": T_REF, "dE12_dT_cal_per_mol_K": slope12,
                            "dE21_dT_cal_per_mol_K": slope21, "alpha12": alpha},
        "validation": validation,
        "dwsim_export": dwsim,
        "checkpoint": str(CHECKPOINT.relative_to(TF_ROOT)),
        "generated_at": datetime.now().isoformat(),
    }
    report_path = OUTDIR / "water_butanol_tf_temperature_dependent_nrtl_298p15K.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"flowsheet": str(flowsheet), "report": str(report_path), "separated": dwsim["dwsim"]["separated"],
                      "dwsim_gamma": dwsim["dwsim_gamma_readback_at_feed"], "expected_gamma": expected_gamma}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
