"""Deterministic short-cut design for extractive distillation columns.

This module implements the Fenske / Underwood / Gilliland short-cut method for
the design of an extractive distillation column that recovers high-purity
ethanol overhead from an ethanol/water feed using a high-boiling entrainer
added near the top.  All numbers are produced deterministically:

* Ternary activity coefficients come from the temperature-dependent UNIFAC
  prediction (``thermo.UNIFAC_gammas``), so no binary-interaction parameter
  table is fabricated.
* Vapor pressures for the two key species (ethanol, water) come from NIST
  Antoine constants already curated in :mod:`thermo_engine.activity_coeff_utils`.
* The ethanol/water relative volatility (and its entrainer-enhanced value) is
  derived from those deterministic gamma/vapor-pressure values.
* Minimum stages (Fenske), minimum reflux (Underwood) and the operating
  reflux/stage trade-off (Gilliland-Eduljee) follow the standard constant
  relative-volatility short-cut equations.

The module never fabricates numbers and never invents binary parameters; a
candidate entrainer without an evaluable relative volatility is simply excluded
from ``recommend_extraction_entrainer``.  LLM/rule heuristics are never used to
produce these equilibrium-derived values.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from math import log

import numpy as np
from scipy.optimize import brentq

from schemas.column_design import (
    EntrainerCandidate,
    EntrainerRecommendation,
    ExtractiveColumnDesign,
    ExtractiveColumnSpec,
)
from thermo_engine.activity_coeff_utils import psat_Pa

#: Components whose saturation pressure comes from the curated NIST Antoine
#: branch (returns Pa).  Other components fall back to ``thermo`` directly.
_NIST_RELIABLE = frozenset({"ethanol", "water", "methanol", "benzene", "toluene", "acetone"})

#: SMILES for the components ThermoFormer needs for molecular-feature encoding.
#: Only the species that can appear in the extractive design are listed; any name
#: without a SMILES entry below cannot use the model-backed volatility path.
_COMPONENT_SMILES: dict[str, str] = {
    "ethanol": "CCO",
    "water": "O",
    "methanol": "CO",
    # NOTE: these must be RDKit CANONICAL SMILES.  The ThermoFormer feature cache
    # is keyed by the canonical string, so a non-canonical spelling (e.g. the
    # equivalent "CC(O)C") misses the cache and forces an on-the-fly Uni-Mol
    # conformer generation, which fails for water in some environments.
    "isopropanol": "CC(C)O",
    "2-propanol": "CC(C)O",
    "isopropyl alcohol": "CC(C)O",
    "propan-2-ol": "CC(C)O",
    "acetic acid": "CC(=O)O",
    "methyl acetate": "CC(=O)OC",
    "methylacetate": "CC(=O)OC",
    "acetone": "CC(=O)C",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "ethylene glycol": "OCCO",
    "eg": "OCCO",
    "ethyleneglycol": "OCCO",
    "glycerol": "OCC(O)CO",
    "glycerine": "OCC(O)CO",
    "glycerin": "OCC(O)CO",
    # Ester / sulfoxide species used for the ethyl-acetate & propyl-acetate
    # extractive-distillation cases (case three in the campaign docs).
    "ethyl acetate": "CC(=O)OCC",
    "propyl acetate": "CC(=O)OCCC",
    "n-propyl acetate": "CC(=O)OCCC",
    "propan-1-yl acetate": "CC(=O)OCCC",
    "dimethyl sulfoxide": "CS(=O)C",
    "dmso": "CS(=O)C",
    # Straight-chain alkanes for the n-heptane / n-nonane reference system
    # (NIST ThermoML 10.1016/j.fluid.2013.05.016).  Both strings are already
    # RDKit-canonical, so the ThermoFormer feature cache is hit directly.
    "heptane": "CCCCCCC",
    "n-heptane": "CCCCCCC",
    "nonane": "CCCCCCCCC",
    "n-nonane": "CCCCCCCCC",
    # Alcohols / aromatics for the 1-butanol + water + toluene system.
    "1-butanol": "CCCCO",
    "n-butanol": "CCCCO",
    "butan-1-ol": "CCCCO",
    "1-butanol (n-butanol)": "CCCCO",
    "tetrahydrofuran": "C1CCOC1",
    "thf": "C1CCOC1",
    "acetonitrile": "CC#N",
}

#: How ``relative_volatility_eivw`` derives activity coefficients: ``unifac``
#: (temperature-dependent UNIFAC, the classic deterministic path) or
#: ``thermoformer`` (given x,T a bubble isothermal VLE from the ThermoFormer
#: backend is solved and the pair effective relative volatility is taken from
#: the predicted vapor composition y).  Defaults to UNIFAC so the design remains
#: deterministic and dependency-free unless the user opts into the model.
_DEFAULT_ALPHA_SOURCE = "unifac"


def _alpha_source() -> str:
    return os.getenv("COLUMN_DESIGN_ALPHA_SOURCE", _DEFAULT_ALPHA_SOURCE).casefold().strip()


#: Lazily-created, shared ThermoFormer backend for the column design.  Loading
#: the checkpoint and the Uni-Mol feature encoder is expensive, so the same
#: backend instance (and its in-memory model) is reused across the multiple
#: relative-volatility evaluations of a single design instead of being rebuilt
#: (and re-downloaded) each time.
_thermoformer_backend: object | None = None


def _get_thermoformer_backend() -> object:
    global _thermoformer_backend
    if _thermoformer_backend is None:
        from thermo_engine.thermoformer_backend import ThermoFormerBackend

        try:
            _thermoformer_backend = ThermoFormerBackend()
        except Exception as exc:  # noqa: BLE001 - configuration missing: fail fast
            raise ValueError(f"ThermoFormer backend is not available: {type(exc).__name__}: {exc}") from exc
    return _thermoformer_backend


def _relative_volatility_from_model(
    components: Sequence[str], xs: Sequence[float], T_K: float
) -> float:
    """Relative volatility for the (components[0], components[1]) pair via ThermoFormer.

    Given the liquid composition ``xs`` (which may include the entrainer as a
    third component) and temperature ``T_K``, run a single-point *isothermal
    bubble* prediction with the ThermoFormer backend at the *exact* liquid
    composition, take the predicted vapor composition ``y``, and use the
    K-value ratio definition

        alpha_ij = (y_i / x_i) / (y_j / x_j)

    for components[0] (ethanol) vs components[1] (water).  This is the
    "equilibrium relative volatility" convention that matches the paper's
    (x, T) -> (y, P) bubble-point task.

    All provided components are encoded (so an extraner in the composition is a
    real species rather than a silently dropped degree of freedom) and the
    ``bubble_point`` single-point path is used so the prediction is evaluated
    exactly at ``xs``, never at a coarse composition sweep.
    """
    from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
    from thermo_engine.errors import ThermoEquiError

    i, j = 0, 1
    names = [components[k].casefold() for k in range(len(components))]
    missing = [n for n in names if _COMPONENT_SMILES.get(n) is None]
    if missing:
        raise ValueError(f"no SMILES for relative-volatility components: {missing}")

    species = [
        ComponentIdentity(
            component_id=names[k],
            name=components[k],
            smiles=_COMPONENT_SMILES[names[k]],
            aliases=[],
        )
        for k in range(len(components))
    ]
    request = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=species,
        conditions=ThermodynamicConditions(temperature_K=T_K, liquid_composition=list(xs)),
        model_name="ThermoFormer",
    )
    backend = _get_thermoformer_backend()
    try:
        result = backend.bubble_point(request)
    except ThermoEquiError as exc:
        raise ValueError(f"ThermoFormer bubble prediction failed: {exc}") from exc
    points = result.points
    if not points:
        raise ValueError("ThermoFormer bubble-point VLE returned no points")
    point = points[0]
    y = np.asarray(list(point.vapor_composition), dtype=float)
    x = np.asarray(list(point.liquid_composition), dtype=float)
    if y.size < 2 or x.size < 2:
        raise ValueError("ThermoFormer bubble-point VLE returned too few composition values")
    ki = y[i] / x[i]
    kj = y[j] / x[j]
    if kj <= 0.0:
        raise ValueError("ThermoFormer relative volatility denominator is non-positive")
    return float(ki / kj)


def relative_volatility_eivw(
    components: Sequence[str], xs: Sequence[float], T_K: float,
    alpha_source: str | None = None,
) -> float:
    """Ethanol/water relative volatility from a configurable activity source.

    ``components[0]`` must be ethanol and ``components[1]`` water; any further
    entries are treated as inert present in the liquid phase.  Returns a
    positive deterministic value.

    Source of activity coefficients:
      * ``unifac`` (default): ``alpha = gamma_E/gamma_W * P_E/P_W`` with
        temperature-dependent UNIFAC gammas and NIST Antoine vapor pressures.
      * ``thermoformer``: an isothermal bubble-point VLE is solved by the
        ThermoFormer backend at (x, T) and the pair effective relative volatility
        is ``(y_E/x_E)/(y_W/x_W)`` from the predicted vapor composition y.

    ``alpha_source`` overrides the ``COLUMN_DESIGN_ALPHA_SOURCE`` environment
    switch for a single call (so a request can opt into ``thermoformer`` without
    mutating process-global state).  When ``None`` the environment default
    applies.
    """
    if len(components) < 2:
        raise ValueError("relative volatility requires at least ethanol and water")
    if components[0].casefold() not in {"ethanol", "alcohol", "c2h5oh"}:
        raise ValueError("relative_volatility_eivw expects ethanol as components[0]")
    if components[1].casefold() not in {"water", "h2o"}:
        raise ValueError("relative_volatility_eivw expects water as components[1]")

    source = (alpha_source or _alpha_source()).casefold().strip()
    if source == "thermoformer":
        return _relative_volatility_from_model(components, list(xs), T_K)

    gamma = _unifac_gammas(components, list(xs), T_K)
    pe = _vap_pressure("ethanol", T_K)
    pw = _vap_pressure("water", T_K)
    if gamma[1] * pw <= 0.0:
        raise ValueError("water saturation pressure is non-positive")
    return float((gamma[0] * pe) / (gamma[1] * pw))


def _vap_pressure(name: str, T_K: float) -> float:
    """Saturation pressure in Pa with a consistent, deterministic source.

    Ethanol/water (the two key species) use the curated NIST Antoine branch in
    :func:`thermo_engine.activity_coeff_utils.psat_Pa`.  Any other component
    (e.g. the entrainer) uses ``thermo.Chemical.VaporPressure`` which returns
    **Pa** directly (it is not multiplied by 1e5, so high boiling glycols are
    correct near their normal boiling points).
    """
    lower = name.casefold()
    if lower in _NIST_RELIABLE:
        return psat_Pa(name, T_K)
    from thermo import Chemical

    return float(Chemical(name).VaporPressure(T_K))

#: Backend marker reported by produced designs.
_BACKEND_VERSION = "column-design-0.1"

#: Default candidate entrainers, ordered by common industrial preference.
#: ``UNIFAC_gammas`` must be able to resolve every candidate name.
DEFAULT_ENTRAINER_CANDIDATES: tuple[str, ...] = ("ethylene glycol", "glycerol")


# --------------------------------------------------------------------------- #
# Activity-coefficient / relative-volatility building blocks
# --------------------------------------------------------------------------- #
def _chemgroups(components: Sequence[str]) -> list[dict[int, int]]:
    """Resolve UNIFAC subgroup counts for each component name."""
    from thermo import Chemical

    groups: list[dict[int, int]] = []
    for name in components:
        chem = Chemical(name)
        counts = getattr(chem, "UNIFAC_groups", None)
        if not counts:
            raise ValueError(f"UNIFAC cannot assign subgroups for component {name!r}.")
        groups.append(dict(counts))
    return groups


def activity_coefficients(components: Sequence[str], xs: Sequence[float], T_K: float) -> list[float]:
    """Deterministic temperature-dependent UNIFAC activity coefficients."""
    if len(components) != len(xs):
        raise ValueError("components and composition must have the same length")
    if abs(sum(float(x) for x in xs) - 1.0) > 1e-6:
        raise ValueError("composition must sum to one within 1e-6")
    if any(x < 0 for x in xs):
        raise ValueError("composition must be non-negative")
    gamma = _unifac_gammas(components, xs, T_K)
    return [float(g) for g in gamma]


def _unifac_gammas(components: Sequence[str], xs: Sequence[float], T_K: float) -> np.ndarray:
    from thermo import UNIFAC_gammas

    return UNIFAC_gammas(T_K, list(xs), _chemgroups(components))


def _bubble_temperature(components: Sequence[str], xs: Sequence[float], P_kPa: float) -> float:
    """Bubble temperature (K) of a liquid composition at a pressure."""
    P_pa = P_kPa * 1000.0

    def imbalance(T: float) -> float:
        gamma = _unifac_gammas(components, list(xs), T)
        total = 0.0
        for i, name in enumerate(components):
            total += xs[i] * gamma[i] * _vap_pressure(name, T)
        return float(total / P_pa - 1.0)

    low, high = 240.0, 900.0
    # Widen bounds until the objective brackets a root.
    for lo, hi in ((240.0, 500.0), (240.0, 700.0), (240.0, 900.0)):
        try:
            if imbalance(lo) * imbalance(hi) < 0.0:
                low, high = lo, hi
                break
        except Exception:
            continue
    root, info = brentq(imbalance, low, high, xtol=1e-7, full_output=True, maxiter=200)
    del info
    return float(root)


def _bubble_temperature_from_model(
    components: Sequence[str], xs: Sequence[float], P_kPa: float
) -> float:
    """Isobaric bubble temperature via the ThermoFormer backend.

    Builds an isobaric single-point ``bubble_point`` request (liquid composition
    ``xs`` and pressure ``P`` given) so the ThermoFormer ``predict_bubble_isobaric``
    path predicts the bubble temperature ``T`` and vapor composition ``y``.  This
    matches the manuscript's Isobaric ``T--x--y`` direction.
    """
    from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
    from thermo_engine.errors import ThermoEquiError

    names = [components[k].casefold() for k in range(len(components))]
    missing = [n for n in names if _COMPONENT_SMILES.get(n) is None]
    if missing:
        raise ValueError(f"no SMILES for bubble-temperature components: {missing}")

    species = [
        ComponentIdentity(
            component_id=names[k],
            name=components[k],
            smiles=_COMPONENT_SMILES[names[k]],
            aliases=[],
        )
        for k in range(len(components))
    ]
    request = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=species,
        conditions=ThermodynamicConditions(
            pressure_kPa=P_kPa, liquid_composition=list(xs)
        ),
        model_name="ThermoFormer",
    )
    backend = _get_thermoformer_backend()
    try:
        result = backend.bubble_point(request)
    except ThermoEquiError as exc:
        raise ValueError(f"ThermoFormer bubble prediction failed: {exc}") from exc
    points = result.points
    if not points or points[0].temperature_K is None:
        raise ValueError("ThermoFormer bubble-point VLE returned no temperature")
    return float(points[0].temperature_K)


def bubble_temperature(
    components: Sequence[str],
    xs: Sequence[float],
    P_kPa: float,
    alpha_source: str | None = None,
) -> float:
    """Isobaric bubble temperature from a configurable source.

    ``unifac`` (default): solve the bubble equation with temperature-dependent
    UNIFAC gammas and NIST Antoine vapor pressures.  ``thermoformer``: an isobaric
    single-point bubble prediction returns ``T`` directly from the model.  The
    source override follows :func:`relative_volatility_eivw` for a single call.
    """
    source = _resolved_alpha_source(alpha_source)
    if source == "thermoformer":
        return _bubble_temperature_from_model(components, list(xs), P_kPa)
    return _bubble_temperature(components, list(xs), P_kPa)


def unifac_bubble_pressure(
    components: Sequence[str], xs: Sequence[float], T_K: float
) -> float:
    """Isothermal UNIFAC bubble pressure ``P = sum_i x_i * gamma_i * Psat_i``.

    Returns pressure in **kPa**.  Uses the same deterministic vapor-pressure
    source as :func:`bubble_temperature` (``_vap_pressure``) so the isothermal
    and isobaric reference results are mutually consistent.  This is the
    modified Raoult's-law bubble pressure with temperature-dependent UNIFAC
    activity coefficients.
    """
    gamma = _unifac_gammas(components, list(xs), T_K)
    P_pa = sum(
        xs[i] * gamma[i] * _vap_pressure(components[i], T_K)
        for i in range(len(components))
    )
    return float(P_pa / 1000.0)


# --------------------------------------------------------------------------- #
# Public model API
# --------------------------------------------------------------------------- #
def recommend_extraction_entrainer(
    spec: ExtractiveColumnSpec,
    candidates: Sequence[str] = DEFAULT_ENTRAINER_CANDIDATES,
    alpha_source: str | None = None,
) -> EntrainerRecommendation:
    """Rank entrainer candidates by their enhancement of ethanol/water volatility.

    Each candidate is evaluated at the same representative entrainer-rich
    liquid composition (the extractive section).  The ranking key is the
    candidate's extracted-section relative volatility; a candidate whose
    UNIFAC subgroups cannot be resolved is skipped rather than guessed.

    ``alpha_source`` follows :func:`relative_volatility_eivw` -- pass
    ``"thermoformer"`` to evaluate selectivity from the neural model instead of
    UNIFAC for this single call.
    """
    if len(spec.feed_components) != 2:
        raise ValueError("entrainer recommendation requires a binary ethanol/water feed")
    # Reference: binary feed composition normalised to ethanol/water only.
    f_e, f_w = float(spec.feed_composition[0]), float(spec.feed_composition[1])
    base_composition = [f_e, f_w]
    base_T = spec.feed_temperature_K

    binary = ["ethanol", "water"]
    alpha_base = relative_volatility_eivw(binary, base_composition, base_T, alpha_source)

    entries: list[EntrainerCandidate] = []
    for candidate in set(candidates):
        comps = ["ethanol", "water", candidate]
        # Representative entrainer-rich composition in the extractive section.
        xs = [0.1, 0.1, 0.8]
        try:
            alpha = relative_volatility_eivw(comps, xs, base_T, alpha_source)
        except (ValueError, Exception):
            continue  # unresolvable candidate -> skip deterministically
        source_label = "ThermoFormer-predicted" if _resolved_alpha_source(alpha_source) == "thermoformer" else "UNIFAC-predicted"
        entries.append(
            EntrainerCandidate(
                name=candidate,
                selectivity=alpha / alpha_base,
                relative_volatility=alpha,
                recovered_at_bottom=True,
                note=(
                    f"{source_label} ethanol/water relative volatility = {alpha:.3f} "
                    f"(baseline binary = {alpha_base:.3f})"
                ),
            )
        )

    if not entries:
        raise ValueError("No evaluable entrainer candidate found for the current feed.")
    entries.sort(key=lambda c: c.relative_volatility, reverse=True)
    return EntrainerRecommendation(recommended=entries[0].name, candidates=entries)


def _resolved_alpha_source(alpha_source: str | None) -> str:
    """Resolve the effective activity source for one call (override or env)."""
    return (alpha_source or _alpha_source()).casefold().strip()


def design_extractive_distillation_column(
    spec: ExtractiveColumnSpec,
    alpha_source: str | None = None,
) -> ExtractiveColumnDesign:
    """Run the deterministic Fenske/Underwood/Gilliland short-cut design.

    ``alpha_source`` selects how the ethanol/water relative volatility is
    derived for this run: the ``unifac`` default, or ``thermoformer`` to take
    the vapor composition from a ThermoFormer bubble-point prediction.  It
    overrides the ``COLUMN_DESIGN_ALPHA_SOURCE`` environment switch for a single
    call.
    """
    if len(spec.feed_components) != 2:
        raise ValueError("extractive distillation design requires a binary ethanol/water feed")

    feed_components = ["ethanol", "water", spec.entrainer]
    x_fe, x_fw = float(spec.feed_composition[0]), float(spec.feed_composition[1])
    P = spec.operating_pressure_kPa
    T_feed = spec.feed_temperature_K
    source = _resolved_alpha_source(alpha_source)

    # --- Relative volatility ------------------------------------------------- #
    # Base (binary, no entrainer) at the feed composition.
    alpha_base = relative_volatility_eivw(
        ["ethanol", "water"], [x_fe, x_fw], T_feed, source
    )
    # Extractive section (entrainer-rich) at the feed temperature.
    xs_ext = [0.10, 0.10, 0.80]
    alpha_ext = relative_volatility_eivw(feed_components, xs_ext, T_feed, source)
    selectivity = alpha_ext / alpha_base
    # Effective average alpha used by the short-cut correlations.
    alpha_avg = (alpha_base * alpha_ext) ** 0.5

    # --- Product splits ------------------------------------------------------ #
    purity = spec.distillate_purity_mole_fraction
    recovery = spec.ethanol_recovery

    # Component molar flows in the feed.
    n_e = spec.feed_flow_mol_s * x_fe
    n_w = spec.feed_flow_mol_s * x_fw
    # Overhead: ethanol recovered, water determined by purity (as the only
    # other overhead component at negligible entrainer loss).
    d_e = recovery * n_e
    d_w = (1.0 - purity) / purity * d_e
    D = d_e + d_w
    x_de = d_e / D
    x_dw = d_w / D
    # Bottoms: everything else.
    b_e = n_e - d_e
    b_w = n_w - d_w
    b_s = spec.entrainer_ratio * spec.feed_flow_mol_s
    B = b_e + b_w + b_s
    x_be = b_e / B
    x_bw = b_w / B
    x_bs = b_s / B

    # --- Fenske minimum stages ----------------------------------------------- #
    fenske_ratio = (x_de / x_dw) / (x_be / x_bw)
    n_min = log(fenske_ratio) / log(alpha_avg) if alpha_avg != 1.0 else 0.0
    n_min = max(n_min, 0.5)

    # --- Underwood minimum reflux (binary constant-alpha form) --------------- #
    r_min = (x_de / x_fe - alpha_avg * (1.0 - x_de) / (1.0 - x_fe)) / (alpha_avg - 1.0)
    r_min = max(r_min, 0.05)

    # --- Gilliland reflux/stage trade-off ------------------------------------ #
    reflux_factor = 1.4  # documented operating R = 1.4 x R_min
    R = reflux_factor * r_min
    x_gill = (R - r_min) / (R + 1.0)
    y_gill = 0.75 * (1.0 - x_gill**0.5668)
    stages = int(np.ceil((n_min + y_gill) / (1.0 - y_gill))) if y_gill < 1.0 else int(np.ceil(n_min + 5))
    if stages < 3:
        stages = 3

    # --- Feed / entrainer stages --------------------------------------------- #
    # Feed positioned ~45% down from the top; entrainer just below stage 1.
    feed_stage = int(np.floor(0.45 * (stages + 1)))
    feed_stage = max(2, min(feed_stage, stages - 1))
    entrainer_stage = 2
    if entrainer_stage >= feed_stage:
        entrainer_stage = max(1, feed_stage - 1)

    # --- Temperatures -------------------------------------------------------- #
    # Overhead bubble temperature at the distillate composition.
    t_cond = bubble_temperature(["ethanol", "water"], [x_de, x_dw], P, source)
    # Bottom bubble temperature at the full bottoms composition.
    t_reb = bubble_temperature(feed_components, [x_be, x_bw, x_bs], P, source)

    warnings: list[str] = []
    use_thermoformer = source == "thermoformer"
    assumptions = [
        "constant relative volatility model (Fenske/Underwood/Gilliland-Eduljee)",
        "operating reflux ratio = 1.4 x minimum reflux ratio",
        "entrainer is a high-boiling component recovered entirely in the bottoms",
        "ethylated overhead has negligible entrainer loss",
        (
            "ethanol/water relative volatility from ThermoFormer bubble-point (ML) predictions"
            if use_thermoformer
            else "activity coefficients from UNIFAC temperature-dependent prediction"
        ),
        (
            "distillate/reboiler bubble temperatures from ThermoFormer isobaric predictions"
            if use_thermoformer
            else "vapor pressures for ethanol/water from NIST Antoine constants"
        ),
        "total condenser",
    ]
    if use_thermoformer:
        warnings.append(
            "ThermoFormer is a predictive ML backend; results are approximate and have not "
            "been validated against experimental data for this system. "
            "Benchmark closure and applicability review are pending."
        )
    if alpha_avg <= 1.15:
        warnings.append(
            f"Enhancement is weak (relative volatility {alpha_avg:.3f}); column will need many stages."
        )

    return ExtractiveColumnDesign(
        spec=spec,
        theoretical_stages=stages,
        minimum_stages=round(n_min, 3),
        reflux_ratio=round(R, 3),
        minimum_reflux_ratio=round(r_min, 3),
        feed_stage=feed_stage,
        entrainer_stage=entrainer_stage,
        relative_volatility=round(alpha_avg, 3),
        base_relative_volatility=round(alpha_base, 3),
        selectivity=round(selectivity, 3),
        condenser_temperature_K=round(t_cond, 2),
        reboiler_temperature_K=round(t_reb, 2),
        operating_pressure_kPa=P,
        distillate_purity_mole_fraction=round(x_de, 6),
        bottoms_composition=[x_be, x_bw, x_bs],
        distillate_flow_mol_s=round(D, 6),
        bottoms_flow_mol_s=round(B, 6),
        warnings=warnings,
        assumptions=assumptions,
        needs_validation=False,
        backend_version=_BACKEND_VERSION + (f"+{source}" if use_thermoformer else ""),
    )


@dataclass
class BinaryDistillationDesign:
    """Result of a plain (non-extractive) binary distillation short-cut design.

    Produced by :func:`design_binary_distillation_column` for an arbitrary binary
    feed (e.g. methanol/water) using the Fenske / Underwood / Gilliland method.
    All numbers come from the deterministic engine; ``alpha_source`` selects
    whether the relative volatility uses UNIFAC or ThermoFormer bubble prediction.
    """

    components: list[str]
    feed_composition: list[float]
    feed_flow_mol_s: float
    operating_pressure_kPa: float
    distillate_purity_mole_fraction: float
    relative_volatility: float
    minimum_stages: float
    minimum_reflux_ratio: float
    reflux_ratio: float
    theoretical_stages: int
    feed_stage: int
    condenser_temperature_K: float
    reboiler_temperature_K: float
    distillate_flow_mol_s: float
    bottoms_flow_mol_s: float
    alpha_source: str
    assumptions: list[str]


def _binary_relative_volatility(
    components: Sequence[str],
    xs: Sequence[float],
    T_K: float,
    alpha_source: str | None,
) -> tuple[float, list[str]]:
    """Relative volatility of components[0] vs components[1] at (x, T).

    ``unifac`` (default): ``alpha = gamma_0*P0^sat / (gamma_1*P1^sat)``.
    ``thermoformer``: the ThermoFormer bubble-point vapor composition ``y`` is
    used as ``alpha = (y_0/x_0) / (y_1/x_1)``.  Returns (alpha, assumptions).
    """
    source = _resolved_alpha_source(alpha_source)
    if source == "thermoformer":
        from schemas.domain import (
            ComponentIdentity,
            TaskManifest,
            ThermodynamicConditions,
        )
        from thermo_engine.errors import ThermoEquiError

        names = [components[k].casefold() for k in range(len(components))]
        missing = [n for n in names if _COMPONENT_SMILES.get(n) is None]
        if missing:
            raise ValueError(f"no SMILES for relative-volatility components: {missing}")
        species = [
            ComponentIdentity(
                component_id=names[k], name=components[k],
                smiles=_COMPONENT_SMILES[names[k]], aliases=[],
            )
            for k in range(len(components))
        ]
        request = TaskManifest(
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            components=species,
            conditions=ThermodynamicConditions(
                temperature_K=float(T_K), liquid_composition=list(xs)
            ),
            model_name="ThermoFormer",
        )
        try:
            result = _get_thermoformer_backend().bubble_point(request)
        except ThermoEquiError as exc:
            raise ValueError(f"ThermoFormer bubble prediction failed: {exc}") from exc
        if not result.points:
            raise ValueError("ThermoFormer bubble-point VLE returned no points")
        point = result.points[0]
        y = np.asarray(list(point.vapor_composition), dtype=float)
        x = np.asarray(list(point.liquid_composition), dtype=float)
        if y.size < 2 or x.size < 2:
            raise ValueError("ThermoFormer bubble-point VLE returned too few compositions")
        ki = y[0] / x[0]
        kj = y[1] / x[1]
        if kj <= 0.0:
            raise ValueError("ThermoFormer relative volatility denominator is non-positive")
        return float(ki / kj), ["relative volatility from ThermoFormer bubble-point prediction"]

    gamma = _unifac_gammas([components[0], components[1]], list(xs), T_K)
    pe = _vap_pressure(components[0], T_K)
    pw = _vap_pressure(components[1], T_K)
    if gamma[1] * pw <= 0.0:
        raise ValueError("second-component vapor pressure is non-positive")
    return float((gamma[0] * pe) / (gamma[1] * pw)), [
        "activity coefficients from UNIFAC temperature-dependent prediction"
    ]


def design_binary_distillation_column(
    components: Sequence[str],
    feed_composition: Sequence[float],
    feed_flow_mol_s: float = 1.0,
    feed_temperature_K: float | None = None,
    operating_pressure_kPa: float = 101.325,
    distillate_purity_mole_fraction: float = 0.995,
    recovery: float = 0.98,
    alpha_source: str | None = None,
    relative_volatility_override: float | None = None,
    condenser_temperature_K_override: float | None = None,
    reboiler_temperature_K_override: float | None = None,
) -> BinaryDistillationDesign:
    """Plain (non-extractive) binary distillation short-cut design.

    Uses the same Fenske / Underwood / Gilliland correlation underlying
    :func:`design_extractive_distillation_column`, but for a general binary feed
    (e.g. ``["methanol", "water"]``) with no third-component extractant.
    ``components[0]`` is the overhead-concentrated (more volatile) species;
    ``components[1]`` the bottoms-concentrated species.  ``recovery`` and
    ``distillate_purity`` set the product split.

    ``alpha_source`` selects UNIFAC (default) or ThermoFormer.  ``feed_temperature_K``
    is the temperature at which the relative volatility is evaluated; when ``None``
    it is taken as the isobaric bubble temperature of the feed at the operating
    pressure.  Distillate/reboiler bubble temperatures come from
    :func:`bubble_temperature` with the same source. The three ``*_override``
    values provide a traceable integration route for a validated external VLE
    engine (for example, a DWSIM UNIQUAC bubble calculation): they are used
    verbatim when supplied and are never inferred or fitted by this module.
    """
    if len(components) != 2:
        raise ValueError("binary distillation design requires exactly two components")
    if len(feed_composition) != 2 or abs(sum(feed_composition) - 1.0) > 1e-6:
        raise ValueError("feed_composition must be two mole fractions summing to one")

    c0, c1 = components[0], components[1]
    x_f0, x_f1 = float(feed_composition[0]), float(feed_composition[1])
    P = operating_pressure_kPa
    purity = distillate_purity_mole_fraction
    source = _resolved_alpha_source(alpha_source)

    # Feed bubble temperature as the reference for the relative volatility.
    if feed_temperature_K is None:
        T_feed = bubble_temperature([c0, c1], [x_f0, x_f1], P, source)
    else:
        T_feed = float(feed_temperature_K)

    # Relative volatility at the feed composition and feed temperature.
    if relative_volatility_override is None:
        alpha, alpha_assumption = _binary_relative_volatility(
            [c0, c1], [x_f0, x_f1], T_feed, source
        )
        alpha = max(alpha, 1e-6)
    else:
        alpha = float(relative_volatility_override)
        if not np.isfinite(alpha) or alpha <= 1.0:
            raise ValueError("relative_volatility_override must be finite and greater than one")
        alpha_assumption = ["relative volatility supplied by an external VLE calculation"]

    # Product material balance.
    F = float(feed_flow_mol_s)
    n0 = F * x_f0  # component 0 (overhead) in feed
    n1 = F * x_f1  # component 1 (bottoms) in feed
    d0 = recovery * n0
    d1 = (1.0 - purity) / purity * d0
    D = d0 + d1
    b0 = n0 - d0
    b1 = n1 - d1
    B = b0 + b1
    x_D = d0 / D             # light (component 0) in distillate
    x_B_light = b0 / B       # light (component 0) in bottoms
    x_B_heavy = b1 / B       # component 1 in bottoms

    # Fenske minimum stages (light-component key).
    fenske_ratio = (x_D / (1.0 - x_D)) / (x_B_light / (1.0 - x_B_light))
    n_min = log(fenske_ratio) / log(alpha) if alpha != 1.0 else 0.0
    n_min = max(n_min, 0.5)

    # Underwood minimum reflux (binary, constant alpha).
    z_F = x_f0
    r_min = (x_D / z_F - alpha * (1.0 - x_D) / (1.0 - z_F)) / (alpha - 1.0)
    r_min = max(r_min, 0.05)

    # Gilliland reflux/stage trade-off.
    reflux_factor = 1.4
    R = reflux_factor * r_min
    x_gill = (R - r_min) / (R + 1.0)
    y_gill = 0.75 * (1.0 - x_gill**0.5668)
    stages = int(np.ceil((n_min + y_gill) / (1.0 - y_gill))) if y_gill < 1.0 else int(np.ceil(n_min + 5))
    stages = max(stages, 3)
    feed_stage = int(np.floor(0.45 * (stages + 1)))
    feed_stage = max(2, min(feed_stage, stages - 1))

    # Bubble temperatures for distillate and bottoms compositions.
    t_cond = (
        bubble_temperature([c0, c1], [x_D, 1.0 - x_D], P, source)
        if condenser_temperature_K_override is None
        else float(condenser_temperature_K_override)
    )
    t_reb = (
        bubble_temperature([c0, c1], [x_B_light, x_B_heavy], P, source)
        if reboiler_temperature_K_override is None
        else float(reboiler_temperature_K_override)
    )
    if not np.isfinite(t_cond) or not np.isfinite(t_reb) or t_cond <= 0.0 or t_reb <= 0.0:
        raise ValueError("condenser/reboiler temperature overrides must be finite and positive")

    assumptions = [
        "non-extractive binary distillation (Fenske/Underwood/Gilliland-Eduljee)",
        f"feed component ordering: {c0} overhead-concentrated, {c1} bottoms-concentrated",
        "operating reflux ratio = 1.4 x minimum reflux ratio",
        "total condenser",
        alpha_assumption[0],
    ]

    return BinaryDistillationDesign(
        components=[c0, c1],
        feed_composition=[x_f0, x_f1],
        feed_flow_mol_s=F,
        operating_pressure_kPa=P,
        distillate_purity_mole_fraction=x_D,
        relative_volatility=round(alpha, 4),
        minimum_stages=round(n_min, 3),
        minimum_reflux_ratio=round(r_min, 3),
        reflux_ratio=round(R, 3),
        theoretical_stages=stages,
        feed_stage=feed_stage,
        condenser_temperature_K=round(t_cond, 2),
        reboiler_temperature_K=round(t_reb, 2),
        distillate_flow_mol_s=round(D, 6),
        bottoms_flow_mol_s=round(B, 6),
        alpha_source=source,
        assumptions=assumptions,
    )


def relative_volatility_key(
    components: Sequence[str],
    xs: Sequence[float],
    T_K: float,
    alpha_source: str | None = None,
) -> float:
    """Relative volatility of ``components[0]`` vs ``components[1]`` at (x, T).

    Generic version of :func:`relative_volatility_eivw` that works for **any**
    light/heavy key pair (not just ethanol/water).  Extra entries in
    ``components`` (e.g. an extractive solvent) are encoded into the UNIFAC /
    ThermoFormer evaluation so the key cut does not silently drop a degree of
    freedom.

    ``unifac`` (default): ``alpha = gamma_0*P0^sat / (gamma_1*P1^sat)`` with
    temperature-dependent UNIFAC activity coefficients.
    ``thermoformer``: the ThermoFormer bubble-point vapor composition ``y`` is
    used as ``alpha = (y_0/x_0) / (y_1/x_1)``.
    """
    source = _resolved_alpha_source(alpha_source)
    if source == "thermoformer":
        from schemas.domain import (
            ComponentIdentity,
            TaskManifest,
            ThermodynamicConditions,
        )
        from thermo_engine.errors import ThermoEquiError

        names = [components[k].casefold() for k in range(len(components))]
        missing = [n for n in names if _COMPONENT_SMILES.get(n) is None]
        if missing:
            raise ValueError(f"no SMILES for relative-volatility components: {missing}")
        species = [
            ComponentIdentity(
                component_id=names[k], name=components[k],
                smiles=_COMPONENT_SMILES[names[k]], aliases=[],
            )
            for k in range(len(components))
        ]
        request = TaskManifest(
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            components=species,
            conditions=ThermodynamicConditions(
                temperature_K=float(T_K), liquid_composition=list(xs)
            ),
            model_name="ThermoFormer",
        )
        result = _get_thermoformer_backend().bubble_point(request)
        if not result.points:
            raise ValueError("ThermoFormer bubble-point VLE returned no points")
        point = result.points[0]
        y = np.asarray(list(point.vapor_composition), dtype=float)
        x = np.asarray(list(point.liquid_composition), dtype=float)
        if y.size < 2 or x.size < 2:
            raise ValueError("ThermoFormer bubble-point VLE returned too few compositions")
        ki = y[0] / x[0]
        kj = y[1] / x[1]
        if kj <= 0.0:
            raise ValueError("ThermoFormer relative volatility denominator is non-positive")
        return float(ki / kj)

    gamma = _unifac_gammas(components, list(xs), T_K)
    p0 = _vap_pressure(components[0], T_K)
    p1 = _vap_pressure(components[1], T_K)
    if gamma[1] * p1 <= 0.0:
        raise ValueError("second-component saturation pressure is non-positive")
    return float((gamma[0] * p0) / (gamma[1] * p1))


def design_ternary_extractive_column(
    light: str,
    heavy: str,
    entrainer: str,
    feed_composition: Sequence[float],
    feed_flow_mol_s: float = 1.0,
    operating_pressure_kPa: float = 101.325,
    distillate_purity_mole_fraction: float = 0.995,
    recovery: float = 0.98,
    entrainer_ratio: float = 2.0,
    alpha_source: str | None = None,
) -> dict[str, object]:
    """Short-cut extractive-distillation design for an arbitrary light/heavy
    key pair with a high-boiling solvent.

    Generic re-implementation of the Fenske / Underwood / Gilliland short-cut
    method used by ``design_extractive_distillation_column`` for *any* binary
    key pair ``(light, heavy)`` plus a third-component ``entrainer`` (e.g.
    ethyl-acetate / n-propyl-acetate separated with dimethyl sulfoxide).
    All equilibrium numbers come from the deterministic engine
    (:func:`relative_volatility_key` and :func:`bubble_temperature`).

    Returns the same design fields as :class:`ExtractiveColumnDesign` as a dict.
    """
    P = operating_pressure_kPa
    purity = distillate_purity_mole_fraction
    source = _resolved_alpha_source(alpha_source)
    feed_components = [light, heavy, entrainer]
    x_fl = float(feed_composition[0])
    x_fh = float(feed_composition[1])
    if abs(x_fl + x_fh - 1.0) > 1e-6:
        raise ValueError("feed_composition must be two normalized key mole fractions")

    # Reference feed temperature = isobaric bubble temperature of the binary key
    # feed at the operating pressure (no entrainer), matching the v3 convention.
    T_feed = bubble_temperature([light, heavy], [x_fl, x_fh], P, source)

    # Base (binary, no entrainer) and extractive-section relative volatilities.
    alpha_base = relative_volatility_key([light, heavy], [x_fl, x_fh], T_feed, source)
    xs_ext = [0.10, 0.10, 0.80]  # representative entrainer-rich composition
    alpha_ext = relative_volatility_key(feed_components, xs_ext, T_feed, source)
    selectivity = alpha_ext / alpha_base
    alpha_avg = (alpha_base * alpha_ext) ** 0.5

    # Product material balance (light key to overhead, heavy key + entrainer to
    # bottoms, negligible entrainer carry-over into the distillate).
    n_l = feed_flow_mol_s * x_fl
    n_h = feed_flow_mol_s * x_fh
    d_l = recovery * n_l
    d_h = (1.0 - purity) / purity * d_l
    D = d_l + d_h
    x_dl = d_l / D
    b_l = n_l - d_l
    b_h = n_h - d_h
    b_s = entrainer_ratio * feed_flow_mol_s
    B = b_l + b_h + b_s
    x_bl = b_l / B
    x_bh = b_h / B
    x_bs = b_s / B

    # Fenske minimum stages.
    fenske_ratio = (x_dl / (1.0 - x_dl)) / (x_bl / (1.0 - x_bl))
    n_min = log(fenske_ratio) / log(alpha_avg) if alpha_avg != 1.0 else 0.0
    n_min = max(n_min, 0.5)

    # Underwood minimum reflux (constant-alpha binary form).
    z_F = x_fl
    r_min = (x_dl / z_F - alpha_avg * (1.0 - x_dl) / (1.0 - z_F)) / (alpha_avg - 1.0)
    r_min = max(r_min, 0.05)

    # Gilliland reflux/stage trade-off (operating R = 1.4 x R_min).
    reflux_factor = 1.4
    R = reflux_factor * r_min
    x_gill = (R - r_min) / (R + 1.0)
    y_gill = 0.75 * (1.0 - x_gill**0.5668)
    stages = int(np.ceil((n_min + y_gill) / (1.0 - y_gill))) if y_gill < 1.0 else int(np.ceil(n_min + 5))
    stages = max(stages, 3)
    feed_stage = int(np.floor(0.45 * (stages + 1)))
    feed_stage = max(2, min(feed_stage, stages - 1))
    entrainer_stage = 2
    if entrainer_stage >= feed_stage:
        entrainer_stage = max(1, feed_stage - 1)

    # Condenser / reboiler bubble temperatures.
    t_cond = bubble_temperature([light, heavy], [x_dl, 1.0 - x_dl], P, source)
    t_reb = bubble_temperature(
        feed_components, [x_bl, x_bh, x_bs], P, source
    )

    return {
        "light": light,
        "heavy": heavy,
        "entrainer": entrainer,
        "feed_temperature_K": round(T_feed, 2),
        "alpha_source": source,
        "alpha_base": round(alpha_base, 4),
        "alpha_ext": round(alpha_ext, 4),
        "selectivity": round(selectivity, 4),
        "alpha_avg": round(alpha_avg, 4),
        "minimum_stages": round(n_min, 3),
        "minimum_reflux_ratio": round(r_min, 3),
        "reflux_ratio": round(R, 3),
        "theoretical_stages": stages,
        "feed_stage": feed_stage,
        "entrainer_stage": entrainer_stage,
        "condenser_temperature_K": round(t_cond, 2),
        "reboiler_temperature_K": round(t_reb, 2),
        "operating_pressure_kPa": P,
        "distillate_purity_mole_fraction": round(x_dl, 6),
        "distillate_flow_mol_s": round(D, 6),
        "bottoms_flow_mol_s": round(B, 6),
        "warnings": [
            "ThermoFormer is a predictive ML backend; results are approximate and unvalidated for this system."
            if source == "thermoformer"
            else "activity coefficients from UNIFAC temperature-dependent prediction"
        ],
    }


def recommend_entrainer_for(
    light: str,
    heavy: str,
    feed_composition: Sequence[float],
    feed_temperature_K: float,
    candidates: Sequence[str] = DEFAULT_ENTRAINER_CANDIDATES,
    alpha_source: str | None = None,
) -> EntrainerRecommendation:
    """Rank entrainer candidates for an *arbitrary* light/heavy key pair.

    This is the generic counterpart of :func:`recommend_extraction_entrainer`
    which only scores the hard-coded ``ethanol``/``water`` feed.  Each candidate
    is evaluated at a representative entrainer-rich liquid composition
    (``[0.1, 0.1, 0.8]``) in the ternary ``[light, heavy, candidate]``.  The
    ranking key is the candidate's extracted-section relative volatility; a
    candidate whose UNIFAC subgroups cannot be resolved (e.g. not present in the
    ``thermo`` compound database) is skipped deterministically, never guessed.
    Equilibrium numbers always come from the deterministic engine.

    ``alpha_source`` follows :func:`relative_volatility_key` (``unifac`` is the
    default, ``thermoformer`` optionally uses the neural bubble-point backend).

    This entry point is used by parallel extractive-distillation paths that
    recover a user-selected light key (e.g. isopropanol) from ``heavy`` water.
    """
    if len(feed_composition) != 2:
        raise ValueError("feed_composition must be two normalized key mole fractions")
    if abs(float(feed_composition[0]) + float(feed_composition[1]) - 1.0) > 1e-6:
        raise ValueError("feed_composition must sum to one within 1e-6")
    T_K = feed_temperature_K
    source = _resolved_alpha_source(alpha_source)

    binary = [light, heavy]
    alpha_base = relative_volatility_key(
        binary, [float(feed_composition[0]), float(feed_composition[1])], T_K, source
    )

    entries: list[EntrainerCandidate] = []
    for candidate in sorted({str(c) for c in candidates}):
        comps = [light, heavy, candidate]
        xs = [0.1, 0.1, 0.8]
        try:
            alpha_ext = relative_volatility_key(comps, xs, T_K, source)
        except (ValueError, Exception):  # noqa: BLE001 - unresolvable -> skip
            continue
        source_label = (
            "ThermoFormer-predicted" if source == "thermoformer" else "UNIFAC-predicted"
        )
        entries.append(
            EntrainerCandidate(
                name=candidate,
                selectivity=alpha_ext / alpha_base,
                relative_volatility=float(alpha_ext),
                recovered_at_bottom=True,
                note=(
                    f"{source_label} {light}/{heavy} relative volatility = {alpha_ext:.3f} "
                    f"(baseline binary = {alpha_base:.3f})"
                ),
            )
        )

    if not entries:
        raise ValueError("No evaluable entrainer candidate found for the current feed.")
    entries.sort(key=lambda c: c.relative_volatility, reverse=True)
    return EntrainerRecommendation(recommended=entries[0].name, candidates=entries)


def design_generic_extractive_column(
    spec: ExtractiveColumnSpec,
    alpha_source: str | None = None,
) -> ExtractiveColumnDesign:
    """Typed short-cut design for an *arbitrary* light/heavy binary extractive feed.

    Generic, orthogonal counterpart of :func:`design_extractive_distillation_column`
    (which is hard-coded to an ethanol/water feed).  It accepts an
    :class:`ExtractiveColumnSpec` whose ``feed_components`` name any two keys
    (e.g. ``["isopropanol", "water"]``) plus a third-high-boiling ``entrainer``.

    All equilibrium numbers come from the deterministic engine via
    :func:`relative_volatility_key` and :func:`bubble_temperature` and the
    short-cut algebra in :func:`design_ternary_extractive_column`; no binary
    parameter is fabricated.  The returned :class:`ExtractiveColumnDesign` is
    the typed model the Web ``extractive`` payload and the generic DWSIM export
    expect, so a parallel light/heavy route reuses the existing frontend and
    export plumbing unchanged.
    """
    if len(spec.feed_components) != 2:
        raise ValueError("generic extractive design requires a binary feed")
    if len(spec.feed_composition) != 2:
        raise ValueError("feed_composition must match the two feed components")
    if spec.entrainer in (spec.feed_components[0], spec.feed_components[1]):
        raise ValueError("entrainer must be distinct from both feed components")

    light, heavy = str(spec.feed_components[0]), str(spec.feed_components[1])
    source = _resolved_alpha_source(alpha_source)
    P = spec.operating_pressure_kPa
    x_light, x_heavy = float(spec.feed_composition[0]), float(spec.feed_composition[1])

    result = design_ternary_extractive_column(
        light=light,
        heavy=heavy,
        entrainer=spec.entrainer,
        feed_composition=[x_light, x_heavy],
        feed_flow_mol_s=spec.feed_flow_mol_s,
        operating_pressure_kPa=P,
        distillate_purity_mole_fraction=spec.distillate_purity_mole_fraction,
        recovery=getattr(spec, "ethanol_recovery", 0.98),
        entrainer_ratio=spec.entrainer_ratio,
        alpha_source=source if source != "unifac" else None,
    )
    f = float

    use_thermoformer = source == "thermoformer"
    assumptions = [
        "constant relative volatility model (Fenske/Underwood/Gilliland-Eduljee)",
        "operating reflux ratio = 1.4 x minimum reflux ratio",
        f"high-boiling entrainer ({spec.entrainer}) recovered entirely in the bottoms",
        "negligible entrainer carry-over into the distillate",
        (
            "relative volatility from ThermoFormer bubble-point (ML) predictions"
            if use_thermoformer
            else "activity coefficients from UNIFAC temperature-dependent prediction"
        ),
        "total condenser",
    ]
    warnings: list[str] = []
    if use_thermoformer:
        warnings.append(
            "ThermoFormer is a predictive ML backend; results are approximate and have not "
            "been validated against experimental data for this system."
        )
    if f(result["alpha_avg"]) <= 1.15:
        warnings.append(
            f"Enhancement is weak (relative volatility {f(result['alpha_avg']):.3f}); "
            "column will need many stages."
        )

    _feed_stage = int(result["feed_stage"])
    _ent_stage = int(result["entrainer_stage"])

    # Reconstruct the bottoms composition (light, heavy, entrainer) from the
    # deterministic material balance so the typed model's composition validator
    # (which requires three normalised entries) is satisfied honestly.
    _feed_flow = spec.feed_flow_mol_s
    _n_l = _feed_flow * x_light
    _n_h = _feed_flow * x_heavy
    _D = f(result["distillate_flow_mol_s"])
    _d_l = _D * f(result["distillate_purity_mole_fraction"])
    _d_h = _D - _d_l
    _b_s = spec.entrainer_ratio * _feed_flow
    _B = f(result["bottoms_flow_mol_s"])
    _b_l = _n_l - _d_l
    _b_h = _n_h - _d_h
    bottoms_composition = [_b_l / _B if _B > 0 else 0.0,
                           _b_h / _B if _B > 0 else 0.0,
                           _b_s / _B if _B > 0 else 1.0]

    if _ent_stage >= _feed_stage:
        raise ValueError(
            "short-cut design produced entrainer stage at/above the feed stage; "
            "cannot build a valid typed design for this system."
        )

    return ExtractiveColumnDesign(
        spec=spec,
        theoretical_stages=int(result["theoretical_stages"]),
        minimum_stages=f(result["minimum_stages"]),
        reflux_ratio=f(result["reflux_ratio"]),
        minimum_reflux_ratio=f(result["minimum_reflux_ratio"]),
        feed_stage=_feed_stage,
        entrainer_stage=_ent_stage,
        relative_volatility=f(result["alpha_avg"]),
        base_relative_volatility=f(result["alpha_base"]),
        selectivity=f(result["selectivity"]),
        condenser_temperature_K=f(result["condenser_temperature_K"]),
        reboiler_temperature_K=f(result["reboiler_temperature_K"]),
        operating_pressure_kPa=P,
        distillate_purity_mole_fraction=f(result["distillate_purity_mole_fraction"]),
        bottoms_composition=[round(float(x), 8) for x in bottoms_composition],
        distillate_flow_mol_s=f(result["distillate_flow_mol_s"]),
        bottoms_flow_mol_s=f(result["bottoms_flow_mol_s"]),
        warnings=warnings,
        assumptions=assumptions,
        needs_validation=False,
        backend_version=_BACKEND_VERSION + ("+thermoformer" if use_thermoformer else ""),
    )


__all__ = [
    "DEFAULT_ENTRAINER_CANDIDATES",
    "BinaryDistillationDesign",
    "activity_coefficients",
    "bubble_temperature",
    "design_binary_distillation_column",
    "design_extractive_distillation_column",
    "design_generic_extractive_column",
    "design_ternary_extractive_column",
    "recommend_entrainer_for",
    "recommend_extraction_entrainer",
    "relative_volatility_eivw",
    "relative_volatility_key",
    "unifac_bubble_pressure",
]
