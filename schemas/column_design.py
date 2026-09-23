"""Deterministic extractive-distillation column-design schemas.

These contracts capture the inputs and outputs of the short-cut design model in
``thermo_engine.column_design``.  Every numerical stage count, reflux ratio,
temperature and pressure is produced by the deterministic engine (Fenske /
Underwood / Gilliland with entrainer-selectivity enhancement); the LLM never
invents these numbers.  The ``needs_validation`` flag is kept for parity with
the flow-design skill and is set to ``False`` once the engine has computed and
checked the design.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Property packages DWSIM can map to for the three-component extractive column.
#: Keys match ``thermo_engine.dwsim_export._PROPERTY_PACKAGES`` where possible.
PropertyPackage = Literal["NRTL", "UNIQUAC", "Wilson", "Peng-Robinson (PR)", "Ideal"]


class ExtractiveColumnSpec(BaseModel):
    """Inputs to the deterministic extractive-distillation design.

    The feed is an ethanol/water binary stream; the entrainer (extractant) is a
    third component added near the top to enhance the ethanol/water relative
    volatility so high-purity ethanol can be recovered overhead while water and
    the entrainer leave the bottom.
    """

    #: Feed component names in the order matching ``feed_composition``.
    feed_components: list[str] = Field(min_length=2)
    #: Feed mole fractions (ethanol/water) summing to one.
    feed_composition: list[float] = Field(min_length=2)
    #: Feed total molar flow rate, mol/s.
    feed_flow_mol_s: float = Field(gt=0)
    #: Feed temperature, K.
    feed_temperature_K: float = Field(gt=0)
    #: Feed pressure, kPa.
    feed_pressure_kPa: float = Field(gt=0)

    #: Entrainer component name (e.g. ethylene glycol / glycerine).
    entrainer: str = Field(min_length=1)
    #: Entrainer molar flow rate relative to the feed (mol entrainer / mol feed).
    #: A value >= 1 drives effectively complete entrainment of the components.
    entrainer_ratio: float = Field(default=2.0, gt=0)

    #: Target mole-fraction purity of ethanol in the overhead distillate.
    distillate_purity_mole_fraction: float = Field(default=0.995, gt=0, lt=1)
    #: Fractional recovery of feed ethanol into the overhead distillate (0..1).
    ethanol_recovery: float = Field(default=0.98, gt=0, le=1)

    #: Operating pressure of the column, kPa. Defaults to atmospheric.
    operating_pressure_kPa: float = Field(default=101.325, gt=0)

    #: Property package DWSIM should use to re-solve the rendered unit.
    property_package: PropertyPackage = "NRTL"

    @field_validator("feed_composition")
    @classmethod
    def _composition_fits_components(cls, value: list[float], info: object) -> list[float]:
        del info
        if any(not isfinite(v) or v < 0 or v > 1 for v in value):
            raise ValueError("feed_composition must contain mole fractions in [0, 1]")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError("feed_composition must sum to one within 1e-6")
        return value

    @model_validator(mode="after")
    def _feed_length(self) -> "ExtractiveColumnSpec":
        if len(self.feed_components) != len(self.feed_composition):
            raise ValueError("feed_components and feed_composition must have the same length")
        return self


class EntrainerCandidate(BaseModel):
    """One candidate entrainer ranked by its selectivity enhancement.

    ``selectivity`` is the entrainer-induced enhancement of the ethanol/water
    relative volatility; ``relative_volatility`` is the effective value used by
    the Ferning/Underwood/Gilliland stage model.  Both come from deterministic
    activity-coefficient evaluation in ``thermo_engine``.
    """

    name: str = Field(min_length=1)
    selectivity: float = Field(gt=0)
    relative_volatility: float = Field(gt=0)
    recovered_at_bottom: bool = True
    note: str | None = None


class EntrainerRecommendation(BaseModel):
    """Ranked entrainer short-list for an extractive column.

    ``recommended`` names the top candidate.  ``candidates`` are ordered by
    descending selectivity so consumers can inspect the trade-off.
    """

    recommended: str = Field(min_length=1)
    candidates: list[EntrainerCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def _recommended_present(self) -> "EntrainerRecommendation":
        first = self.candidates[0].name if self.candidates else self.recommended
        if not any(c.name == self.recommended for c in self.candidates):
            raise ValueError("recommended entrainer must appear in candidates")
        return self


class EntrainerOption(BaseModel):
    """One entrainer choice offered back to the user for confirmation.

    This is the chat-facing projection of an :class:`EntrainerCandidate`; it
    carries everything the frontend needs to render a picker and let the user
    confirm which entrainer to run the short-cut design with.
    """

    name: str = Field(min_length=1)
    canon_name: str = Field(min_length=1)
    selectivity: float = Field(gt=0)
    relative_volatility: float = Field(gt=0)
    recommended: bool = False
    note: str | None = None


class ExtractiveColumnDesign(BaseModel):
    """Deterministic output of the extractive-distillation short-cut design.

    All quantitative fields are produced by ``thermo_engine.column_design`` and
    are deterministic for a given :class:`ExtractiveColumnSpec`.  They feed
    directly into ``dwsim_export.export_dwsim_extractive_column``.
    """

    model_config = ConfigDict(use_attribute_docstrings=False)

    spec: ExtractiveColumnSpec

    #: Number of theoretical stages in the column (including reboiler, excluding
    #: the total condenser).
    theoretical_stages: int = Field(ge=1)
    #: Minimum stages from the Fenske equation (for reporting/validation).
    minimum_stages: float = Field(gt=0)
    #: Operating reflux ratio, L/D.
    reflux_ratio: float = Field(gt=0)
    #: Minimum reflux ratio from Underwood (for reporting/validation).
    minimum_reflux_ratio: float = Field(gt=0)
    #: Stage (numbering from the top, 1 = top tray) on which the feed enters.
    feed_stage: int = Field(ge=1)
    #: Stage (numbering from the top) on which the entrainer is introduced.
    entrainer_stage: int = Field(ge=1)

    #: Effective ethanol/water relative volatility used by the stage model.
    relative_volatility: float = Field(gt=0)
    #: Reference ethanol/water relative volatility without entrainer.
    base_relative_volatility: float = Field(gt=0)
    #: Entrainer selectivity enhancement factor (relative_volatility / base).
    selectivity: float = Field(gt=0)

    #: Overhead (condenser) bubble temperature, K.
    condenser_temperature_K: float = Field(gt=0)
    #: Bottom (reboiler) temperature, K.
    reboiler_temperature_K: float = Field(gt=0)
    #: Operating pressure, kPa (echoed from the spec).
    operating_pressure_kPa: float = Field(gt=0)

    #: Overhead ethanol mole fraction actually reconciled by the design.
    distillate_purity_mole_fraction: float = Field(gt=0, lt=1)
    #: Bottom composition (ethanol, water, entrainer) mole fractions.
    bottoms_composition: list[float] = Field(min_length=3)
    #: Overhead vapor flow, mol/s (D = V at total reflux top? kept as D).
    distillate_flow_mol_s: float = Field(gt=0)
    #: Bottoms product flow, mol/s.
    bottoms_flow_mol_s: float = Field(gt=0)

    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    #: False once the deterministic engine has validated the overall balances.
    needs_validation: bool = False
    #: Backend version that produced this design.
    backend_version: str = Field(default="", min_length=1)

    @model_validator(mode="after")
    def _consistency(self) -> "ExtractiveColumnDesign":
        if len(self.bottoms_composition) != 3:
            raise ValueError("bottoms_composition must contain exactly [ethanol, water, entrainer]")
        if abs(sum(self.bottoms_composition) - 1.0) > 1e-6:
            raise ValueError("bottoms_composition must sum to one within 1e-6")
        if self.entrainer_stage >= self.feed_stage:
            # Entrainer is always introduced above the feed tray.
            raise ValueError("entrainer_stage must be above feed_stage")
        if self.feed_stage >= self.theoretical_stages:
            raise ValueError("feed_stage must be below the total number of theoretical stages")
        if self.reflux_ratio <= self.minimum_reflux_ratio:
            raise ValueError("reflux_ratio must exceed minimum_reflux_ratio")
        return self


class BinaryDistillationResult(BaseModel):
    """Structured result of a plain (non-extractive) binary distillation design.

    Serialized form of the engine result returned by
    :func:`thermo_engine.column_design.design_binary_distillation_column`.
    """

    components: list[str] = Field(min_length=2, max_length=2)
    feed_composition: list[float] = Field(min_length=2, max_length=2)
    feed_flow_mol_s: float = Field(gt=0)
    operating_pressure_kPa: float = Field(gt=0)
    distillate_purity_mole_fraction: float = Field(gt=0, lt=1)
    relative_volatility: float = Field(gt=0)
    minimum_stages: float = Field(gt=0)
    minimum_reflux_ratio: float = Field(gt=0)
    reflux_ratio: float = Field(gt=0)
    theoretical_stages: int = Field(ge=1)
    feed_stage: int = Field(ge=1)
    condenser_temperature_K: float = Field(gt=0)
    reboiler_temperature_K: float = Field(gt=0)
    distillate_flow_mol_s: float = Field(gt=0)
    bottoms_flow_mol_s: float = Field(gt=0)
    alpha_source: str = ""
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "BinaryDistillationResult":
        valids = [self.reboiler_temperature_K > self.condenser_temperature_K,
                  self.reflux_ratio > self.minimum_reflux_ratio]
        # note: the temperature ordering can invert when alpha<1 (model-degenerate),
        # so we only enforce it when alpha>1.
        if self.relative_volatility <= 1.0:
            valids = [self.reflux_ratio > self.minimum_reflux_ratio]
        if not all(valids):
            raise ValueError("inconsistent binary distillation result")
        return self


class BinaryDistillationPayload(BaseModel):
    """Chat-facing result of a plain binary-distillation request.

    ``status`` distinguishes a computed design from a missing/invalid request and
    from a design that ran but could not be rendered because DWSIM is not
    installed/configured in this environment.  ``result`` is populated whenever
    the deterministic short-cut design ran -- including under
    ``status="dwsim_unavailable"``, so the computed stages/reflux/temperatures
    still reach the user.  ``dwsim_file_uri`` is present only when a DWSIM
    ``.dwxmz`` was actually generated.
    """

    status: Literal["ready", "missing_parameters", "dwsim_unavailable", "failed"]
    result: BinaryDistillationResult | None = None
    missing_parameters: list[str] = Field(default_factory=list)
    message: str = Field(default="")
    alpha_source: str | None = None
    #: Bubble temperature of the saturated-liquid feed, K.  Used as the DWSIM
    #: feed-stream temperature so the rendered column sits on its bubble point.
    feed_temperature_K: float | None = Field(default=None, gt=0)
    #: Relative URI to download the generated ``.dwxmz`` file.
    dwsim_file_uri: str | None = None
    #: Stable id identifying the stored file for the download endpoint.
    file_id: str | None = None


class ExtractiveExportPayload(BaseModel):
    """Chat-facing result of an extractive-distillation request.

    ``status`` distinguishes a ready download from a request that still needs
    parameters, from a request waiting for the user to pick an entrainer, or
    that could not reach DWSIM.  ``design`` is always present when the
    deterministic model could run, even if the DWSIM file could not be produced
    (so the user still gets the computed stages/reflux/temperatures).
    """

    status: Literal["ready", "awaiting_entrainer", "missing_parameters", "dwsim_unavailable", "failed"]
    #: Deterministic short-cut design (theoretical stages, reflux ratio, T/P).
    design: ExtractiveColumnDesign | None = None
    #: Local-model-ranked entrainer options for the user to confirm.  Present
    #: when ``status == "awaiting_entrainer"``.
    entrainer_candidates: list[EntrainerOption] = Field(default_factory=list)
    #: Effective activity source used for scoring (``unifac`` or ``thermoformer``).
    alpha_source: str | None = None
    #: Relative URI to download the generated ``.dwxmz`` file.
    dwsim_file_uri: str | None = None
    #: Stable id identifying the stored file for the download endpoint.
    file_id: str | None = None
    #: Parameter names still required from the user.
    missing_parameters: list[str] = Field(default_factory=list)
    #: Human-readable outcome explaining the status to the caller.
    message: str = Field(default="")


class LLEExtractionSpec(BaseModel):
    """Inputs to the DWSIM liquid-liquid extraction (decanter) export.

    The feed is a mixture of two solutes separated from one another by a third
    extraction ``solvent`` (e.g. the ternary n-propyl acetate + ethyl acetate
    solutes using DMSO as a selective solvent).  The spec carries only the
    flowsheet inputs; equilibrium numbers are never computed by this system —
    DWSIM resolves the liquid-liquid split with its own NRTL parameters.
    """

    #: Feed solute component names in the order matching ``feed_composition``.
    feed_components: list[str] = Field(min_length=2)
    #: Feed solute mole fractions summing to one.
    feed_composition: list[float] = Field(min_length=2)
    #: The extraction solvent component name (third component of the system).
    solvent: str = Field(min_length=1)
    #: Solvent molar flow relative to the feed (mol solvent / mol feed).
    solvent_ratio: float = Field(default=1.0, gt=0)
    #: Feed total molar flow rate, mol/s.
    feed_flow_mol_s: float = Field(default=1.0, gt=0)
    #: Feed/separator temperature, K.
    feed_temperature_K: float = Field(default=298.15, gt=0)
    #: Feed/operating pressure, kPa.
    feed_pressure_kPa: float = Field(default=101.325, gt=0)
    #: DWSIM property package used to model the liquid-liquid split.
    property_package: PropertyPackage = "NRTL"

    @field_validator("feed_composition")
    @classmethod
    def _composition_fits_components(cls, value: list[float], info: object) -> list[float]:
        del info
        if any(not isfinite(v) or v < 0 or v > 1 for v in value):
            raise ValueError("feed_composition must contain mole fractions in [0, 1]")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError("feed_composition must sum to one within 1e-6")
        return value

    @model_validator(mode="after")
    def _lengths_and_solvents(self) -> "LLEExtractionSpec":
        if len(self.feed_components) != len(self.feed_composition):
            raise ValueError("feed_components and feed_composition must have the same length")
        if len(self.feed_components) != 2:
            raise ValueError("LLE extraction requires exactly two feed solutes")
        if self.solvent in self.feed_components:
            raise ValueError("the extraction solvent must be distinct from both feed solutes")
        return self


class LLEExportPayload(BaseModel):
    """Chat-facing result of a liquid-liquid extraction export request.

    Covers both liquid-liquid flows:

    * a **ternary solvent extraction** (two solutes + a selective solvent) whose
      ``spec`` carries a distinct ``solvent``, and
    * a **binary partially miscible pair** (e.g. water / 1-butanol) rendered from
      the ``report/dwsim`` ``Vessel_LLE`` template, whose inputs live in
      ``binary_spec`` because such a pair has no third extraction solvent.

    Because the production backends cannot compute LLE numerically, the only
    output of a successful run is the structural DWSIM ``.dwxmz`` file plus the
    structured echo of the resolved inputs.  ``dwsim_file_uri`` is present when
    the file was written.
    """

    status: Literal["ready", "missing_parameters", "dwsim_unavailable", "failed"]
    spec: LLEExtractionSpec | None = None
    dwsim_file_uri: str | None = None
    file_id: str | None = None
    missing_parameters: list[str] = Field(default_factory=list)
    message: str = Field(default="")
    warnings: list[str] = Field(default_factory=list)
    #: ``"binary"`` for a two-component partially miscible pair rendered from the
    #: ``report/dwsim`` template, ``"ternary"`` for a solvent extraction.
    lle_kind: Literal["binary", "ternary"] = "ternary"
    #: Fully-resolved inputs of a *binary* LLE case.  Present only when
    #: ``lle_kind == "binary"``; the ternary path uses :attr:`spec` instead.  A
    #: binary partially miscible pair has no third extraction solvent, so it does
    #: not fit :class:`LLEExtractionSpec` and uses :class:`BinaryCaseSpec`.
    binary_spec: "BinaryCaseSpec | None" = None
    #: Fully-resolved inputs of a *generic three-component* LLE case.  Present
    #: only for the generic ternary path, where all three components are simply
    #: named and there is no separate solvent stream.
    ternary_spec: "TernaryLLESpec | None" = None


#: Binary phase-equilibrium kinds the generic binary DWSIM export can render.
BinaryCaseKind = Literal["vle", "lle", "vlle"]

#: DWSIM flowsheet topologies a binary case can be rendered as.
BinaryFlowsheetMode = Literal["tp_flash", "bubble_point", "two_liquid_vessel", "extractor"]


class BinaryCaseRequirement(BaseModel):
    """One user-supplied field a binary DWSIM case still needs.

    Emitted as machine-readable configuration so the frontend (or a user editing
    a JSON file) can collect exactly what is missing instead of guessing.  The
    generic binary flow infers everything it can, so this list is normally short
    or empty.
    """

    #: Stable programmatic key, e.g. ``components`` or ``feed_composition``.
    key: str = Field(min_length=1)
    #: Human-readable label for the field.
    label: str = Field(min_length=1)
    #: JSON type the value must be, e.g. ``list[string]`` or ``number``.
    type: str = Field(min_length=1)
    #: Physical unit, when the field is numeric.
    unit: str | None = None
    #: Whether the case cannot be rendered without it.
    required: bool = True
    #: Value the system would use if the field is omitted and a default exists.
    default: object | None = None
    #: Short guidance on what to supply.
    description: str = Field(default="")


class BinaryCaseSpec(BaseModel):
    """Fully-resolved inputs for one binary DWSIM flowsheet export.

    A "binary case" is a two-component phase-equilibrium system rendered as a
    DWSIM flowsheet.  The same structure covers both fundamentally different
    physics, selected by ``kind``:

    * ``kind="vle"`` — a vapour-liquid system (e.g. 2-propanol / water) rendered
      as a TP flash, optionally positioned at the bubble point.
    * ``kind="lle"`` — a partially miscible liquid-liquid system (e.g.
      water / 1-butanol) rendered as a native two-liquid ``Vessel``, or as a
      solvent ``extractor`` when a third extraction solvent is supplied.

    Only flowsheet *inputs* live here.  Equilibrium numbers are never produced by
    this project — DWSIM resolves the flash with its own parameters.
    """

    #: Component names in the order matching ``feed_composition``.
    components: list[str] = Field(min_length=2, max_length=2)
    #: Overall feed mole fractions summing to one (binary pair only).
    feed_composition: list[float] = Field(min_length=2, max_length=2)
    #: Phase-equilibrium kind driving the flowsheet topology.
    kind: BinaryCaseKind = "vle"
    #: DWSIM flowsheet topology to render.
    mode: BinaryFlowsheetMode = "tp_flash"
    #: Optional third component acting as an extraction solvent (``kind="lle"``).
    solvent: str | None = None
    #: Solvent molar flow relative to the feed (mol solvent / mol feed).
    solvent_ratio: float = Field(default=1.0, gt=0)
    #: Feed total molar flow rate, mol/s.
    feed_flow_mol_s: float = Field(default=1.0, gt=0)
    #: Flash temperature, K.  ``None`` requests the bubble point instead.
    temperature_K: float | None = Field(default=None, gt=0)
    #: Operating pressure, kPa.
    pressure_kPa: float = Field(default=101.325, gt=0)
    #: DWSIM property package used for the flash.
    property_package: PropertyPackage = "NRTL"

    @field_validator("feed_composition")
    @classmethod
    def _fractions_in_range(cls, value: list[float]) -> list[float]:
        if any(not isfinite(v) or v < 0 or v > 1 for v in value):
            raise ValueError("feed_composition must contain mole fractions in [0, 1]")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError("feed_composition must sum to one within 1e-6")
        return value

    @model_validator(mode="after")
    def _kind_matches_topology(self) -> "BinaryCaseSpec":
        if len(self.components) != len(self.feed_composition):
            raise ValueError("components and feed_composition must have the same length")
        if len(set(c.casefold() for c in self.components)) != 2:
            raise ValueError("a binary case requires two distinct components")
        if self.mode == "bubble_point" and self.temperature_K is not None:
            raise ValueError("bubble_point mode derives the temperature; leave temperature_K unset")
        if self.mode == "extractor":
            if not self.solvent:
                raise ValueError("extractor mode requires a solvent component")
            if self.solvent.casefold() in {c.casefold() for c in self.components}:
                raise ValueError("the extraction solvent must be distinct from the binary pair")
        return self


class TernaryLLESpec(BaseModel):
    """Inputs for a generic three-component liquid-liquid DWSIM export.

    Unlike :class:`LLEExtractionSpec` (two solutes plus a *distinct* extraction
    solvent) or :class:`BinaryCaseSpec` (exactly two components), this describes
    three feed components on an equal footing: the case where a user simply names
    three components and their mutual solubility does the separating, with no
    separate solvent stream.  Rendered as the ``report/dwsim`` ``Vessel_LLE``
    template with three compounds.
    """

    #: Component names in the order matching ``feed_composition``.
    components: list[str] = Field(min_length=3, max_length=3)
    #: Overall feed mole fractions summing to one.
    feed_composition: list[float] = Field(min_length=3, max_length=3)
    #: Flash temperature, K.
    temperature_K: float = Field(default=298.15, gt=0)
    #: Operating pressure, kPa.
    pressure_kPa: float = Field(default=101.325, gt=0)
    #: Feed total molar flow rate, mol/s.
    feed_flow_mol_s: float = Field(default=1.0, gt=0)
    #: DWSIM property package used for the flash.
    property_package: PropertyPackage = "NRTL"

    @field_validator("feed_composition")
    @classmethod
    def _fractions_in_range(cls, value: list[float]) -> list[float]:
        if any(not isfinite(v) or v < 0 or v > 1 for v in value):
            raise ValueError("feed_composition must contain mole fractions in [0, 1]")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError("feed_composition must sum to one within 1e-6")
        return value

    @model_validator(mode="after")
    def _three_distinct_components(self) -> "TernaryLLESpec":
        if len(self.components) != len(self.feed_composition):
            raise ValueError("components and feed_composition must have the same length")
        if len({c.casefold() for c in self.components}) != 3:
            raise ValueError("a ternary LLE case requires three distinct components")
        return self


class BinaryCasePayload(BaseModel):
    """Chat-facing result of a generic binary DWSIM export request.

    ``status="ready"`` carries the download URI plus the resolution report; the
    other statuses carry ``requirements`` so a caller can collect the missing
    configuration and retry.
    """

    status: Literal["ready", "missing_parameters", "dwsim_unavailable", "failed"]
    spec: BinaryCaseSpec | None = None
    #: Relative URI to download the generated ``.dwxmz`` file.
    dwsim_file_uri: str | None = None
    #: Stable id identifying the stored file for the download endpoint.
    file_id: str | None = None
    #: Fields still needed from the user; empty when nothing is required.
    requirements: list[BinaryCaseRequirement] = Field(default_factory=list)
    #: How each input was obtained (inferred / defaulted / user-supplied).
    resolution: dict[str, str] = Field(default_factory=dict)
    #: DWSIM phase results read back after the flash, when available.
    results: dict[str, object] = Field(default_factory=dict)
    missing_parameters: list[str] = Field(default_factory=list)
    message: str = Field(default="")
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "BinaryCaseKind",
    "BinaryCasePayload",
    "BinaryCaseRequirement",
    "BinaryCaseSpec",
    "BinaryDistillationPayload",
    "BinaryDistillationResult",
    "BinaryFlowsheetMode",
    "EntrainerCandidate",
    "EntrainerOption",
    "EntrainerRecommendation",
    "ExtractiveColumnDesign",
    "ExtractiveColumnSpec",
    "ExtractiveExportPayload",
    "LLEExtractionSpec",
    "LLEExportPayload",
    "PropertyPackage",
    "TernaryLLESpec",
]
