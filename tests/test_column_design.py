"""Behavioral tests for the deterministic extractive-distillation design model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from schemas.column_design import ExtractiveColumnSpec
from thermo_engine.column_design import (
    design_binary_distillation_column,
    design_extractive_distillation_column,
    recommend_extraction_entrainer,
    relative_volatility_eivw,
)

_EG_SPEC_KWARGS: dict[str, object] = {
    "feed_components": ["ethanol", "water"],
    "feed_composition": [0.40, 0.60],
    "feed_flow_mol_s": 1.0,
    "feed_temperature_K": 298.15,
    "feed_pressure_kPa": 101.325,
    "entrainer": "ethylene glycol",
    "operating_pressure_kPa": 101.325,
}


def _spec(**overrides: object) -> ExtractiveColumnSpec:
    return ExtractiveColumnSpec(**{**_EG_SPEC_KWARGS, **overrides})


def test_relative_volatility_is_positive_and_finite() -> None:
    alpha = relative_volatility_eivw(["ethanol", "water"], [0.5, 0.5], 350.0)
    assert math.isfinite(alpha)
    assert alpha > 0.0
    assert alpha > 1.0  # ethanol should be more volatile than water here


def test_binary_design_accepts_traceable_external_vle_values() -> None:
    """A DWSIM-derived alpha and product bubble temperatures are preserved.

    This enables a direct binary-column file for pairs whose volatility data is
    supplied by the installed DWSIM property package rather than silently
    substituting a different in-process model.
    """
    design = design_binary_distillation_column(
        ["light", "heavy"], [0.47, 0.53], feed_temperature_K=350.0,
        relative_volatility_override=1.05,
        condenser_temperature_K_override=348.0,
        reboiler_temperature_K_override=352.0,
    )
    assert design.relative_volatility == pytest.approx(1.05)
    assert design.condenser_temperature_K == pytest.approx(348.0)
    assert design.reboiler_temperature_K == pytest.approx(352.0)
    assert design.theoretical_stages >= design.minimum_stages
    assert design.reflux_ratio > design.minimum_reflux_ratio


def test_recommend_extraction_entrainer_ranks_by_selectivity() -> None:
    recommendation = recommend_extraction_entrainer(_spec())
    assert recommendation.recommended == recommendation.candidates[0].name
    # Candidates are monotonically decreasing by relative volatility.
    values = [c.relative_volatility for c in recommendation.candidates]
    assert values == sorted(values, reverse=True)
    # Both default candidates (ethylene glycol, glycerol) must be evaluable.
    assert {c.name for c in recommendation.candidates} >= {"ethylene glycol", "glycerol"}


def test_design_produces_internally_consistent_numbers() -> None:
    design = design_extractive_distillation_column(_spec())
    assert design.theoretical_stages >= design.minimum_stages
    assert design.reflux_ratio > design.minimum_reflux_ratio
    assert design.feed_stage < design.theoretical_stages
    assert design.entrainer_stage < design.feed_stage
    assert design.selectivity > 0.0
    assert design.condenser_temperature_K < design.reboiler_temperature_K
    assert design.distillate_purity_mole_fraction >= 0.995
    assert math.isclose(sum(design.bottoms_composition), 1.0, abs_tol=1e-6)


def test_design_reflects_higher_purity_in_more_stages_or_reflux() -> None:
    tol = 1e-6
    base = design_extractive_distillation_column(_spec())
    tight = design_extractive_distillation_column(
        _spec().model_copy(update={"distillate_purity_mole_fraction": 0.99999})
    )
    # A harder split must not demand *less* separation work.
    assert (tight.minimum_stages - base.minimum_stages) > -tol
    assert (tight.minimum_reflux_ratio - base.minimum_reflux_ratio) > -tol


def test_design_rejects_non_binary_feed() -> None:
    with pytest.raises(ValueError):
        design_extractive_distillation_column(
            _spec().model_copy(update={"feed_composition": [0.3, 0.3, 0.4]})
        )


def test_design_rejects_invalid_starting_spec() -> None:
    with pytest.raises(ValueError):
        # Constructed directly so Pydantic validation runs (model_copy skips it).
        ExtractiveColumnSpec(
            feed_components=["ethanol", "water"],
            feed_composition=[0.5, 0.4],  # does not sum to one
            feed_flow_mol_s=1.0,
            feed_temperature_K=298.15,
            feed_pressure_kPa=101.325,
            entrainer="ethylene glycol",
        )


def test_relative_volatility_via_thermoformer_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """With COLUMN_DESIGN_ALPHA_SOURCE=thermoformer the volatility is taken from
    the model's (x, T) -> (y, P) bubble-point vapor composition.

    The backend's bubble_point is mocked so the test does not need a loaded
    checkpoint.  A y_E/x_E > y_W/x_W composition must yield alpha > 1.
    """
    from thermo_engine import column_design

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            class _Point:
                liquid_composition = [0.5, 0.5]
                vapor_composition = [0.75, 0.25]

            class _Result:
                points = [_Point()]

            return _Result()

    monkeypatch.setattr(column_design, "_alpha_source", lambda: "thermoformer")
    monkeypatch.setattr(column_design, "_thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend",
        _FakeBackend,
    )
    monkeypatch.setenv("COLUMN_DESIGN_ALPHA_SOURCE", "thermoformer")

    alpha = relative_volatility_eivw(["ethanol", "water"], [0.5, 0.5], 350.0)
    # From mocked y=[0.75,0.25], x=[0.5,0.5]:
    assert math.isclose(alpha, (0.75 / 0.5) / (0.25 / 0.5), rel_tol=1e-9)  # == 3.0
    assert alpha > 1.0


def test_relative_volatility_thermoformer_override_ignores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-call ``alpha_source='thermoformer'`` override must be honoured even
    when the environment switch points at UNIFAC (no global env mutation)."""
    from thermo_engine import column_design

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            class _Point:
                liquid_composition = [0.5, 0.5]
                vapor_composition = [0.8, 0.2]

            class _Result:
                points = [_Point()]

            return _Result()

    monkeypatch.setenv("COLUMN_DESIGN_ALPHA_SOURCE", "unifac")  # default says UNIFAC
    monkeypatch.setattr(column_design, "_thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend",
        _FakeBackend,
    )

    # Per-call override wins over the env default -> the fake ML backend runs.
    alpha = relative_volatility_eivw(
        ["ethanol", "water"], [0.5, 0.5], 350.0, alpha_source="thermoformer"
    )
    assert math.isclose(alpha, (0.8 / 0.5) / (0.2 / 0.5), rel_tol=1e-9)  # == 4.0


def test_bubble_temperature_via_thermoformer_isobaric(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bubble_temperature(..., alpha_source='thermoformer')`` routes to the
    ThermoFormer isobaric branch: given x and P it predicts T (the manuscript
    Isobaric T--x--y direction)."""
    from thermo_engine import column_design

    received: list[tuple[object, list[float]]] = []

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            xs = list(request.conditions.liquid_composition)
            received.append((list(request.components), xs))
            assert request.conditions.pressure_kPa is not None  # isobaric branch
            class _Point:
                liquid_composition = xs
                vapor_composition = [0.7, 0.3]
                temperature_K = 351.45

            class _Result:
                points = [_Point()]

            return _Result()

    monkeypatch.setattr(column_design, "_thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeBackend
    )

    T = column_design.bubble_temperature(
        ["ethanol", "water"], [0.4, 0.6], 101.325, alpha_source="thermoformer"
    )
    assert math.isclose(T, 351.45, rel_tol=1e-9)
    assert len(received) == 1
    names = {comp.component_id for comp in received[0][0]}
    assert names == {"ethanol", "water"}
    # Isothermal-only route (no alpha_source) still solves via UNIFAC, no backend.
    assert column_design.bubble_temperature(["ethanol", "water"], [0.4, 0.6], 101.325) > 0


def test_design_thermoformer_path_uses_ternary_bubble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """design_extractive_distillation_column(alpha_source='thermoformer') runs a
    single-point bubble prediction at the exact ternary extractive composition
    (entrainer included) and flags the ML caveat."""
    from thermo_engine import column_design

    calls: list[tuple[object, list[float]]] = []

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            xs = list(request.conditions.liquid_composition)
            calls.append((list(request.components), xs))
            class _Point:
                liquid_composition = xs
                vapor_composition = [0.7, 0.2, 0.1]  # ethanol-rich vapor
                temperature_K = 351.15  # needed for the isobaric bubble-temperature calls

            class _Result:
                points = [_Point()]

            return _Result()

    monkeypatch.setattr(column_design, "_thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend",
        _FakeBackend,
    )

    design = design_extractive_distillation_column(_spec(), alpha_source="thermoformer")

    # Four ThermoFormer calls: 2 isothermal alpha evaluations (binary base +
    # ternary extractive section) and 2 isobaric bubble-temperature evaluations
    # (distillate + bottoms).
    assert len(calls) == 4
    # The ternary extractive-section alpha call must include the entrainer (ethylene
    # glycol) and run at the exact ternary composition [0.1, 0.1, 0.8].
    ternary = [c for c in calls if len(c[0]) == 3 and list(c[1]) == pytest.approx([0.1, 0.1, 0.8])]
    assert len(ternary) == 1
    names = {comp.component_id for comp in ternary[0][0]}
    assert names == {"ethanol", "water", "ethylene glycol"}
    # The user-facing caveat is surfaced.
    assert any("ThermoFormer is a predictive ML backend" in w for w in design.warnings)
    assert any("ThermoFormer" in a for a in design.assumptions)
    assert design.backend_version.endswith("+thermoformer")


def test_e2e_user_request_runs_through_thermoformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the user's exact request opting into ThermoFormer.

    A realistic ThermoFormer backend (checkpoint mocked away) predicts an
    entrainer-enhanced ethanol/water vapor split; the pipeline must thread
    ``alpha_source='thermoformer'``, invoke the ML bubble prediction for both the
    binary base and the ternary extractive section, and return a design that
    visibly uses ThermoFormer with the ML caveat.

    ``tmp_path`` is deliberately avoided so the case also runs under sandboxes
    that disallow pytest's tmp-dir teardown; a local temp export folder is used
    instead and cleaned up in a ``finally``.
    """
    import os
    import shutil
    import tempfile
    from unittest import mock

    import agent.extractive_distillation as extractive
    from thermo_engine import column_design

    export_dir = tempfile.mkdtemp(prefix="e2e_tf_", dir=os.getcwd())

    captured_alpha: list[str | None] = []
    bubble_calls: list[tuple[object, list[float]]] = []

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            xs = list(request.conditions.liquid_composition)
            bubble_calls.append((list(request.components), xs))
            # Entrainer (ethylene glycol) is high-boiling: keep it in the liquid,
            # ethanol is the most volatile species.
            y = [0.55, 0.40, 0.05] if len(xs) == 3 else [0.70, 0.30]
            class _Point:
                liquid_composition = xs
                vapor_composition = y
                temperature_K = 351.15  # needed for the isobaric bubble-temperature calls

            class _Result:
                points = [_Point()]

            return _Result()

    def _fake_design(spec: object, alpha_source: str | None = None) -> object:  # noqa: ARG002
        captured_alpha.append(alpha_source)
        from thermo_engine.column_design import design_extractive_distillation_column as real
        return real(spec, alpha_source="thermoformer")

    def fake_export(design: object, destination: Path, *_: object, **__: object) -> Path:  # noqa: ARG002
        destination.write_bytes(b"FAKE-DWSIM-COLUMN")
        return destination

    monkeypatch.setattr(column_design, "_thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeBackend
    )

    try:
        with (
            mock.patch.object(
                extractive, "export_dwsim_extractive_column", side_effect=fake_export
            ),
            mock.patch.object(
                extractive,
                "design_extractive_distillation_column",
                side_effect=_fake_design,
            ),
        ):
            payload = extractive.run_extractive_export(
                "乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，"
                "用乙二醇做萃取剂，塔顶乙醇纯度99.5%",
                export_dir=export_dir,
            )
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)

    # Opt-in triggered and threaded to the deterministic design.
    # ``status`` may be ``ready`` (file written) or ``dwsim_unavailable``
    # (sandbox/DWSIM blocked the file write) -- both prove the design itself ran
    # through ThermoFormer, which is the behaviour under test.
    assert payload.status in {"ready", "dwsim_unavailable"}
    assert captured_alpha == ["thermoformer"]
    assert payload.design is not None
    # The ML caveat must reach the user in the ready message.
    assert "ThermoFormer" in payload.message
    if payload.status == "ready":
        assert "ML 近似值" in payload.message

    # Design is well-formed and explicitly ThermoFormer-backed.
    design = payload.design
    assert design.backend_version.endswith("+thermoformer")
    assert any("ThermoFormer is a predictive ML backend" in w for w in design.warnings)
    assert any("ThermoFormer" in a for a in design.assumptions)

    # Four ThermoFormer calls: 2 isothermal alpha evaluations (binary base +
    # ternary extractive section) and 2 isobaric bubble-temperature evaluations
    # (distillate + bottoms).
    assert len(bubble_calls) == 4
    assert sorted(len(c[0]) for c in bubble_calls) == [2, 2, 3, 3]
    namesets = [" ".join(sorted(c.component_id for c in comps)) for comps, _ in bubble_calls]
    assert "ethanol water" in namesets
    assert "ethanol ethylene glycol water" in namesets
    # The ternary extractive-section alpha call evaluates exactly [0.1,0.1,0.8].
    ternary_alpha = [xs for comps, xs in bubble_calls if len(comps) == 3 and list(xs) == pytest.approx([0.1, 0.1, 0.8])]
    assert len(ternary_alpha) == 1


def test_binary_distillation_methanol_water() -> None:
    """Plain (non-extractive) binary design for methanol/water is well-formed and
    physically sane: alpha > 1 (methanol more volatile), positive stages/reflux,
    distillate hotter-colder relation consistent."""
    from thermo_engine.column_design import design_binary_distillation_column

    d = design_binary_distillation_column(["methanol", "water"], [0.5, 0.5])
    assert d.alpha_source == "unifac"
    # Methanol is more volatile than water.
    assert d.relative_volatility > 1.0
    assert d.theoretical_stages >= d.minimum_stages
    assert d.reflux_ratio > d.minimum_reflux_ratio
    assert 2 <= d.feed_stage < d.theoretical_stages
    assert d.distillate_flow_mol_s + d.bottoms_flow_mol_s == pytest.approx(1.0)
    assert d.condenser_temperature_K > 0
    assert d.reboiler_temperature_K > d.condenser_temperature_K
    # Components survive round-trip.
    assert d.components[0] == "methanol"
    assert d.components[1] == "water"
    # Assumption records that UNIFAC supplied the activity coefficients.
    assert any("UNIFAC" in a for a in d.assumptions)


def test_binary_distillation_rejects_bad_feed() -> None:
    from thermo_engine.column_design import design_binary_distillation_column

    with pytest.raises(ValueError):
        design_binary_distillation_column(["methanol", "water"], [0.5, 0.6])  # does not sum to 1
    with pytest.raises(ValueError):
        design_binary_distillation_column(["methanol"], [1.0])  # not binary


def test_binary_distillation_alpha_source_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """alpha_source='thermoformer' uses the ThermoFormer bubble vapor composition to
    derive alpha; the fake backend returns an ethanol-vapor-enriched result so
    alpha must come out from the model rather than UNIFAC."""
    from thermo_engine.column_design import (
        _COMPONENT_SMILES,
        design_binary_distillation_column,
    )

    _COMPONENT_SMILES.setdefault("methanol", "CO")
    _COMPONENT_SMILES.setdefault("water", "O")

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            xs = list(request.conditions.liquid_composition)
            class _Point:
                liquid_composition = xs
                vapor_composition = [0.7, 0.3]  # methanol-rich vapor
                temperature_K = 337.0
            class _Result:
                points = [_Point()]
            return _Result()

    monkeypatch.setattr("thermo_engine.column_design._thermoformer_backend", None)
    monkeypatch.setattr(
        "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeBackend
    )
    d = design_binary_distillation_column(
        ["methanol", "water"], [0.5, 0.5], alpha_source="thermoformer"
    )
    assert d.alpha_source == "thermoformer"
    # From mocked y=[0.7,0.3], x=[0.5,0.5]: alpha=(0.7/0.5)/(0.3/0.5)=2.333...
    assert math.isclose(d.relative_volatility, (0.7 / 0.5) / (0.3 / 0.5), rel_tol=1e-3)
    assert any("ThermoFormer" in a for a in d.assumptions)


def test_agent_binary_distillation_methanol_water() -> None:
    """The agent entry ``run_binary_distillation`` returns a structured
    design for a methanol-water distillation request.

    ``status`` may be ``ready`` (DWSIM file written) or ``dwsim_unavailable``
    (no DWSIM in this environment); both prove the design itself ran.
    """
    from agent.extractive_distillation import run_binary_distillation

    payload = run_binary_distillation("甲醇-水精馏塔，甲醇50%、水50%")
    assert payload.status in {"ready", "dwsim_unavailable"}
    assert payload.result is not None
    assert payload.result.components[0] == "methanol"
    assert payload.result.components[1] == "water"
    assert payload.result.relative_volatility > 1.0
    assert payload.result.theoretical_stages >= payload.result.minimum_stages
    assert payload.message  # human-readable summary present


def test_agent_binary_distillation_missing_feed() -> None:
    from agent.extractive_distillation import run_binary_distillation

    payload = run_binary_distillation("distillation column for methanol and water")
    # No composition parsed -> missing_parameters, not a crash.
    assert payload.status == "missing_parameters"


def test_intent_distillation_detection() -> None:
    from agent.extractive_distillation import is_distillation_request

    assert is_distillation_request("甲醇-水精馏塔设计")
    assert is_distillation_request("蒸馏塔，苯-甲苯")
    assert not is_distillation_request("甲醇-水萃取精馏")  # extractive takes precedence
