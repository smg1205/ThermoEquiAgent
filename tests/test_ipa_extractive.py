"""Behavioral tests for the isopropanol(IPA)/water extractive-distillation path.

This route is the *orthogonal* generic counterpart of the ethanol/water
extractive path: it recovers ``isopropanol`` from water using a high-boiling
third-component entrainer, ranking candidates and running the generic short-cut
design + generic DWSIM export.  All numbers come from the deterministic engine;
nothing is fabricated here.  These are deliberately sync tests that mirror
``tests/test_extractive_chat.py`` so they run without an async pytest plugin.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import agent.extractive_distillation as extractive
from agent.extractive_distillation import (
    is_extractive_distillation_request,
    is_ipa_extractive_distillation_request,
    run_extractive_export,
    run_ipa_extractive_export,
)


def test_intent_detection_recognises_ipa_water_extraction() -> None:
    hits = (
        "2-丙醇-水萃取精馏模拟",
        "帮我做异丙醇和水萃取精馏",
        "异丙醇-水萃取的dwsim模拟",
        "IPA water extractive distillation simulate",
    )
    for message in hits:
        assert is_ipa_extractive_distillation_request(message), message
        # IPA requests must also be classified as generic extractive so the
        # orchestrator routes them through ``run_extractive_export``.
        assert is_extractive_distillation_request(message), message

    # True ethanol/water extractive must NOT be misrouted to the IPA feed.
    assert not is_ipa_extractive_distillation_request(
        "乙醇水萃取精馏，乙醇40%水60%，用乙二醇"
    )
    # A non-extractive IPA/water note must not be treated as an extraction.
    assert not is_ipa_extractive_distillation_request("解释一下异丙醇和水的共沸")


def test_missing_parameters_asks_for_numbered_feed(tmp_path: Path) -> None:
    payload = run_ipa_extractive_export(
        "异丙醇-水萃取精馏，忘了给组成", export_dir=str(tmp_path)
    )
    assert payload.status == "missing_parameters"
    assert "feed_composition" in payload.missing_parameters


def test_awaiting_entrainer_lists_candidates(tmp_path: Path) -> None:
    payload = run_ipa_extractive_export(
        "异丙醇-水萃取精馏，异丙醇80% 水20%，1mol/s，常压",
        export_dir=str(tmp_path),
    )
    assert payload.status == "awaiting_entrainer"
    assert payload.entrainer_candidates, "no candidate was offered"
    names = {c.name for c in payload.entrainer_candidates}
    assert {"ethylene glycol", "glycerol"} & names


def test_run_extractive_export_dispatches_ipa(tmp_path: Path) -> None:
    """``run_extractive_export`` fans an IPA message into the IPA pipe."""
    payload = run_extractive_export(
        "2-丙醇-水萃取精馏，异丙醇80% 水20%，常压",
        export_dir=str(tmp_path),
    )
    assert payload.status == "awaiting_entrainer"
    assert payload.entrainer_candidates


def test_design_path_uses_unifac_and_reports(tmp_path: Path) -> None:
    captured: list[tuple[object, object]] = []

    def fake_design(spec: object, alpha_source: object = None) -> object:
        captured.append((spec, alpha_source))
        from thermo_engine.column_design import (
            design_generic_extractive_column as real_design,
        )
        return real_design(spec, alpha_source="unifac")

    def fake_export(*_: object, **__: object) -> Path:
        return Path(str(__["destination"]))  # type: ignore[index]

    with (
        mock.patch.object(
            extractive, "design_generic_extractive_column", side_effect=fake_design
        ),
        mock.patch.object(
            extractive, "export_generic_extractive_column", side_effect=fake_export
        ),
    ):
        payload = run_ipa_extractive_export(
            "异丙醇-水萃取精馏，异丙醇80% 水20%，常温，常压，用乙二醇",
            export_dir=str(tmp_path),
        )
    assert payload.status == "ready"
    assert captured, "generic design must be invoked"
    assert payload.design is not None
    assert payload.dwsim_file_uri is not None
    assert payload.dwsim_file_uri.startswith("/api/export/extractive/")
    # The typed feed is IPA/water (never the ethanol feed).
    assert payload.design.spec.feed_components == ["isopropanol", "water"]
    assert "异丙醇" in payload.message


# --- ThermoFormer / UNIFAC backend selection --------------------------------#
# Mirror the ethanol tests that patch the ML backend so this verifies the
# opt-in "use ThermoFormer instead of UNIFAC" marker threads alpha_source
# through the IPA picker and design seams without needing a real checkpoint.


class _FakeTFBackend:
    """Minimal bubble-point backend: returns temperature & a vapor split."""

    def bubble_point(self, request: object) -> object:  # noqa: ARG002
        xs = list(getattr(request.conditions, "liquid_composition", []))
        n = len(xs)

        class _Point:
            liquid_composition = xs
            vapor_composition = (
                [0.50, 0.10, 0.40] if n == 3 else [0.90, 0.10, 0.0][:n]
            )
            temperature_K = 360.0

        class _Result:
            points = [_Point()]

        return _Result()


def test_ipa_picker_threads_thermoformer() -> None:
    """'用ThermoFormer' on the IPA/water picker scores candidates with the ML
    backend (alpha_source == 'thermoformer') instead of UNIFAC."""
    from thermo_engine import column_design

    column_design._thermoformer_backend = None
    try:
        with mock.patch(
            "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeTFBackend
        ):
            payload = run_ipa_extractive_export(
                "异丙醇-水萃取精馏，用ThermoFormer，异丙醇80% 水20%，1mol/s，常压",
                export_dir="./_tf_test_one",
            )
    finally:
        column_design._thermoformer_backend = None
        import shutil

        shutil.rmtree("./_tf_test_one", ignore_errors=True)
    assert payload.status == "awaiting_entrainer"
    assert payload.alpha_source == "thermoformer"
    assert payload.entrainer_candidates


def test_ipa_design_runs_when_thermoformer_named() -> None:
    """With an entrainer committed and ThermoFormer requested, a valid design is
    produced (real deterministic engine logic, ML bubble mocked only)."""
    from thermo_engine import column_design

    column_design._thermoformer_backend = None
    try:
        with mock.patch(
            "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeTFBackend
        ):
            payload = run_ipa_extractive_export(
                "异丙醇-水萃取精馏，用ThermoFormer，异丙醇80% 水20%，常压，用乙二醇",
                export_dir="./_tf_test_two",
            )
    finally:
        column_design._thermoformer_backend = None
        import shutil

        shutil.rmtree("./_tf_test_two", ignore_errors=True)
    assert payload.status in {"ready", "dwsim_unavailable"}
    assert payload.design is not None
    assert payload.design.spec.feed_components == ["isopropanol", "water"]
    # ThermoFormer is reported through the design/message (top-level
    # ``alpha_source`` is only set on the awaiting_entrainer picker, matching the
    # ethanol path's payload contract).
    design_uses_ml = any("ThermoFormer" in a for a in payload.design.assumptions)
    assert design_uses_ml or "ThermoFormer is a predictive ML backend" in " ".join(
        payload.design.warnings
    )
    assert "ThermoFormer" in payload.message


def test_ipa_defaults_to_unifac_without_marker() -> None:
    """Without the marker the IPA/water picker stays on the UNIFAC default."""
    payload = run_ipa_extractive_export(
        "异丙醇-水萃取精馏，异丙醇80% 水20%，1mol/s，常压",
        export_dir="./_tf_test_zero",
    )
    import shutil

    shutil.rmtree("./_tf_test_zero", ignore_errors=True)
    assert payload.status == "awaiting_entrainer"
    assert payload.alpha_source is None  # UNIFAC default
    assert payload.entrainer_candidates
