"""Behavioral tests for the extractive-distillation chat integration."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import agent.extractive_distillation as extractive
from agent.extractive_distillation import (
    build_extractive_spec,
    extract_extractive_params,
    is_extractive_distillation_request,
    requests_thermoformer,
    run_extractive_export,
)
from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from schemas.domain import Intent


def test_intent_detection_matches_extractive_requests() -> None:
    hits = (
        "帮我做乙醇-水的萃取精馏模拟",
        "模拟乙醇水萃取精馏制取高纯乙醇",
        "萃取模拟，乙醇0.4水0.6，用乙二醇",
        "乙醇水萃取的dwsim模拟",
    )
    for message in hits:
        assert is_extractive_distillation_request(message), message

    non_hits = (
        "计算乙醇-水的气液平衡",
        "苯和甲苯的泡点是多少",
        "解释一下共沸的概念",
    )
    for message in non_hits:
        assert not is_extractive_distillation_request(message), message


def test_parameter_extraction_from_free_text() -> None:
    params = extract_extractive_params(
        "乙醇水萃取精馏，进料乙醇40%、水60%，1mol/s，25°C，常压，"
        "用乙二醇做萃取剂，塔顶乙醇纯度99.5%"
    )
    assert params["feed_composition"] == pytest.approx([0.4, 0.6])
    assert params["feed_flow_mol_s"] == pytest.approx(1.0)
    assert params["feed_temperature_K"] == pytest.approx(298.15)
    assert params["feed_pressure_kPa"] == pytest.approx(101.325)
    assert params["entrainer"] == "ethylene glycol"
    assert params["distillate_purity_mole_fraction"] == pytest.approx(0.995)


def test_parameter_extraction_does_not_confuse_purity_with_composition() -> None:
    params = extract_extractive_params("乙醇水萃取，乙醇40%水60%，塔顶纯度99.5%")
    # 40/60 is the feed; 99.5% is the distillate purity - never swapped.
    assert params["feed_composition"] == pytest.approx([0.4, 0.6])
    assert params["distillate_purity_mole_fraction"] == pytest.approx(0.995)


def test_entrainer_alias_resolution() -> None:
    assert extract_extractive_params("用甘油做萃取剂").get("entrainer") == "glycerol"
    assert extract_extractive_params("用乙二醇").get("entrainer") == "ethylene glycol"


def test_missing_parameters_returns_clarifying_payload(tmp_path: Path) -> None:
    payload = run_extractive_export("我想做萃取精馏模拟", export_dir=str(tmp_path))
    assert payload.status == "missing_parameters"
    assert "feed_composition" in payload.missing_parameters
    assert "entrainer" in payload.missing_parameters


def test_ready_path_writes_and_exposes_file(tmp_path: Path) -> None:
    def fake_export(design: object, destination: Path, *_: object, **__: object) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"FAKE-DWSIM-COLUMN")
        return destination

    with mock.patch.object(extractive, "export_dwsim_extractive_column", side_effect=fake_export):
        payload = run_extractive_export(
            "乙醇水萃取精馏，乙醇40%水60%，1mol/s，常压，乙二醇，塔顶纯度99.5%",
            export_dir=str(tmp_path),
        )
    assert payload.status == "ready"
    assert payload.design is not None
    assert payload.dwsim_file_uri is not None
    assert payload.dwsim_file_uri.startswith("/api/export/extractive/")
    stored = tmp_path / f"{payload.file_id}.dwxmz"
    assert stored.is_file()
    assert stored.read_bytes() == b"FAKE-DWSIM-COLUMN"
    # The ready message states the file is generated and flags the final manual
    # DWSIM step (condenser/reboiler duty) that this tool cannot automate.
    assert "DWSIM 文件已生成" in payload.message


def test_dwsim_unavailable_still_returns_design(tmp_path: Path) -> None:
    from thermo_engine.errors import FailureType, ThermoEquiError

    def raise_missing(*_: object, **__: object) -> None:
        raise ThermoEquiError(
            FailureType.MISSING_DATA,
            "DWSIM_HOME is not configured.",
            "Install DWSIM and set DWSIM_HOME.",
        )

    with mock.patch.object(extractive, "export_dwsim_extractive_column", side_effect=raise_missing):
        payload = run_extractive_export(
            "乙醇水萃取，乙醇40%水60%，乙二醇，塔顶纯度99.5%",
            export_dir=str(tmp_path),
        )
    assert payload.status == "dwsim_unavailable"
    assert payload.design is not None  # design survives even when DWSIM cannot render.
    assert payload.dwsim_file_uri is None


async def test_chat_routes_extractive_intent() -> None:
    orchestrator = ConversationOrchestrator(DeterministicProvider())
    response = await orchestrator.chat(
        "帮我做乙醇-水的萃取精馏模拟，用乙二醇",
        "conv-extractive-intent",
    )
    assert response.intent == Intent.EXTRACTIVE_DISTILLATION
    assert response.extractive is not None
    # No composition yet -> structured clarification.
    assert response.extractive.status == "missing_parameters"


async def test_chat_missing_parameters_asks_for_numbers() -> None:
    orchestrator = ConversationOrchestrator(DeterministicProvider())
    response = await orchestrator.chat(
        "我想进行萃取精馏模拟，但忘了给组成",
        "conv-extractive-missing",
    )
    assert response.extractive is not None
    assert response.extractive.status == "missing_parameters"
    assert "进料组成" in response.answer


@pytest.mark.parametrize(
    "message",
    [
        "用ThermoFormer算乙醇水萃取精馏",
        "萃取精馏用thermoformer后端",
        "extractive distillation with ThermoFormer",
    ],
)
def test_requests_thermoformer_detects_opt_in(message: str) -> None:
    assert requests_thermoformer(message)


def test_requests_thermoformer_is_opt_in() -> None:
    # A plain extractive request without the ThermoFormer marker stays UNIFAC.
    assert not requests_thermoformer("乙醇水萃取精馏，用乙二醇")


def _local_export_dir() -> str:
    """A throwaway export dir under the workspace (avoids pytest tmp teardown)."""
    import tempfile

    return tempfile.mkdtemp(prefix="tfchat_", dir=".")


def test_run_extractive_export_threads_thermoformer() -> None:
    """Requesting ThermoFormer forwards alpha_source='thermoformer' to the
    deterministic design without mutating process-global state."""
    import shutil

    captured: list[tuple[object, object]] = []
    export_dir = _local_export_dir()

    def fake_design(spec: object, alpha_source: object = None) -> object:
        captured.append((spec, alpha_source))
        # Delegate to the real deterministic design (UNIFAC) via the engine
        # module, not the patched name, to avoid self-recursion through the mock.
        from thermo_engine.column_design import design_extractive_distillation_column as real_design
        return real_design(spec, alpha_source="unifac")

    def fake_export(design: object, destination: Path, *_: object, **__: object) -> Path:
        # Returned without writing so the test never hits sandbox file-write
        # restrictions; the export outcome itself is not under test here.
        return destination

    try:
        with (
            mock.patch.object(
                extractive, "design_extractive_distillation_column", side_effect=fake_design
            ),
            mock.patch.object(
                extractive, "export_dwsim_extractive_column", side_effect=fake_export
            ),
        ):
            payload = run_extractive_export(
                "乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，常压，乙二醇，塔顶纯度99.5%",
                export_dir=export_dir,
            )
        assert payload.status == "ready"
        assert captured, "design function must be invoked"
        assert captured[0][1] == "thermoformer"
        assert "ThermoFormer" in payload.message
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


def test_run_extractive_export_defaults_to_unifac() -> None:
    import shutil

    captured: list[tuple[object, object]] = []
    export_dir = _local_export_dir()

    def fake_design(spec: object, alpha_source: object = None) -> object:
        captured.append((spec, alpha_source))
        from thermo_engine.column_design import design_extractive_distillation_column as real_design
        return real_design(spec, alpha_source="unifac")

    def fake_export(design: object, destination: Path, *_: object, **__: object) -> Path:
        return destination

    try:
        with (
            mock.patch.object(
                extractive, "design_extractive_distillation_column", side_effect=fake_design
            ),
            mock.patch.object(
                extractive, "export_dwsim_extractive_column", side_effect=fake_export
            ),
        ):
            payload = run_extractive_export(
                "乙醇水萃取精馏，乙醇40%水60%，1mol/s，常压，乙二醇，塔顶纯度99.5%",
                export_dir=export_dir,
            )
        assert payload.status == "ready"
        assert captured[0][1] is None  # UNIFAC default retained
        assert "ThermoFormer" not in payload.message
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


def test_no_entrainer_offers_local_picker() -> None:
    """Without an explicit entrainer the local model (UNIFAC default) returns a
    candidate picker (awaiting_entrainer) instead of going straight to design."""
    payload = run_extractive_export(
        "乙醇水萃取精馏，乙醇40%水60%，1mol/s，25°C，常压，塔顶纯度99.5%",
        export_dir=_local_export_dir(),
    )
    assert payload.status == "awaiting_entrainer"
    assert payload.alpha_source is None  # UNIFAC default
    names = {c.name for c in payload.entrainer_candidates}
    assert {"ethylene glycol", "glycerol"} <= names
    assert payload.design is None


def test_thermoformer_named_entrainer_picker_uses_thermoformer() -> None:
    """Naming ThermoFormer on the picker request scores candidates with the
    local ML backend (alpha_source == 'thermoformer')."""
    from unittest import mock as _mock

    from thermo_engine import column_design

    class _FakeBackend:
        def bubble_point(self, request: object) -> object:  # noqa: ARG002
            xs = list(request.conditions.liquid_composition)
            class _Point:
                liquid_composition = xs
                vapor_composition = [0.55, 0.40, 0.05] if len(xs) == 3 else [0.70, 0.30]
            class _Result:
                points = [_Point()]
            return _Result()

    column_design._thermoformer_backend = None
    try:
        with _mock.patch(
            "thermo_engine.thermoformer_backend.ThermoFormerBackend", _FakeBackend
        ):
            payload = run_extractive_export(
                "乙醇水萃取精馏，用ThermoFormer，乙醇40%水60%，1mol/s，25°C，常压，塔顶纯度99.5%",
                export_dir=_local_export_dir(),
            )
    finally:
        column_design._thermoformer_backend = None
    assert payload.status == "awaiting_entrainer"
    assert payload.alpha_source == "thermoformer"
    assert len(payload.entrainer_candidates) >= 2


def test_explicit_entrainer_bypasses_picker_into_design() -> None:
    """Once the user picks an entrainer (e.g. 用甘油) the design runs directly."""
    payload = run_extractive_export(
        "乙醇水萃取精馏，乙醇40%水60%，1mol/s，25°C，常压，用甘油，塔顶纯度99.5%",
        export_dir=_local_export_dir(),
    )
    assert payload.status in {"ready", "dwsim_unavailable"}
    assert payload.design is not None
    assert payload.design.spec.entrainer == "glycerol"


def test_ask_candidates_forces_picker_even_with_entrainer() -> None:
    """'看看候选' forces the picker even when an entrainer is already named."""
    payload = run_extractive_export(
        "乙醇水萃取精馏，乙醇40%水60%，用乙二醇，看看候选，塔顶纯度99.5%",
        export_dir=_local_export_dir(),
    )
    assert payload.status == "awaiting_entrainer"
    assert len(payload.entrainer_candidates) >= 2
