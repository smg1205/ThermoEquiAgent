"""Behavioral tests for the binary-VLE DWSIM export path.

These cover the report §1.5-1 flow: a two-component VLE system (canonically
2-propanol / water) that the user asks to render as a downloadable DWSIM
``.dwxmz``.  The tests assert at the public seam -- request classification,
``run_binary_distillation`` and the orchestrator ``chat`` entry -- and mock only
the DWSIM Automation call, which cannot load in a sandboxed process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

import agent.extractive_distillation as extractive
from agent.extractive_distillation import (
    evidence_for_binary_distillation,
    is_binary_vle_dwsim_request,
    wants_dwsim_file,
)
from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from schemas.domain import Intent


def test_wants_dwsim_file_markers() -> None:
    assert wants_dwsim_file("导出 dwsim 文件")
    assert wants_dwsim_file("download the .dwxmz")
    assert wants_dwsim_file("DWISM export please")
    # "vle" names the physics, not the deliverable.
    assert not wants_dwsim_file("苯-甲苯的 VLE 曲线")


def test_is_binary_vle_dwsim_request_hits() -> None:
    hits = (
        "2-丙醇-水二元VLE精馏，导出dwsim",
        "isopropanol-water binary VLE, download the dwsim file",
        "2-propanol / water binary VLE distillation column design, export dwsim",
        "甲醇-水精馏塔设计，导出 dwsim",
    )
    for message in hits:
        assert is_binary_vle_dwsim_request(message), message


def test_is_binary_vle_dwsim_request_requires_both_halves() -> None:
    # A binary pair with no DWSIM marker is a design-only request.
    assert not is_binary_vle_dwsim_request("甲醇-水精馏塔设计")
    # A DWSIM marker with no recognised binary pair is not this flow.
    assert not is_binary_vle_dwsim_request("导出 dwsim 文件")
    assert not is_binary_vle_dwsim_request("苯-甲苯的 VLE 曲线，导出 dwsim")


def test_extractive_and_lle_take_precedence() -> None:
    # Overlapping extractive wording must not fall through to the plain-binary path.
    assert not is_binary_vle_dwsim_request("乙醇-水萃取精馏，导出 dwsim")
    assert not is_binary_vle_dwsim_request("乙醇/乙酸乙酯/水液液萃取，导出 dwsim")


# --------------------------------------------------------------------------- #
# Feed-composition parsing (report §1.5-1 notation)
# --------------------------------------------------------------------------- #
#: A single reported mole fraction fully determines a binary feed: the other
#: component is ``1 - x``.
def test_paired_feed_notation_is_parsed() -> None:
    """The explicit pair form already works and must keep working."""
    assert extractive._find_feed_mole_fractions("异丙醇0.3 水0.7") == pytest.approx([0.3, 0.7])
    assert extractive._find_feed_mole_fractions("乙醇40% 水60%") == pytest.approx([0.4, 0.6])


def test_absent_feed_is_reported_not_invented() -> None:
    """With no fraction in the message the feed must be ``None``.

    Silently defaulting to 50/50 would design a column at the wrong composition
    and still report ``ready``.
    """
    assert extractive._find_feed_mole_fractions("甲醇-水精馏塔") is None
    payload = extractive.run_binary_distillation("甲醇-水精馏塔")
    assert payload.status == "missing_parameters"
    assert payload.missing_parameters == ["feed_composition"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # The report's own notation: a single named light-key fraction.
        ("2-丙醇-水精馏塔，x_IPA=0.3，导出 dwsim", [0.3, 0.7]),
        # A named light key written in Chinese.
        ("2-丙醇-水精馏塔，异丙醇摩尔分数0.3，导出 dwsim", [0.3, 0.7]),
        ("甲醇-水精馏塔，甲醇0.3，导出dwsim", [0.3, 0.7]),
        # Bare "x=" carries no component name; the light key is the convention.
        ("2-丙醇-水精馏塔，x=0.3，导出 dwsim", [0.3, 0.7]),
    ],
)
def test_single_fraction_notation_implies_the_other_component(
    message: str, expected: list[float]
) -> None:
    """A single mole fraction must yield ``[x, 1 - x]``, not ``missing_parameters``.

    Report §1.5-1 is written entirely as ``x_IPA=0.3``; rejecting that notation
    locks the standard input out of the flow.
    """
    fractions = extractive._find_feed_mole_fractions(message)
    assert fractions == pytest.approx(expected), message


def test_single_fraction_named_heavy_is_not_silently_inverted() -> None:
    """``水0.7`` means x_water=0.7, i.e. feed ``[0.3, 0.7]`` -- light first.

    This is the case that makes a naive "the first number is the light key" rule
    unsafe: it would return ``[0.7, 0.3]`` and swap the overhead and bottoms
    products without any visible error.
    """
    fractions = extractive._find_feed_mole_fractions("甲醇-水精馏塔，水0.7，导出dwsim")
    assert fractions == pytest.approx([0.3, 0.7])


def test_feed_parsing_does_not_swallow_temperature_or_pressure() -> None:
    """"80C" and "1 atm" are conditions, not mole fractions."""
    assert extractive._find_feed_mole_fractions("2-丙醇-水精馏塔，导出dwsim，操作温度 80C") is None
    fractions = extractive._find_feed_mole_fractions("2-丙醇-水精馏塔，x=0.3，1 atm，导出dwsim")
    assert fractions == pytest.approx([0.3, 0.7])


def _fake_export(destination: Path, *_: object, **__: object) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"FAKE-DWSIM-BINARY")
    return destination


def _local_export_dir() -> Path:
    """A throwaway export store under the workspace.

    Deliberately avoids pytest's ``tmp_path`` machinery: its teardown re-chmods
    the directory, which a confined file sandbox refuses, and that failure would
    be indistinguishable from a real test failure.  The directory is created with
    ``os.makedirs`` rather than ``tempfile.mkdtemp`` for the same reason -- the
    sandbox grants access to paths created by the workspace process itself.
    """
    directory = Path(".tmp-tfbinexport") / uuid4().hex[:8]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_ready_path_writes_and_exposes_file() -> None:
    """A resolved binary VLE + DWSIM request returns a ready, downloadable file."""
    export_store = _local_export_dir()
    with (
        mock.patch.object(extractive, "export_dwsim_binary_column", side_effect=_fake_export),
        mock.patch.object(extractive, "export_directory", return_value=export_store),
    ):
        payload = extractive.run_binary_distillation(
            "2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim"
        )
    assert payload.status == "ready"
    assert payload.dwsim_file_uri is not None
    assert payload.dwsim_file_uri.startswith("/api/export/extractive/")
    assert payload.file_id is not None
    stored = export_store / f"{payload.file_id}.dwxmz"
    assert stored.is_file()
    assert stored.read_bytes() == b"FAKE-DWSIM-BINARY"


def test_report_operating_point_reproduces_section_1_5_1() -> None:
    """The x_IPA=0.3 UNIFAC column must reproduce report §1.5-1 (alpha 2.713, N 19, R 2.694).

    The feed tray and product split come from the requested exported operating
    point (feed on the tray above the reboiler; bottoms = half the feed), not from
    the report's values, so only the short-cut design numbers are asserted here.
    """
    with (
        mock.patch.object(extractive, "export_dwsim_binary_column", side_effect=_fake_export),
        mock.patch.object(extractive, "export_directory", return_value=Path(".")),
    ):
        payload = extractive.run_binary_distillation(
            "2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim"
        )
    assert payload.result is not None
    result = payload.result
    assert result.components == ["isopropanol", "water"]
    assert result.feed_composition == pytest.approx([0.3, 0.7])
    assert result.feed_flow_mol_s == pytest.approx(1.0)
    assert result.distillate_purity_mole_fraction == pytest.approx(0.995)
    assert result.relative_volatility == pytest.approx(2.713, abs=1e-3)
    assert result.theoretical_stages == 19
    assert result.minimum_stages == pytest.approx(10.069, abs=1e-3)
    assert result.reflux_ratio == pytest.approx(2.694, abs=1e-3)
    assert result.minimum_reflux_ratio == pytest.approx(1.924, abs=1e-3)
    # Requested export operating point: feed on the tray above the reboiler.
    assert result.feed_stage == 18
    assert result.condenser_temperature_K == pytest.approx(355.26, abs=0.01)
    assert result.reboiler_temperature_K == pytest.approx(367.46, abs=0.01)
    # The feed stream sits on the feed bubble point, as the DWSIM export requires.
    assert payload.feed_temperature_K == pytest.approx(354.60, abs=0.05)
    assert result.alpha_source == "unifac"


def test_exported_product_split_is_half_of_the_feed() -> None:
    """Bottoms take half the feed and the distillate the remainder.

    These are the two rigorous-column specifications written into the ``.dwxmz``;
    the payload must report the same numbers the file carries, so the chat answer
    and the exported tower cannot drift apart.
    """
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path("unused.dwxmz")

    with (
        mock.patch.object(extractive, "export_dwsim_binary_column", side_effect=capture),
        mock.patch.object(extractive, "export_directory", return_value=Path(".")),
    ):
        payload = extractive.run_binary_distillation(
            "2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim"
        )

    assert payload.result is not None
    # Bottoms = feed / 2, distillate = the rest; the two must sum to the feed.
    assert payload.result.bottoms_flow_mol_s == pytest.approx(0.5)
    assert payload.result.distillate_flow_mol_s == pytest.approx(0.5)
    assert payload.result.distillate_flow_mol_s + payload.result.bottoms_flow_mol_s == pytest.approx(1.0)

    # The exporter must receive the same split and the requested feed tray.
    assert captured["bottoms_flow_mol_s"] == pytest.approx(0.5)
    assert captured["distillate_flow_mol_s"] == pytest.approx(0.5)
    assert captured["feed_stage"] == payload.result.feed_stage == 18


def test_exported_feed_tray_is_the_one_above_the_reboiler() -> None:
    """The exporter converts the 1-based feed_stage to DWSIM's 0-based index.

    ``stages - 1`` is the tray above the reboiler, so the file must bind the feed
    to index ``stages - 2`` = 17 (Stage17) in a 19-entry stage list.
    """
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> Path:
        captured.update(kwargs)
        return Path("unused.dwxmz")

    with (
        mock.patch.object(extractive, "export_dwsim_binary_column", side_effect=capture),
        mock.patch.object(extractive, "export_directory", return_value=Path(".")),
    ):
        extractive.run_binary_distillation("2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim")

    stages = int(captured["stages"])
    assert stages == 19
    assert captured["feed_stage"] == stages - 1
    # DWSIM's 0-based stage index the exporter writes:
    assert int(captured["feed_stage"]) - 1 == 17


def test_dwsim_unavailable_still_returns_the_design() -> None:
    """When DWSIM cannot render, the design numbers must survive."""

    def raise_unavailable(*_: object, **__: object) -> None:
        raise RuntimeError("Failed to initialize Python.Runtime.dll")

    with (
        mock.patch.object(extractive, "export_dwsim_binary_column", side_effect=raise_unavailable),
        mock.patch.object(extractive, "export_directory", return_value=Path(".")),
    ):
        payload = extractive.run_binary_distillation(
            "2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim"
        )
    assert payload.status == "dwsim_unavailable"
    assert payload.result is not None
    assert payload.result.theoretical_stages == 19
    assert payload.dwsim_file_uri is None
    assert payload.file_id is None


def test_missing_dwsim_mapping_is_reported_not_guessed() -> None:
    payload = extractive.run_binary_distillation("苯-甲苯精馏塔，导出 dwsim")
    # Benzene/toluene is not a supported distillation pair, so no design is run.
    assert payload.status == "missing_parameters"
    assert payload.result is None


def test_evidence_states_deterministic_design_and_dwsim_status() -> None:
    ready = extractive.BinaryDistillationPayload(status="ready", message="")
    categories = {statement.category for statement in evidence_for_binary_distillation(ready)}
    assert "Calculation" in categories

    unavailable = extractive.BinaryDistillationPayload(status="dwsim_unavailable", message="")
    text = " ".join(s.text for s in evidence_for_binary_distillation(unavailable))
    assert "未安装" in text or "未配置" in text

    thermoformer = extractive.BinaryDistillationPayload(
        status="ready", message="", alpha_source="thermoformer"
    )
    assert any(s.category == "Estimate" for s in evidence_for_binary_distillation(thermoformer))


async def _chat(message: str, conversation_id: str):  # pragma: no cover - helper
    orchestrator = ConversationOrchestrator(DeterministicProvider())
    return await orchestrator.chat(message, conversation_id)


def test_chat_routes_binary_vle_dwsim_intent() -> None:
    """End-to-end: the chat entry classifies and carries the distillation payload."""
    response = asyncio.run(
        _chat("2-丙醇-水二元VLE精馏塔，x=0.3/0.7，导出 dwsim", "conv-binary-vle-dwsim")
    )
    assert response.intent == Intent.DISTILLATION_DESIGN
    assert response.distillation is not None
    # DWSIM may be unavailable in this environment; the design must still be present.
    assert response.distillation.status in {"ready", "dwsim_unavailable"}
    assert response.distillation.result is not None
    assert response.distillation.result.components == ["isopropanol", "water"]
    assert response.statements


def test_chat_binary_design_without_dwsim_marker_still_works() -> None:
    """A design-only request keeps routing to DISTILLATION_DESIGN."""
    response = asyncio.run(_chat("甲醇-水精馏塔，甲醇50%、水50%", "conv-binary-plain"))
    assert response.intent == Intent.DISTILLATION_DESIGN
    assert response.distillation is not None
    assert response.distillation.result is not None
    assert response.distillation.result.components[0] == "methanol"
