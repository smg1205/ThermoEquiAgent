"""API-level tests for the binary LLE DWSIM deliverable.

The binary-LLE path was previously exercised only through
``ConversationOrchestrator.chat``.  These tests cover the seam the Web client
actually uses: ``POST /api/chat`` must carry the ``lle_extraction`` payload with
a downloadable ``dwsim_file_uri``, and that URI must serve the rendered file.

Only the DWSIM *Automation* call is faked (it needs a local DWSIM install);
routing, payload shaping, and the download endpoint are all real.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import agent.extractive_distillation as extractive
from apps.api.main import app

_BUTANOL_WATER = (
    "\u6b63\u4e01\u9187-\u6c34\u4e8c\u5143\u6db2\u6db2\u8403\u53d6\uff0c0.3/0.7\uff0c298.15 K\uff0c\u5bfc\u51fa dwsim \u6587\u4ef6"
)


def _fake_export(**kwargs: object) -> Path:
    """Write a stand-in .dwxmz and report a real two-liquid split."""
    destination = Path(kwargs["destination"])  # type: ignore[arg-type]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"FAKE-BINARY-LLE-DWXMZ")
    split_out = kwargs.get("split_out")
    if isinstance(split_out, dict):
        split_out.update(
            {
                "separated": True,
                "light_flow_mol_s": 0.748107,
                "heavy_flow_mol_s": 0.251893,
                "light_composition": [0.399151, 0.600849],
                "heavy_composition": [0.005528, 0.994472],
            }
        )
    return destination


@pytest.fixture
def export_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway export store wired up the way production resolves it.

    ``apps.api.main`` binds ``export_directory`` at import time, so patching the
    agent module would not affect the download route.  Pointing the real
    environment variable at a temp dir exercises both the writer and the reader
    through the same production code path.
    """
    directory = tmp_path_factory.mktemp("binary_lle_api")
    monkeypatch.setenv("EXTRACTIVE_EXPORT_DIR", str(directory))
    return directory


def test_chat_returns_a_downloadable_binary_lle_file(export_store: Path) -> None:
    client = TestClient(app)
    with mock.patch.object(extractive, "export_dwsim_binary_lle_flowsheet", side_effect=_fake_export):
        response = client.post(
            "/api/chat",
            json={"message": _BUTANOL_WATER, "conversation_id": "api-binary-lle"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "LLE_EXTRACTION"

    lle = body["lle_extraction"]
    assert lle["status"] == "ready"
    assert lle["lle_kind"] == "binary"
    assert lle["dwsim_file_uri"] is not None
    assert lle["dwsim_file_uri"].startswith("/api/export/extractive/")
    # The binary path is described by binary_spec, not the ternary spec.
    assert lle["spec"] is None
    assert lle["binary_spec"]["kind"] == "lle"
    assert lle["binary_spec"]["mode"] == "two_liquid_vessel"
    assert lle["binary_spec"]["components"] == ["1-Butanol", "Water"]


def test_download_endpoint_serves_the_binary_lle_file(export_store: Path) -> None:
    """The URI handed to the Web client must actually resolve to the file."""
    client = TestClient(app)
    with mock.patch.object(extractive, "export_dwsim_binary_lle_flowsheet", side_effect=_fake_export):
        body = client.post(
            "/api/chat",
            json={"message": _BUTANOL_WATER, "conversation_id": "api-binary-lle-dl"},
        ).json()

        uri = body["lle_extraction"]["dwsim_file_uri"]
        download = client.get(uri)

    assert download.status_code == 200
    assert download.content == b"FAKE-BINARY-LLE-DWXMZ"
    assert download.headers["content-type"] == "application/octet-stream"
    assert body["lle_extraction"]["file_id"] in download.headers["content-disposition"]


def test_binary_lle_message_reports_the_heavy_phase(export_store: Path) -> None:
    """The chat answer must state DWSIM's heavy-phase flow, not just claim success."""
    client = TestClient(app)
    with mock.patch.object(extractive, "export_dwsim_binary_lle_flowsheet", side_effect=_fake_export):
        body = client.post(
            "/api/chat",
            json={"message": _BUTANOL_WATER, "conversation_id": "api-binary-lle-msg"},
        ).json()

    answer = body["answer"]
    assert "Heavy_Liquid" in answer
    assert "0.25189" in answer


def test_download_endpoint_rejects_traversal_for_lle(export_store: Path) -> None:
    """The LLE download route must not escape the export store."""
    client = TestClient(app)
    for probe in ("..%2F..%2F.env", "..%2Fpyproject.toml"):
        assert client.get(f"/api/export/extractive/{probe}").status_code == 404