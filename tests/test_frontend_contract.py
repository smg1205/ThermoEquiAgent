"""Guard the hand-maintained TypeScript contract against backend field drift."""

from pathlib import Path

from schemas.column_design import (
    BinaryCaseSpec,
    BinaryDistillationPayload,
    BinaryDistillationResult,
    ExtractiveExportPayload,
    LLEExportPayload,
    LLEExtractionSpec,
)
from schemas.domain import (
    CalculationResult,
    ModelCard,
    ModelRecommendation,
    RunListResponse,
    RunSummary,
    TaskManifest,
    ValidationReport,
)


def test_frontend_contract_declares_all_backend_fields() -> None:
    source = (Path(__file__).parents[1] / "apps" / "web" / "src" / "lib" / "types.ts").read_text(encoding="utf-8")
    for model in (
        TaskManifest,
        CalculationResult,
        ValidationReport,
        ModelRecommendation,
        RunSummary,
        RunListResponse,
        ModelCard,
    ):
        for field_name in model.model_fields:
            assert field_name in source, f"Frontend contract is missing {model.__name__}.{field_name}"


def test_frontend_contract_declares_all_dwsim_payload_fields() -> None:
    """The DWSIM export payloads are hand-mirrored in ``types.ts`` too.

    These are what the Web client reads to auto-download a rendered ``.dwxmz``
    (``dwsim_file_uri``) and to label it (``lle_kind``), so a backend field added
    without a matching TypeScript field silently loses functionality.
    """
    source = (Path(__file__).parents[1] / "apps" / "web" / "src" / "lib" / "types.ts").read_text(encoding="utf-8")
    for model in (
        LLEExportPayload,
        LLEExtractionSpec,
        BinaryCaseSpec,
        ExtractiveExportPayload,
        BinaryDistillationPayload,
        BinaryDistillationResult,
    ):
        for field_name in model.model_fields:
            assert field_name in source, f"Frontend contract is missing {model.__name__}.{field_name}"
