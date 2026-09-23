"""Structured domain exceptions."""

from schemas.domain import FailureDetail, FailureType


class ThermoEquiError(Exception):
    """Exception that retains the public failure taxonomy and recovery guidance."""

    def __init__(
        self,
        failure_type: FailureType,
        message: str,
        recovery_action: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = FailureDetail(
            failure_type=failure_type,
            message=message,
            recovery_action=recovery_action,
            details=details or {},
        )
