"""Typed failure modes and their process exit codes.

Exit codes are part of the CLI contract:
    0  success
    2  user error (bad circuit file, bad flag, unusable input)
    3  calibration error (no source, stale snapshot without --allow-stale)
    4  transpiler/compiler error (no layout found, API drift, injection failure)
"""

from __future__ import annotations


class QAuditError(Exception):
    """Base class for every error q-audit raises deliberately."""

    exit_code: int = 2
    kind: str = "q_audit_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "hint": self.hint}


class CircuitLoadError(QAuditError):
    exit_code = 2
    kind = "circuit_load_error"


class UnsupportedCircuitError(QAuditError):
    """Circuit contains something the MVP cannot audit (>2q gates, control flow)."""

    exit_code = 2
    kind = "unsupported_circuit"


class CalibrationError(QAuditError):
    exit_code = 3
    kind = "calibration_error"


class StaleCalibrationError(CalibrationError):
    exit_code = 3
    kind = "stale_calibration"


class TranspileAuditError(QAuditError):
    exit_code = 4
    kind = "transpile_error"


class InjectionContractError(TranspileAuditError):
    """The private VF2 error-map hook no longer behaves as expected (Qiskit drift)."""

    exit_code = 4
    kind = "injection_contract_error"
