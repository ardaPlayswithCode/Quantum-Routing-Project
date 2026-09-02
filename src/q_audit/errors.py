"""the errors we raise on purpose, plus what exit code each one means.

exit codes: 0 ok, 2 you did something wrong, 3 calibration problem,
4 transpiler/qiskit problem
"""


class QAuditError(Exception):
    exit_code = 2
    kind = "q_audit_error"

    def __init__(self, message, *, hint=None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self):
        return {"kind": self.kind, "message": self.message, "hint": self.hint}


class CircuitLoadError(QAuditError):
    exit_code = 2
    kind = "circuit_load_error"


class UnsupportedCircuitError(QAuditError):
    """circuit has something we can't audit, 3q gates or control flow"""

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
    """the vf2 error map hook stopped behaving, probably a qiskit upgrade"""

    exit_code = 4
    kind = "injection_contract_error"
