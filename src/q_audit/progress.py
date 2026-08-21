"""Progress events as NDJSON on **stderr**.

stdout is reserved for the report (and for exactly one JSON document when
``--json`` is passed).  Anything chatty that lands on stdout breaks
``q-audit run ... --json | jq``, so progress goes to stderr, one JSON object
per line, machine-readable for CI.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class ProgressReporter:
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    enabled: bool = True
    quiet: bool = False
    _t0: float = field(default_factory=time.monotonic)

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled or self.quiet:
            return
        payload = {
            "ts": round(time.monotonic() - self._t0, 4),
            "event": event,
            **fields,
        }
        try:
            self.stream.write(json.dumps(payload, default=str) + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def step(self, name: str, **fields: Any) -> None:
        self.emit("step", step=name, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self.emit("warning", message=message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.emit("error", message=message, **fields)


NULL_REPORTER = ProgressReporter(enabled=False)
