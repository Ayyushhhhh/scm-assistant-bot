"""Structured JSON logger used across the entire codebase.

Every log line is a single JSON object — easy to grep, parse, and ship
to any observability backend (ELK, Datadog, CloudWatch).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator


# ── Structured formatter ─────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emits each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "fn": record.funcName,
            "msg": record.getMessage(),
        }
        # Merge any extra fields attached via `logger.info("...", extra={...})`
        extra: dict = getattr(record, "extra", {})
        if extra:
            entry["extra"] = extra
        return json.dumps(entry, default=str)


# ── Logger factory ───────────────────────────────────────────────────────────

_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with JSON formatting (configured once)."""
    if name not in _CONFIGURED:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    return logging.getLogger(name)


# ── LLM call tracker ────────────────────────────────────────────────────────

@dataclass
class LLMMetrics:
    """Accumulates LLM usage stats across a single request lifecycle."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    breakdown: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, label: str, input_tokens: int, output_tokens: int,
               latency_ms: float) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        self.breakdown.append({
            "call": self.total_calls,
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
        })

    def summary(self) -> dict[str, Any]:
        return {
            "total_llm_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "breakdown": self.breakdown,
        }


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that measures wall-clock time in milliseconds."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000
