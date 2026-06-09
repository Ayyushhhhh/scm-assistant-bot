"""Gemini API wrapper with multi-key rotation and automatic failover.

Design decisions
────────────────
• Maintains a pool of API keys and rotates on 429 (quota exhausted).
• Each key gets its own configured SDK instance.
• Round-robin rotation ensures even quota distribution.
• Exponential backoff only on transient errors; key switch on quota errors.
• Every call records input/output tokens + latency into LLMMetrics.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any

import google.generativeai as genai

from src import config
from src.logger import LLMMetrics, get_logger, timer

log = get_logger(__name__)

# ── Key rotation pool ────────────────────────────────────────────────────────
_key_cycle = itertools.cycle(config.GEMINI_API_KEYS)
_current_key: str = next(_key_cycle) if config.GEMINI_API_KEYS else ""
genai.configure(api_key=_current_key)

_MAX_RETRIES = 8
_BACKOFF_BASE = 2.0
_RPM_LIMIT = 4  # stay under the 5 RPM project limit
_MIN_INTERVAL = 60.0 / _RPM_LIMIT  # ~15 seconds between calls
_last_call_time: float = 0.0


def _rate_limit_wait():
    """Ensure we don't exceed the per-minute rate limit."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _MIN_INTERVAL:
        wait = _MIN_INTERVAL - elapsed
        log.info(f"Rate limit: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_call_time = time.time()


def _rotate_key() -> str:
    """Switch to the next API key in the pool."""
    global _current_key
    _current_key = next(_key_cycle)
    genai.configure(api_key=_current_key)
    key_hint = _current_key[-6:]  # last 6 chars for logging (safe)
    log.info("Rotated to next API key", extra={"key_tail": key_hint})
    return _current_key


# ── Public API ───────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    *,
    system_instruction: str | None = None,
    json_mode: bool = False,
    metrics: LLMMetrics | None = None,
    label: str = "llm_call",
) -> str:
    """Call Gemini and return text. Rotates keys on quota errors."""
    model_kwargs: dict[str, Any] = {}
    if system_instruction:
        model_kwargs["system_instruction"] = system_instruction

    gen_config: dict[str, Any] = {}
    if json_mode:
        gen_config["response_mime_type"] = "application/json"

    for attempt in range(1, _MAX_RETRIES + 1):
        _rate_limit_wait()
        try:
            model = genai.GenerativeModel(config.GEMINI_MODEL, **model_kwargs)

            with timer() as t:
                response = model.generate_content(
                    prompt,
                    generation_config=gen_config if gen_config else None,
                )
            elapsed = t["elapsed_ms"]

            usage = getattr(response, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0

            if metrics:
                metrics.record(
                    label=label, input_tokens=in_tok,
                    output_tokens=out_tok, latency_ms=elapsed,
                )

            log.info(
                "Gemini call complete",
                extra={
                    "label": label, "model": config.GEMINI_MODEL,
                    "in_tok": in_tok, "out_tok": out_tok,
                    "ms": round(elapsed, 1), "attempt": attempt,
                },
            )
            return response.text

        except Exception as exc:
            err_str = str(exc)
            is_quota = "429" in err_str or "ResourceExhausted" in err_str
            is_not_found = "404" in err_str or "NotFound" in err_str

            if is_quota:
                _rotate_key()
                wait = 35.0  # server-suggested retry delay for RPM quota
            elif is_not_found:
                raise  # model doesn't exist, no point retrying
            else:
                wait = _BACKOFF_BASE * attempt

            log.warning(
                "Gemini call failed",
                extra={
                    "attempt": attempt, "error": err_str[:120],
                    "is_quota": is_quota, "wait": wait,
                },
            )
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(wait)

    raise RuntimeError("Exhausted retries")


def generate_json(
    prompt: str,
    *,
    system_instruction: str | None = None,
    metrics: LLMMetrics | None = None,
    label: str = "llm_json_call",
) -> dict | list:
    """Call Gemini in JSON mode and return parsed output."""
    raw = generate(
        prompt, system_instruction=system_instruction,
        json_mode=True, metrics=metrics, label=label,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())
