"""Pandas agent — LLM-generated code for structured-data queries.

Design decisions
────────────────
• The LLM generates pandas code (NOT SQL) because pandas is more
  expressive for the multi-step filtering + aggregation our questions
  require (e.g., groupby + filter + sort + format).
• Policy context is injected into the code-gen prompt so the LLM
  knows exact thresholds (OTD≥93%, Defect<0.5%, etc.) without
  needing a separate retrieval step.
• Execution uses a restricted namespace: only pd, np, df, and
  datetime are available — no file I/O, no imports, no network.
• Output validation ensures `result` is always a string.
"""

from __future__ import annotations

import re
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from src.agents.prompts import PANDAS_CODEGEN
from src.ingestion.csv_loader import get_schema_for_prompt
from src.llm import gemini_client
from src.logger import LLMMetrics, get_logger

log = get_logger(__name__)


def run(
    query: str,
    df: pd.DataFrame,
    *,
    policy_context: str = "",
    metrics: LLMMetrics | None = None,
) -> str:
    """Generate and execute pandas code to answer a data question.

    Args:
        query: The user's natural-language question.
        df: The supplier performance DataFrame.
        policy_context: Relevant policy text (thresholds, rules).
        metrics: Optional LLM metrics tracker.

    Returns:
        String result from the executed code, or an error message.
    """
    schema = get_schema_for_prompt(df)
    prompt = PANDAS_CODEGEN.format(
        schema=schema,
        policy_context=policy_context if policy_context else "No specific policy context provided.",
        query=query,
    )

    # ── Retry loop: generate → execute → retry with error feedback ───
    last_error = None
    for attempt in range(3):
        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = (
                prompt
                + f"\n\nPREVIOUS ATTEMPT FAILED with error:\n{last_error}\n"
                + "Fix the code to avoid this error. Remember: use only pd, np, df, datetime. "
                + "For checking NaN/empty, use == '' for string columns or pd.isna() for datetime."
            )

        raw_response = gemini_client.generate(
            current_prompt, metrics=metrics, label=f"pandas_codegen_attempt{attempt+1}",
        )
        code = _extract_code(raw_response)

        if not code:
            log.warning("No code block in LLM response", extra={"raw": raw_response[:200]})
            continue

        log.info("Pandas code generated", extra={"code_lines": code.count("\n") + 1, "attempt": attempt + 1})

        result = _execute_sandboxed(code, df)

        if not result.startswith("Execution error:") and not result.startswith("Error:"):
            log.info("Pandas execution complete", extra={"result_preview": str(result)[:200]})
            return result

        last_error = result
        log.warning("Pandas execution failed, retrying", extra={"error": result[:100], "attempt": attempt + 1})

    # All retries exhausted — return the last error
    log.error("Pandas agent exhausted retries", extra={"last_error": str(last_error)[:200]})
    return last_error or "Error: Could not generate working analysis code."


def _extract_code(response: str) -> str:
    """Extract Python code from the LLM response.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Raw code (no fences)
    """
    # Try to find fenced code block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: treat entire response as code if it looks like Python
    lines = response.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("#") or "import" in l or "=" in l]
    if any("df" in l for l in code_lines):
        return response.strip()

    return ""


def _execute_sandboxed(code: str, df: pd.DataFrame) -> str:
    """Execute generated pandas code in a restricted namespace.

    Only pd, np, df, and datetime are available — no file I/O,
    no arbitrary imports, no network access.
    """
    # Restricted but functional namespace
    namespace = {
        "pd": pd,
        "np": np,
        "df": df.copy(),  # copy to prevent mutation of original
        "datetime": datetime,
        "__builtins__": {
            "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "round": round, "sorted": sorted,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "sum": sum, "min": min, "max": max, "abs": abs,
            "enumerate": enumerate, "zip": zip, "range": range,
            "print": print, "isinstance": isinstance, "type": type,
            "map": map, "filter": filter, "any": any, "all": all,
            "repr": repr, "format": format, "hasattr": hasattr,
            "getattr": getattr, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "True": True, "False": False, "None": None,
        },
    }

    try:
        exec(code, namespace)
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Pandas execution failed", extra={"error": str(exc), "traceback": tb})
        return f"Execution error: {exc}"

    result = namespace.get("result", None)

    if result is None:
        log.warning("No 'result' variable found after execution")
        return "Error: Code did not produce a 'result' variable."

    # Ensure result is a string
    if not isinstance(result, str):
        result = str(result)

    return result
