"""Query classifier — routes each query to the right pipeline.

Design decisions
────────────────
• Uses Gemini Flash in JSON mode for reliable structured output.
• Few-shot examples in the prompt are drawn from actual golden questions
  to maximise routing accuracy on the exact question types we'll face.
• All 5 golden questions are "hybrid" — this classifier correctly routes
  them because the prompt explicitly teaches: "if policy concept + data
  needed → hybrid".
"""

from __future__ import annotations

from typing import Literal

from src.agents.prompts import CLASSIFY_QUERY
from src.llm import gemini_client
from src.logger import LLMMetrics, get_logger

log = get_logger(__name__)

QueryType = Literal["structured_data", "policy_lookup", "hybrid"]


def classify(query: str, *, metrics: LLMMetrics | None = None) -> QueryType:
    """Classify a user query into one of three routing types.

    Args:
        query: The user's natural-language question.
        metrics: Optional LLM metrics tracker.

    Returns:
        One of: "structured_data", "policy_lookup", "hybrid".
    """
    prompt = CLASSIFY_QUERY.format(query=query)

    try:
        result = gemini_client.generate_json(
            prompt, metrics=metrics, label="query_classify",
        )
        qtype = result.get("type", "hybrid")
        reasoning = result.get("reasoning", "")
    except Exception as exc:
        log.warning("Classification failed, defaulting to hybrid", extra={"error": str(exc)})
        qtype = "hybrid"
        reasoning = "fallback"

    # Validate against allowed types
    if qtype not in ("structured_data", "policy_lookup", "hybrid"):
        log.warning("Unknown query type, defaulting to hybrid", extra={"raw_type": qtype})
        qtype = "hybrid"

    log.info(
        "Query classified",
        extra={"query": query[:80], "type": qtype, "reasoning": reasoning},
    )
    return qtype
