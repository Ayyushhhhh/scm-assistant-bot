"""NLP guardrails and conversational layer for production chatbot.

Design decisions
────────────────
• Greeting detection uses a keyword set — no LLM call wasted on "hello".
• Off-topic detection uses a lightweight keyword heuristic first,
  falling back to the query classifier if uncertain.
• Input sanitisation strips prompt injection attempts.
• Response formatter adds markdown structure to raw answers for
  beautiful chat display.
"""

from __future__ import annotations

import re

from src.logger import get_logger

log = get_logger(__name__)

# ── Greeting patterns ────────────────────────────────────────────────────────
_GREETINGS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "what's up", "sup", "yo", "hola",
}

WELCOME_MESSAGE = (
    "Welcome to SCM Assistant.\n\n"
    "I am your supply chain governance assistant for BQBYTE Technologies. "
    "Feel free to ask me anything about supplier performance, policies, or compliance checks.\n\n"
    "How can I help you today?"
)

# ── Off-topic keywords that signal non-supply-chain questions ────────────────
_OFFTOPIC_PATTERNS = [
    r"\b(weather|recipe|joke|movie|sport|game|music|lyrics|poem)\b",
    r"\b(who is|tell me about yourself|what are you)\b",
    r"\b(write me|code|script|hack|crack)\b",
]

_OFFTOPIC_RESPONSE = (
    "I'm specifically trained for **BQBYTE Technologies' Supply Chain Governance**.\n\n"
    "I can answer questions about:\n"
    "- Supplier performance data (OTD rates, defect rates, compliance scores)\n"
    "- Governance policy rules (tier thresholds, penalties, audit schedules)\n"
    "- Risk assessment and disruption response procedures\n\n"
    "Please ask a supply-chain related question! 🏭"
)

# ── Prompt injection patterns ────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+",
    r"system\s*prompt",
    r"pretend\s+you\s+are",
    r"disregard",
    r"override",
    r"jailbreak",
]


def detect_greeting(query: str) -> bool:
    """Check if the query is a greeting (no LLM call needed)."""
    cleaned = query.lower().strip().rstrip("!?.")
    # Only exact match to avoid triggering on substrings like "Which" containing "hi"
    return cleaned in _GREETINGS


def detect_offtopic(query: str) -> bool:
    """Check if the query is clearly off-topic."""
    text = query.lower()
    for pattern in _OFFTOPIC_PATTERNS:
        if re.search(pattern, text):
            log.info("Off-topic query detected", extra={"query": query[:80]})
            return True
    return False


def sanitise_input(query: str) -> str:
    """Strip prompt injection attempts and clean the input."""
    cleaned = query.strip()

    # Check for injection attempts
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            log.warning("Prompt injection detected", extra={"query": cleaned[:80]})
            return ""  # empty string signals rejection

    # Basic length guard
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000]
        log.warning("Query truncated to 2000 chars")

    return cleaned


def format_response(answer: str) -> str:
    """Polish the raw LLM answer for chat display.

    Enhancements:
    - Ensure supplier lists are properly bulleted
    - Add emphasis to key numbers
    - Clean up any raw formatting artifacts
    """
    # Clean up common LLM artifacts
    answer = answer.replace("```", "").strip()

    # Ensure proper line breaks before lists
    answer = re.sub(r"(\d+\.)\s+", r"\n\1 ", answer)

    return answer
