"""PageIndex — Vectorless, reasoning-first retrieval.

Core idea
─────────
Instead of embedding + similarity search, the LLM "reads" a hierarchical
table-of-contents and reasons about which sections are relevant.  It then
extracts the raw text from those sections.  Zero vectors involved.

Pipeline stages
───────────────
1. OFFLINE: Parse policy → hierarchical JSON tree (sections → sub-sections).
2. RUNTIME: LLM reads tree summaries → selects relevant section IDs.
3. RUNTIME: Extract raw text from selected sections.

Why this works for our use case
──────────────────────────────
The governance policy is structured by design (§1–§10 with clear titles).
Section titles alone are sufficient for the LLM to route accurately.
The policy is small (~5K tokens), so the tree fits in a single LLM context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ingestion.pdf_parser import PolicySection
from src.llm import gemini_client
from src.logger import LLMMetrics, get_logger

log = get_logger(__name__)


# ── Tree construction (offline) ──────────────────────────────────────────────

def build_tree(sections: list[PolicySection]) -> dict[str, Any]:
    """Build a hierarchical JSON tree from flat PolicySection list.

    Structure:
    {
      "document": "...",
      "sections": [
        {
          "id": "5",
          "title": "Risk Assessment & Escalation Protocol",
          "summary": "...",
          "children": [
            {"id": "5.1", "title": "Risk Level Definitions", "summary": "...", "content": "..."},
            ...
          ]
        }
      ]
    }
    """
    # Separate top-level sections from sub-sections
    top_level: dict[str, dict] = {}
    sub_sections: list[PolicySection] = []

    for sec in sections:
        if sec.parent_id == "":
            # Top-level section (e.g. "5")
            top_level[sec.section_id] = {
                "id": sec.section_id,
                "title": sec.title,
                "summary": _summarise(sec.content),
                "content": sec.content,
                "children": [],
            }
        else:
            sub_sections.append(sec)

    # Attach sub-sections to their parents
    for sub in sub_sections:
        parent = top_level.get(sub.parent_id)
        if parent:
            parent["children"].append({
                "id": sub.section_id,
                "title": sub.title,
                "summary": _summarise(sub.content),
                "content": sub.content,
            })
        else:
            # Orphan sub-section — treat as top-level
            top_level[sub.section_id] = {
                "id": sub.section_id,
                "title": sub.title,
                "summary": _summarise(sub.content),
                "content": sub.content,
                "children": [],
            }

    tree = {
        "document": "SupplyChain_Governance_Policy_v3.2",
        "sections": list(top_level.values()),
    }

    log.info(
        "PageIndex tree built",
        extra={
            "top_sections": len(top_level),
            "sub_sections": len(sub_sections),
        },
    )
    return tree


def _summarise(text: str, max_words: int = 60) -> str:
    """Create a concise summary (first N words) for tree navigation."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


# ── Persistence ──────────────────────────────────────────────────────────────

def save_tree(tree: dict, path: str | Path) -> None:
    """Save the tree to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    log.info("PageIndex tree saved", extra={"path": str(p)})


def load_tree(path: str | Path) -> dict:
    """Load a previously saved tree."""
    with open(str(path), "r", encoding="utf-8") as f:
        tree = json.load(f)
    log.info("PageIndex tree loaded", extra={"path": str(path)})
    return tree


# ── Reasoning-based navigation (runtime) ────────────────────────────────────

def navigate(
    query: str,
    tree: dict,
    *,
    metrics: LLMMetrics | None = None,
) -> list[dict[str, Any]]:
    """Use LLM reasoning to navigate the tree and extract relevant sections.

    Args:
        query: User's natural-language query.
        tree: The hierarchical document tree.
        metrics: Optional LLM metrics tracker.

    Returns:
        List of dicts with 'section_id', 'title', and 'content'.
    """
    from src.agents.prompts import PAGEINDEX_NAVIGATE

    # Build a concise tree summary for the LLM (titles + summaries only)
    tree_lines = []
    for sec in tree["sections"]:
        tree_lines.append(f"§{sec['id']} {sec['title']}: {sec['summary']}")
        for child in sec.get("children", []):
            tree_lines.append(f"  §{child['id']} {child['title']}: {child['summary']}")

    tree_summary = "\n".join(tree_lines)

    # Step 1: LLM selects relevant section IDs
    prompt = PAGEINDEX_NAVIGATE.format(tree_summary=tree_summary, query=query)
    try:
        selected_ids = gemini_client.generate_json(
            prompt, metrics=metrics, label="pageindex_navigate",
        )
    except Exception as exc:
        log.warning("PageIndex navigation failed, returning all sections", extra={"error": str(exc)})
        all_sections = _flatten_tree(tree)
        return [
            {"chunk_id": f"pageindex-{sid}", "section_id": sid,
             "title": s["title"], "content": s["content"], "source": "pageindex"}
            for sid, s in all_sections.items()
        ]

    if not isinstance(selected_ids, list):
        selected_ids = [selected_ids]

    log.info("PageIndex navigation", extra={"query": query[:80], "selected": selected_ids})

    # Step 2: Extract content from selected sections
    results = []
    all_sections = _flatten_tree(tree)

    for sid in selected_ids:
        sid_str = str(sid)
        if sid_str in all_sections:
            sec = all_sections[sid_str]
            results.append({
                "chunk_id": f"pageindex-{sid_str}",
                "section_id": sid_str,
                "title": sec["title"],
                "content": sec["content"],
                "source": "pageindex",
            })

    log.info("PageIndex extraction", extra={"sections_found": len(results)})
    return results


def _flatten_tree(tree: dict) -> dict[str, dict]:
    """Flatten the tree into {section_id: section_dict}."""
    flat: dict[str, dict] = {}
    for sec in tree["sections"]:
        flat[sec["id"]] = sec
        for child in sec.get("children", []):
            flat[child["id"]] = child
    return flat
