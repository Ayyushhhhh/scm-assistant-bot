"""Reciprocal Rank Fusion — merges results from multiple retrievers.

Design decisions
────────────────
• RRF is superior to simple score averaging because different retrievers
  (semantic, BM25, fuzzy) use incomparable scoring scales.  RRF normalises
  by rank position, making fusion fair.
• Formula: RRF(d) = Σ 1 / (k + rank_i(d))  for each retriever i.
• k = 60 is the standard constant from the original Cormack et al. paper.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src import config
from src.logger import get_logger

log = get_logger(__name__)


def reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = config.RRF_K,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using RRF.

    Args:
        *result_lists: One or more lists of dicts. Each dict must contain
            `id_key` (for deduplication) and `content`.
        k: RRF constant (default 60).
        id_key: Key used to identify unique documents across lists.

    Returns:
        Merged list sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_lookup: dict[str, dict[str, Any]] = {}

    for retriever_idx, results in enumerate(result_lists):
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get(id_key, doc.get("content", "")[:80])
            rrf_scores[doc_id] += 1.0 / (k + rank)

            # Keep the first occurrence of each document
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused = []
    for doc_id in sorted_ids:
        doc = doc_lookup[doc_id].copy()
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        fused.append(doc)

    log.info(
        "RRF fusion complete",
        extra={
            "input_lists": len(result_lists),
            "input_total": sum(len(r) for r in result_lists),
            "output": len(fused),
        },
    )
    return fused
