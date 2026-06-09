"""Jina Reranker wrapper — cross-encoder second-pass scoring.

Design decisions
────────────────
• Cross-encoder reranking is the single highest-impact accuracy boost
  in any RAG pipeline.  It reads (query, document) pairs jointly,
  producing much more accurate relevance scores than bi-encoder cosine.
• Jina chosen over Cohere because: 10M free tokens, no credit card,
  100 RPM — more than enough for this project's lifetime.
• Called via raw HTTP (httpx) to avoid dependency on a Jina SDK.
"""

from __future__ import annotations

from typing import Any

import httpx

from src import config
from src.logger import get_logger, timer

log = get_logger(__name__)


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = config.RERANK_TOP_K,
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Rerank documents against a query using Jina cross-encoder.

    Args:
        query: The user's original query.
        documents: List of dicts, each with a `content_key` field.
        top_k: Number of top results to return.
        content_key: Key containing the text to rerank.

    Returns:
        Top-k documents sorted by reranker relevance score.
    """
    if not documents:
        return []

    texts = [doc[content_key] for doc in documents]

    with timer() as t:
        resp = httpx.post(
            config.JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {config.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.JINA_RERANK_MODEL,
                "query": query,
                "documents": texts,
                "top_n": min(top_k, len(texts)),
            },
            timeout=30.0,
        )

    if resp.status_code != 200:
        log.warning(
            "Jina rerank failed, returning documents unranked",
            extra={"status": resp.status_code, "body": resp.text[:200]},
        )
        return documents[:top_k]

    data = resp.json()
    results_raw = data.get("results", [])

    reranked = []
    for item in results_raw:
        idx = item["index"]
        doc = documents[idx].copy()
        doc["rerank_score"] = item["relevance_score"]
        reranked.append(doc)

    log.info(
        "Jina rerank complete",
        extra={
            "input_docs": len(documents),
            "output_docs": len(reranked),
            "latency_ms": round(t["elapsed_ms"], 1),
            "top_score": round(reranked[0]["rerank_score"], 4) if reranked else 0,
        },
    )
    return reranked
