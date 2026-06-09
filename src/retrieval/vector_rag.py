"""Vector RAG pipeline — 5-layer hybrid retrieval.

Pipeline stages
───────────────
1. Semantic search  (FAISS + Gemini embeddings)  → conceptual matches
2. BM25 keyword     (rank_bm25)                  → exact term matches
3. Fuzzy entity     (thefuzz)                     → typo-tolerant entity matches
4. RRF fusion       (reciprocal rank fusion)      → fair rank-based merging
5. Jina reranker    (cross-encoder)               → precision scoring

This five-layer approach ensures we never miss context due to any single
retrieval method's blind spots.
"""

from __future__ import annotations

from typing import Any

from src import config
from src.logger import LLMMetrics, get_logger
from src.retrieval.reranker import rerank
from src.retrieval.rrf import reciprocal_rank_fusion
from src.vectorstore.bm25_index import BM25Index
from src.vectorstore.embeddings import embed_query
from src.vectorstore.faiss_store import FAISSStore

log = get_logger(__name__)


def retrieve(
    query: str,
    *,
    faiss_store: FAISSStore,
    bm25_index: BM25Index,
    metrics: LLMMetrics | None = None,
) -> list[dict[str, Any]]:
    """Run the full 5-layer retrieval pipeline.

    Args:
        query: User's natural-language query.
        faiss_store: The FAISS vector store.
        bm25_index: The BM25 keyword index.
        metrics: Optional LLM metrics tracker.

    Returns:
        Top-k reranked policy chunks, ready for LLM synthesis.
    """
    # ── Stage 1: Semantic search ─────────────────────────────────────────
    query_vec = embed_query(query)
    semantic_results = faiss_store.search(query_vec, top_k=config.SEMANTIC_TOP_K)
    log.info("Stage 1: Semantic search", extra={"results": len(semantic_results)})

    # ── Stage 2: BM25 keyword search ─────────────────────────────────────
    bm25_results = bm25_index.search(query, top_k=config.BM25_TOP_K)
    log.info("Stage 2: BM25 search", extra={"results": len(bm25_results)})

    # ── Stage 3: (Fuzzy match handled at orchestrator level) ─────────────
    # Fuzzy entity resolution is done before retrieval in the orchestrator,
    # enriching the query with canonical entity names.

    # ── Stage 4: Reciprocal Rank Fusion ──────────────────────────────────
    fused = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        id_key="chunk_id",
    )
    candidates = fused[:config.RERANK_CANDIDATES]
    log.info("Stage 4: RRF fusion", extra={"candidates": len(candidates)})

    # ── Stage 5: Jina reranker ───────────────────────────────────────────
    reranked = rerank(
        query=query,
        documents=candidates,
        top_k=config.RERANK_TOP_K,
    )
    log.info("Stage 5: Reranked", extra={"final_chunks": len(reranked)})

    return reranked
