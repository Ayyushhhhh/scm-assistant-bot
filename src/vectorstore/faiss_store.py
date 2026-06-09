"""FAISS vector store with parallel metadata storage.

Design decisions
────────────────
• FAISS chosen over Qdrant Cloud because: (a) zero network latency, (b) no
  account setup needed, (c) perfectly adequate for ~50 policy chunks.
• Uses IndexFlatIP (inner product) — equivalent to cosine similarity when
  vectors are L2-normalised (which Gemini embeddings are).
• Metadata stored in a parallel list (index position → document dict).
  Simple, O(1) lookup, and trivially serialisable.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src import config
from src.logger import get_logger, timer

log = get_logger(__name__)


class FAISSStore:
    """Thin wrapper around a FAISS flat index + metadata list."""

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine on normed vecs)
        self.documents: list[dict[str, Any]] = []  # parallel to index rows

    # ── Write operations ─────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, documents: list[dict[str, Any]]) -> None:
        """Add vectors and their corresponding documents.

        Args:
            embeddings: (N, D) float32 array, L2-normalised.
            documents: List of dicts (must be same length as embeddings).
        """
        assert len(embeddings) == len(documents), "Mismatch between embeddings and docs"

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)

        log.info("Vectors added to FAISS", extra={"added": len(documents), "total": self.index.ntotal})

    # ── Read operations ──────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[dict[str, Any]]:
        """Search the index and return top-k results with scores.

        Returns:
            List of dicts: [{content, metadata, score}, ...]
        """
        if self.index.ntotal == 0:
            return []

        query = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        with timer() as t:
            scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        log.info(
            "FAISS search complete",
            extra={
                "top_k": top_k,
                "results": len(results),
                "latency_ms": round(t["elapsed_ms"], 1),
            },
        )
        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Persist index and metadata to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, default=str)

        log.info("FAISS store saved", extra={"dir": str(path), "vectors": self.index.ntotal})

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSStore":
        """Load a previously saved store."""
        path = Path(directory)
        store = cls()
        store.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "documents.json", "r", encoding="utf-8") as f:
            store.documents = json.load(f)
        store.dim = store.index.d

        log.info("FAISS store loaded", extra={"dir": str(path), "vectors": store.index.ntotal})
        return store
