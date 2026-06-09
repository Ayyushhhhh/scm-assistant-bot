"""BM25 keyword search index over policy chunks.

Design decisions
────────────────
• BM25 catches exact-term matches that semantic search misses.
  Example: "Tier-3" is a precise keyword — semantic search might
  return "Tier-1" context because the embeddings are very close.
• Uses rank_bm25.BM25Okapi — the Okapi variant includes document
  length normalisation, which is important when chunk sizes vary
  (section-aware chunking produces variable-length chunks).
• Tokenisation is simple whitespace + lowering — sufficient for
  our structured policy language.  No need for NLTK/spaCy overhead.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.logger import get_logger, timer

log = get_logger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace tokeniser with lowering and punctuation removal."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-.]", " ", text)  # keep hyphens and dots (Tier-1, §5.3)
    return text.split()


class BM25Index:
    """In-memory BM25 index over text documents."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []
        self._corpus: list[list[str]] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of document dicts.

        Each dict must have a 'content' key with the text.
        """
        self._documents = documents
        self._corpus = [_tokenise(doc["content"]) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

        log.info("BM25 index built", extra={"docs": len(documents)})

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return top-k documents ranked by BM25 score."""
        if not self._bm25:
            return []

        tokens = _tokenise(query)
        with timer() as t:
            scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._documents[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)

        log.info(
            "BM25 search complete",
            extra={
                "query_tokens": len(tokens),
                "top_k": top_k,
                "results_with_score": len(results),
                "latency_ms": round(t["elapsed_ms"], 1),
            },
        )
        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """Pickle the entire index to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self._documents, "corpus": self._corpus}, f)
        log.info("BM25 index saved", extra={"path": str(path)})

    @classmethod
    def load(cls, filepath: str | Path) -> "BM25Index":
        """Load a pickled index."""
        with open(str(filepath), "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._documents = data["documents"]
        idx._corpus = data["corpus"]
        idx._bm25 = BM25Okapi(idx._corpus)
        log.info("BM25 index loaded", extra={"path": str(filepath), "docs": len(idx._documents)})
        return idx
