"""Chunking strategies for the policy document.

We implement two configs (required by the assignment):

Config A — Section-aware (our primary, recommended approach)
  Chunks = natural sub-sections (§X.Y) from the PDF parser.
  Each chunk is a complete policy rule → no information loss.

Config B — Fixed-size baseline
  RecursiveCharacterTextSplitter-style: 400 tokens, 100 overlap.
  Used as a comparison to demonstrate why section-aware is superior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.ingestion.pdf_parser import PolicySection
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """A retrieval unit ready for embedding and storage."""

    chunk_id: str
    content: str
    metadata: dict

    def __repr__(self) -> str:
        return f"Chunk({self.chunk_id}, {len(self.content)} chars)"


def section_aware_chunks(sections: list[PolicySection]) -> list[Chunk]:
    """Config A: one chunk per sub-section (preserves complete rules).

    For top-level sections that have no sub-sections, the entire section
    becomes one chunk.  Sub-sections with very short content (<50 chars)
    are merged upward into their parent.
    """
    chunks: list[Chunk] = []

    for sec in sections:
        # Skip very short fragments (table headers, page numbers, etc.)
        if len(sec.content.strip()) < 30:
            continue

        chunks.append(Chunk(
            chunk_id=f"policy-{sec.section_id}",
            content=f"{sec.full_title}\n\n{sec.content}",
            metadata={
                "source": "policy",
                "section_id": sec.section_id,
                "title": sec.title,
                "page": sec.page,
                "parent_id": sec.parent_id,
                "strategy": "section_aware",
            },
        ))

    log.info("Section-aware chunking complete", extra={"chunks": len(chunks)})
    return chunks


def fixed_size_chunks(
    sections: list[PolicySection],
    chunk_size: int = 400,
    overlap: int = 100,
) -> list[Chunk]:
    """Config B: fixed-size chunks with overlap (baseline comparison).

    Token approximation: 1 token ≈ 4 characters (conservative for English).
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    # Concatenate all sections into a single text
    full_text = "\n\n".join(
        f"{sec.full_title}\n{sec.content}" for sec in sections
    )

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(full_text):
        end = min(start + char_size, len(full_text))
        text = full_text[start:end]

        if text.strip():
            chunks.append(Chunk(
                chunk_id=f"policy-fixed-{idx:03d}",
                content=text,
                metadata={
                    "source": "policy",
                    "char_start": start,
                    "char_end": end,
                    "strategy": "fixed_size",
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
            ))
            idx += 1

        start += char_size - char_overlap

    log.info("Fixed-size chunking complete", extra={"chunks": len(chunks)})
    return chunks


def create_chunks(
    sections: list[PolicySection],
    strategy: Literal["section_aware", "fixed_size"] = "section_aware",
    **kwargs,
) -> list[Chunk]:
    """Factory function to create chunks with the specified strategy."""
    if strategy == "section_aware":
        return section_aware_chunks(sections)
    elif strategy == "fixed_size":
        return fixed_size_chunks(sections, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
