"""One-shot ingestion script — parses, chunks, embeds, and indexes all data.

Run this ONCE before starting the server:
    python -m scripts.ingest

What it does:
1. Parse the governance PDF into sections
2. Chunk using BOTH strategies (Config A: section-aware, Config B: fixed-size)
3. Embed Config A chunks with Gemini embeddings
4. Build FAISS index + BM25 index
5. Build PageIndex hierarchical tree
6. Save everything to disk under indexes/

Why Config A is default:
  Section-aware chunking preserves complete policy rules as atomic units.
  Fixed-size chunking splits rules across chunks → retrieval misses half the rule.
  We build both for the README comparison but only index Config A.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingestion.chunker import create_chunks
from src.ingestion.csv_loader import generate_schema_doc, load_csv
from src.ingestion.pdf_parser import parse_pdf
from src.logger import get_logger
from src.retrieval.pageindex import build_tree, save_tree
from src.vectorstore.bm25_index import BM25Index
from src.vectorstore.embeddings import embed_texts
from src.vectorstore.faiss_store import FAISSStore

log = get_logger(__name__)


def main() -> None:
    log.info("=== INGESTION START ===")

    # ── 1. Parse PDF ─────────────────────────────────────────────────────
    log.info("Step 1: Parsing PDF...")
    sections = parse_pdf(config.PDF_PATH)
    log.info(f"Parsed {len(sections)} sections from PDF")

    for sec in sections:
        log.info(f"  §{sec.section_id} {sec.title} ({len(sec.content)} chars)")

    # ── 2. Chunk with BOTH strategies ────────────────────────────────────
    log.info("Step 2: Chunking...")

    # Config A: Section-aware (primary)
    chunks_a = create_chunks(sections, strategy="section_aware")
    log.info(f"Config A (section-aware): {len(chunks_a)} chunks")

    # Config B: Fixed-size (baseline comparison)
    chunks_b = create_chunks(sections, strategy="fixed_size")
    log.info(f"Config B (fixed-size 400tok/100overlap): {len(chunks_b)} chunks")

    # ── 3. Load CSV and generate schema doc ──────────────────────────────
    log.info("Step 3: Loading CSV...")
    df = load_csv(config.CSV_PATH)

    schema_doc = generate_schema_doc(df)
    from src.ingestion.chunker import Chunk
    schema_chunk = Chunk(
        chunk_id="csv-schema",
        content=schema_doc,
        metadata={"source": "csv_schema", "strategy": "section_aware"},
    )

    # Add schema doc to chunks so the retriever knows about CSV data
    all_chunks = chunks_a + [schema_chunk]
    log.info(f"Total chunks for indexing: {len(all_chunks)} (policy + schema)")

    # ── 4. Embed and build FAISS index ───────────────────────────────────
    log.info("Step 4: Embedding chunks with Gemini...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    log.info(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS store
    faiss_store = FAISSStore(dim=config.EMBEDDING_DIM)
    documents = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            **c.metadata,
        }
        for c in all_chunks
    ]
    faiss_store.add(embeddings, documents)

    # Save FAISS
    faiss_dir = config.INDEX_DIR / "faiss"
    faiss_store.save(faiss_dir)

    # ── 5. Build BM25 index ──────────────────────────────────────────────
    log.info("Step 5: Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build(documents)
    bm25_index.save(config.INDEX_DIR / "bm25_index.pkl")

    # ── 6. Build PageIndex tree ──────────────────────────────────────────
    log.info("Step 6: Building PageIndex tree...")
    tree = build_tree(sections)
    save_tree(tree, config.INDEX_DIR / "policy_tree.json")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("=== INGESTION COMPLETE ===")
    log.info(f"Config A chunks: {len(chunks_a)}")
    log.info(f"Config B chunks: {len(chunks_b)}")
    log.info(f"FAISS vectors:   {faiss_store.index.ntotal}")
    log.info(f"BM25 documents:  {len(documents)}")
    log.info(f"PageIndex tree:  {len(tree['sections'])} top-level sections")
    log.info(f"Indexes saved to: {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
