"""Centralised configuration – single source of truth for every tuneable."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv("/etc/secrets/.env")  # Render uploads secret files here by default


# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _PROJECT_ROOT / "data"
PDF_PATH: Path = DATA_DIR / "SupplyChain_Governance_Policy_v3.2.pdf"
CSV_PATH: Path = DATA_DIR / "supplier_performance_data.csv"
INDEX_DIR: Path = _PROJECT_ROOT / "indexes"

# ── API keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS: list[str] = [
    k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
] or ([GEMINI_API_KEY] if GEMINI_API_KEY else [])
JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")

# ── LLM settings ────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM: int = 768          # Matryoshka truncation (from 3072 → 768)

# ── Retrieval settings ───────────────────────────────────────────────────────
SEMANTIC_TOP_K: int = 20          # candidates from FAISS
BM25_TOP_K: int = 20             # candidates from BM25
RRF_K: int = 60                  # Reciprocal Rank Fusion constant
RERANK_TOP_K: int = 5            # final chunks sent to LLM after reranking
RERANK_CANDIDATES: int = 30      # chunks sent to reranker

# ── Chunking defaults ────────────────────────────────────────────────────────
FIXED_CHUNK_SIZE: int = 400      # tokens (Config B)
FIXED_CHUNK_OVERLAP: int = 100   # tokens (Config B)

# ── Fuzzy matching ───────────────────────────────────────────────────────────
FUZZY_THRESHOLD: int = 80        # minimum similarity score (0-100)

# ── Jina reranker ────────────────────────────────────────────────────────────
JINA_RERANK_URL: str = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL: str = "jina-reranker-v2-base-multilingual"

# ── Server ───────────────────────────────────────────────────────────────────
HOST: str = "0.0.0.0"
PORT: int = 8000
