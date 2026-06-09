"""FastAPI application — serves the RAG chatbot as HTTP endpoints + chat UI.

Endpoints
─────────
POST /chat          Main chat endpoint (query + optional mode)
GET  /health        Health check for uptime monitoring
GET  /              Serves the chat frontend

Design decisions
────────────────
• All heavy objects (FAISS, BM25, DataFrame, PageIndex tree) are loaded
  ONCE at startup via the lifespan context manager.
• NLP guardrails (greeting, off-topic, injection) run BEFORE any LLM
  call to save quota and protect the system.
• CORS is wide-open for Flowise Cloud compatibility.
• Static chat UI is served from the same server for single-deployment.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.agents.entity_resolver import EntityResolver
from src.agents.guardrails import (
    WELCOME_MESSAGE,
    detect_greeting,
    detect_offtopic,
    format_response,
    sanitise_input,
    _OFFTOPIC_RESPONSE,
)
from src.agents.orchestrator import Orchestrator, PipelineMode
from src.ingestion.csv_loader import load_csv
from src.logger import get_logger
from src.retrieval.pageindex import load_tree
from src.vectorstore.bm25_index import BM25Index
from src.vectorstore.faiss_store import FAISSStore

log = get_logger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────
_orchestrator: Orchestrator | None = None
_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all indexes and data at startup."""
    global _orchestrator

    log.info("Loading data and indexes...")

    df = load_csv(config.CSV_PATH)
    index_dir = config.INDEX_DIR

    faiss_store = FAISSStore.load(index_dir / "faiss")
    bm25_index = BM25Index.load(index_dir / "bm25_index.pkl")
    policy_tree = load_tree(index_dir / "policy_tree.json")
    entity_resolver = EntityResolver(df)

    _orchestrator = Orchestrator(
        df=df,
        faiss_store=faiss_store,
        bm25_index=bm25_index,
        policy_tree=policy_tree,
        entity_resolver=entity_resolver,
    )

    log.info("All indexes loaded — server ready")
    yield
    log.info("Server shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SCM Assistant",
    description="Supply Chain RAG Chatbot — BQBYTE Technologies",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    mode: PipelineMode = Field(
        default="vector_rag",
        description="Pipeline mode: 'vector_rag' or 'pageindex'",
    )


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    mode: str
    metrics: dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>SCM Assistant — API is running</h1>")


@app.get("/welcome")
async def welcome():
    """Return the welcome message for the chat UI."""
    return {"message": WELCOME_MESSAGE}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user query through the NLP guardrails + RAG pipeline."""
    raw_query = request.query
    log.info("Chat request", extra={"query": raw_query[:80], "mode": request.mode})

    # ── Guardrail 1: Input sanitisation ──────────────────────────────────
    cleaned = sanitise_input(raw_query)
    if not cleaned:
        return ChatResponse(
            answer="⚠️ I couldn't process that query. Please ask a supply-chain related question.",
            query_type="rejected",
            mode=request.mode,
            metrics={"total_llm_calls": 0, "reason": "input_sanitisation"},
        )

    # ── Guardrail 2: Greeting detection ──────────────────────────────────
    if detect_greeting(cleaned):
        return ChatResponse(
            answer=WELCOME_MESSAGE,
            query_type="greeting",
            mode=request.mode,
            metrics={"total_llm_calls": 0},
        )

    # ── Guardrail 3: Off-topic detection ─────────────────────────────────
    if detect_offtopic(cleaned):
        return ChatResponse(
            answer=_OFFTOPIC_RESPONSE,
            query_type="off_topic",
            mode=request.mode,
            metrics={"total_llm_calls": 0},
        )

    # ── Main pipeline ────────────────────────────────────────────────────
    result = _orchestrator.answer(cleaned, mode=request.mode)

    # Format the response for display
    result["answer"] = format_response(result["answer"])

    log.info(
        "Chat response",
        extra={
            "query_type": result["query_type"],
            "llm_calls": result["metrics"]["total_llm_calls"],
            "answer_len": len(result["answer"]),
        },
    )
    return ChatResponse(**result)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": config.GEMINI_MODEL,
        "indexes_loaded": _orchestrator is not None,
    }
