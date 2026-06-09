"""Master orchestrator — the brain that wires every agent together.

Design decisions
────────────────
• Single entry point: `answer(query, mode)` handles ALL query types.
• For "hybrid" queries (all 5 golden questions are hybrid), the flow is:
    1. Retrieve policy context (via Vector RAG or PageIndex)
    2. Pass policy context → Pandas Agent (so it knows exact thresholds)
    3. Synthesise both results with Gemini Flash
  This ordering is critical: the Pandas agent CANNOT know the Volume
  Rebate criteria (OTD≥93%, Defect<0.5%, Sust≥85) unless it first
  receives the policy context.
• Entity resolution happens BEFORE retrieval to enrich the query with
  canonical names.
• LLMMetrics tracks every call across all agents for cost visibility.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from src.agents import pandas_agent, query_classifier
from src.agents.entity_resolver import EntityResolver
from src.agents.prompts import SYNTHESISE, SYNTHESISE_POLICY_ONLY
from src.llm import gemini_client
from src.logger import LLMMetrics, get_logger
from src.retrieval import pageindex, vector_rag
from src.vectorstore.bm25_index import BM25Index
from src.vectorstore.faiss_store import FAISSStore

log = get_logger(__name__)

PipelineMode = Literal["vector_rag", "pageindex"]


class Orchestrator:
    """Wires query classification, retrieval, and synthesis together."""

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        faiss_store: FAISSStore,
        bm25_index: BM25Index,
        policy_tree: dict,
        entity_resolver: EntityResolver,
    ) -> None:
        self.df = df
        self.faiss_store = faiss_store
        self.bm25_index = bm25_index
        self.policy_tree = policy_tree
        self.entity_resolver = entity_resolver

    def answer(
        self,
        query: str,
        *,
        mode: PipelineMode = "vector_rag",
    ) -> dict[str, Any]:
        """End-to-end: query → classified → retrieved → synthesised → answer.

        Args:
            query: User's natural-language question.
            mode: "vector_rag" or "pageindex".

        Returns:
            Dict with keys: answer, query_type, mode, metrics, policy_chunks.
        """
        metrics = LLMMetrics()

        # ── Step 1: Entity resolution (zero LLM calls) ──────────────────
        enriched_query = self.entity_resolver.enrich_query(query)

        # ── Step 2: Classify query type (1 LLM call) ────────────────────
        query_type = query_classifier.classify(query, metrics=metrics)
        log.info("Orchestrator routing", extra={"type": query_type, "mode": mode})

        # ── Step 3: Route to pipeline(s) ─────────────────────────────────
        if query_type == "structured_data":
            answer_text = self._handle_structured(query, metrics=metrics)
        elif query_type == "policy_lookup":
            answer_text = self._handle_policy(
                query, enriched_query, mode=mode, metrics=metrics,
            )
        else:  # hybrid
            answer_text = self._handle_hybrid(
                query, enriched_query, mode=mode, metrics=metrics,
            )

        return {
            "answer": answer_text,
            "query_type": query_type,
            "mode": mode,
            "metrics": metrics.summary(),
        }

    # ── Private handlers ─────────────────────────────────────────────────

    def _handle_structured(
        self, query: str, *, metrics: LLMMetrics,
    ) -> str:
        """Pure CSV query — no policy retrieval needed."""
        return pandas_agent.run(query, self.df, metrics=metrics)

    def _handle_policy(
        self,
        query: str,
        enriched_query: str,
        *,
        mode: PipelineMode,
        metrics: LLMMetrics,
    ) -> str:
        """Pure policy query — retrieve and synthesise."""
        policy_chunks = self._retrieve_policy(enriched_query, mode=mode, metrics=metrics)
        policy_context = self._format_policy_context(policy_chunks)

        answer = gemini_client.generate(
            SYNTHESISE_POLICY_ONLY.format(
                policy_context=policy_context,
                query=query,
            ),
            metrics=metrics,
            label="synthesise_policy",
        )
        return answer

    def _handle_hybrid(
        self,
        query: str,
        enriched_query: str,
        *,
        mode: PipelineMode,
        metrics: LLMMetrics,
    ) -> str:
        """Hybrid query — retrieve policy FIRST, then run Pandas with context.

        Critical ordering: policy → pandas → synthesise.
        The Pandas agent needs the policy thresholds to generate correct filters.
        """
        # Step A: Retrieve policy context
        policy_chunks = self._retrieve_policy(enriched_query, mode=mode, metrics=metrics)
        policy_context = self._format_policy_context(policy_chunks)

        # Step B: Run Pandas agent WITH policy context (so it knows thresholds)
        data_result = pandas_agent.run(
            query, self.df, policy_context=policy_context, metrics=metrics,
        )

        # Step C: Final synthesis — combine policy + data into complete answer
        answer = gemini_client.generate(
            SYNTHESISE.format(
                policy_context=policy_context,
                data_result=data_result,
                query=query,
            ),
            metrics=metrics,
            label="synthesise_hybrid",
        )
        return answer

    def _retrieve_policy(
        self,
        query: str,
        *,
        mode: PipelineMode,
        metrics: LLMMetrics,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant policy chunks using the selected pipeline."""
        if mode == "pageindex":
            return pageindex.navigate(
                query, self.policy_tree, metrics=metrics,
            )
        else:
            return vector_rag.retrieve(
                query,
                faiss_store=self.faiss_store,
                bm25_index=self.bm25_index,
                metrics=metrics,
            )

    @staticmethod
    def _format_policy_context(chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a single context string."""
        if not chunks:
            return "No relevant policy sections found."

        parts = []
        for chunk in chunks:
            title = chunk.get("title", chunk.get("section_id", ""))
            content = chunk.get("content", "")
            if title:
                parts.append(f"[{title}]\n{content}")
            else:
                parts.append(content)

        return "\n\n---\n\n".join(parts)
