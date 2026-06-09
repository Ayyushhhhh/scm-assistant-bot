"""Evaluation script — tests both pipelines against the 5 golden questions.

Run:
    python -m scripts.evaluate

Compares chatbot answers against expected answers and reports accuracy.
Results are printed to stdout and saved to indexes/eval_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.agents.entity_resolver import EntityResolver
from src.agents.orchestrator import Orchestrator
from src.ingestion.csv_loader import load_csv
from src.logger import get_logger
from src.retrieval.pageindex import load_tree
from src.vectorstore.bm25_index import BM25Index
from src.vectorstore.faiss_store import FAISSStore

log = get_logger(__name__)

# ── Golden questions and key facts to verify ─────────────────────────────────
GOLDEN_QUESTIONS = [
    {
        "id": "Q1",
        "question": "Which Tier-3 suppliers have an active disruption flag, and what response level applies per policy?",
        "must_contain": [
            "11",  # 11 Tier-3 suppliers
            "Level 3",
            "Dravex Components India",
            "Plataforma Metales SA",
            "Maghreb Castworks",
            "Helios Pack Greece",
            "Cerromax Mineria",
            "Orinoco Pack SAPI",
            "Quetzal Textiles",
            "Sibertek Molding",
            "Archipelago PCB Corp",
            "Varna Electronics EAD",
            "Deltaforge Vietnam",
        ],
    },
    {
        "id": "Q2",
        "question": "Which suppliers qualify for the annual Volume Rebate Program and how many are there?",
        "must_contain": [
            "19",  # 19 suppliers
            "Tier-1",
            "OTD",
            "93",
            "Defect",
            "0.5",
            "Sustainability",
            "85",
        ],
    },
    {
        "id": "Q3",
        "question": "Which region has the highest total PO value, and does it breach the concentration limit?",
        "must_contain": [
            "EMEA",
            "48",  # ~48.5%
            "45%",  # concentration cap
            "breach",
            "Diversification",
        ],
    },
    {
        "id": "Q4",
        "question": "Which suppliers are on Supplier Watch List (SWL) status and what does it restrict?",
        "must_contain": [
            "11",  # 11 suppliers
            "Compliance Score",
            "60",
            "20%",  # PO restriction
            "Deltaforge Vietnam",
            "Maghreb Castworks",
        ],
    },
    {
        "id": "Q5",
        "question": "Which product category has the highest average defect rate and does it exceed the Tier-2 limit?",
        "must_contain": [
            "Mechanical Components",
            "2.1",  # avg ~2.12%
            "2.5",  # Tier-2 ceiling
            "below",  # does not exceed
        ],
    },
]


def evaluate(orchestrator: Orchestrator, mode: str = "vector_rag") -> dict:
    """Run all golden questions and evaluate accuracy."""
    results = []
    total_score = 0

    print(f"\n{'='*70}")
    print(f"  EVALUATION — Mode: {mode.upper()}")
    print(f"{'='*70}\n")

    for gq in GOLDEN_QUESTIONS:
        print(f"\n--- {gq['id']}: {gq['question'][:60]}... ---")

        response = orchestrator.answer(gq["question"], mode=mode)
        answer = response["answer"]

        # Check how many required facts are present
        hits = []
        misses = []
        for fact in gq["must_contain"]:
            if fact.lower() in answer.lower():
                hits.append(fact)
            else:
                misses.append(fact)

        score = len(hits) / len(gq["must_contain"])
        total_score += score

        results.append({
            "id": gq["id"],
            "question": gq["question"],
            "answer": answer,
            "score": round(score, 2),
            "hits": hits,
            "misses": misses,
            "metrics": response["metrics"],
        })

        print(f"Answer preview: {answer[:200]}...")
        print(f"Score: {score:.0%} ({len(hits)}/{len(gq['must_contain'])} key facts)")
        if misses:
            print(f"Missing: {misses}")
        print(f"LLM calls: {response['metrics']['total_llm_calls']}")

    avg_score = total_score / len(GOLDEN_QUESTIONS)
    print(f"\n{'='*70}")
    print(f"  OVERALL ACCURACY: {avg_score:.0%}")
    print(f"{'='*70}\n")

    return {
        "mode": mode,
        "overall_accuracy": round(avg_score, 3),
        "results": results,
    }


def main() -> None:
    log.info("Loading indexes for evaluation...")

    df = load_csv(config.CSV_PATH)
    faiss_store = FAISSStore.load(config.INDEX_DIR / "faiss")
    bm25_index = BM25Index.load(config.INDEX_DIR / "bm25_index.pkl")
    policy_tree = load_tree(config.INDEX_DIR / "policy_tree.json")
    entity_resolver = EntityResolver(df)

    orchestrator = Orchestrator(
        df=df,
        faiss_store=faiss_store,
        bm25_index=bm25_index,
        policy_tree=policy_tree,
        entity_resolver=entity_resolver,
    )

    # Evaluate both modes
    all_results = []

    for mode in ["vector_rag", "pageindex"]:
        result = evaluate(orchestrator, mode=mode)
        all_results.append(result)

    # Save results
    output_path = config.INDEX_DIR / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
