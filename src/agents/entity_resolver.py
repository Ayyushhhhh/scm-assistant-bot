"""Fuzzy entity resolver — typo-tolerant matching for supplier names.

Design decisions
────────────────
• Pre-builds an index of ALL known entity values at startup (116 supplier
  names, 6 categories, 4 regions, 3 tiers, 116 supplier IDs).
• Uses thefuzz.fuzz.partial_ratio for substring matching (handles queries
  like "Orrentek" matching "Orrentek Precision Mfg").
• Threshold of 80 balances recall (catch misspellings) vs precision
  (reject unrelated matches).
• Returns canonical names so downstream agents use exact CSV values.
"""

from __future__ import annotations

import pandas as pd
from thefuzz import fuzz, process

from src import config
from src.logger import get_logger

log = get_logger(__name__)


class EntityResolver:
    """Resolves fuzzy entity mentions in user queries to canonical values."""

    def __init__(self, df: pd.DataFrame) -> None:
        # Pre-build entity indexes (one-time, at startup)
        self.supplier_names: list[str] = sorted(df["Supplier_Name"].unique().tolist())
        self.supplier_ids: list[str] = sorted(df["Supplier_ID"].unique().tolist())
        self.categories: list[str] = sorted(df["Product_Category"].unique().tolist())
        self.regions: list[str] = sorted(df["Region"].dropna().unique().tolist())
        self.tiers: list[str] = sorted(df["Contract_Tier"].unique().tolist())
        self.risk_levels: list[str] = sorted(df["Risk_Level"].unique().tolist())

        # Combined lookup for general entity search
        self._all_entities: dict[str, str] = {}
        for name in self.supplier_names:
            self._all_entities[name] = "supplier_name"
        for sid in self.supplier_ids:
            self._all_entities[sid] = "supplier_id"
        for cat in self.categories:
            self._all_entities[cat] = "category"

        log.info(
            "Entity resolver initialised",
            extra={
                "suppliers": len(self.supplier_names),
                "categories": len(self.categories),
                "regions": len(self.regions),
            },
        )

    def resolve_suppliers(self, query: str) -> list[tuple[str, int]]:
        """Find supplier names mentioned in the query (fuzzy).

        Returns:
            List of (canonical_name, similarity_score) tuples.
        """
        matches = []
        for name in self.supplier_names:
            score = fuzz.partial_ratio(query.lower(), name.lower())
            if score >= config.FUZZY_THRESHOLD:
                matches.append((name, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            log.info(
                "Supplier entities resolved",
                extra={"matches": [(m[0], m[1]) for m in matches[:5]]},
            )
        return matches

    def resolve_category(self, query: str) -> str | None:
        """Find the product category mentioned in the query."""
        result = process.extractOne(
            query, self.categories, scorer=fuzz.partial_ratio,
        )
        if result and result[1] >= config.FUZZY_THRESHOLD:
            log.info("Category resolved", extra={"category": result[0], "score": result[1]})
            return result[0]
        return None

    def enrich_query(self, query: str) -> str:
        """Enrich the query by appending resolved canonical entity names.

        This helps downstream search components match exact values.
        For example, if the user types "Orrentek", we append
        "[Entity: Orrentek Precision Mfg]" so BM25 can match it.
        """
        enrichments = []

        # Resolve supplier names
        suppliers = self.resolve_suppliers(query)
        for name, score in suppliers[:3]:  # top 3 to avoid noise
            if score >= 90:  # high confidence only
                enrichments.append(f"[Supplier: {name}]")

        # Resolve categories
        cat = self.resolve_category(query)
        if cat:
            enrichments.append(f"[Category: {cat}]")

        if enrichments:
            enriched = query + " " + " ".join(enrichments)
            log.info("Query enriched", extra={"additions": enrichments})
            return enriched

        return query
