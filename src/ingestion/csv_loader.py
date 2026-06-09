"""CSV loader with schema extraction for the Pandas agent.

Design decisions
────────────────
• CSV rows are NEVER vectorised — tabular data is handled entirely by the
  Pandas code-generation agent.  Vectorising rows is an anti-pattern that
  destroys numeric precision and makes aggregation impossible.
• Instead, we generate a concise *schema document* that IS embedded into the
  vector store.  This lets the retriever know what structured data exists
  without polluting the vector space with 2 000 rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the supplier CSV and perform minimal cleaning.

    Cleaning steps (each justified):
    1. Parse Last_Audit_Date as datetime — needed for audit-overdue calculations.
    2. Strip whitespace from string columns — prevents fuzzy-match failures.
    3. Leave Active_Disruptions NaN as-is — NaN = no disruption (semantically correct).
    """
    # CRITICAL: keep_default_na=False prevents pandas from treating "NA"
    # (North America region code) as NaN.  Without this fix, 255 rows lose
    # their Region, making APAC appear largest instead of EMEA.
    df = pd.read_csv(str(csv_path), keep_default_na=False, na_values=[""])

    # 1. Parse dates
    df["Last_Audit_Date"] = pd.to_datetime(df["Last_Audit_Date"], errors="coerce")

    # 2. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]

    log.info(
        "CSV loaded",
        extra={
            "rows": len(df),
            "cols": len(df.columns),
            "suppliers": df["Supplier_ID"].nunique(),
            "null_region": int(df["Region"].isna().sum()),
            "null_disruptions": int(df["Active_Disruptions"].isna().sum()),
        },
    )
    return df


def generate_schema_doc(df: pd.DataFrame) -> str:
    """Generate a human-readable schema document for embedding.

    This document is vectorised so the retriever can understand what
    structured data is available, without embedding actual rows.
    """
    lines = [
        "# Supplier Performance Data Schema",
        f"Total rows: {len(df)} purchase orders across {df['Supplier_ID'].nunique()} suppliers.",
        "",
        "## Columns:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        sample = df[col].dropna().unique()[:5].tolist()
        lines.append(f"- **{col}** ({dtype}): {non_null} non-null values. Examples: {sample}")

    lines.extend([
        "",
        "## Key categorical values:",
        f"- Product_Category: {df['Product_Category'].unique().tolist()}",
        f"- Contract_Tier: {df['Contract_Tier'].unique().tolist()}",
        f"- Risk_Level: {df['Risk_Level'].unique().tolist()}",
        f"- Region: {df['Region'].dropna().unique().tolist()}",
        f"- PO_Quarter: {sorted(df['PO_Quarter'].unique().tolist())}",
    ])

    schema_doc = "\n".join(lines)
    log.info("Schema document generated", extra={"chars": len(schema_doc)})
    return schema_doc


def get_schema_for_prompt(df: pd.DataFrame) -> str:
    """Generate a compact schema string for the Pandas agent prompt.

    Kept deliberately terse to minimise token usage in every LLM call
    that needs to reason about the CSV structure.
    """
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_info.append(f"  {col} ({dtype})")

    return (
        f"DataFrame `df` — {len(df)} rows, {df['Supplier_ID'].nunique()} unique suppliers.\n"
        f"Columns:\n" + "\n".join(col_info)
    )
