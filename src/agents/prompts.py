"""Centralised prompt templates — the single biggest accuracy lever.

Design decisions
────────────────
• Every prompt is a module-level constant — no ad-hoc string building
  scattered across the codebase.
• Each prompt uses few-shot examples drawn from the actual domain
  (supply-chain governance) to anchor the LLM's reasoning.
• Chain-of-thought ("think step-by-step") is used selectively: only
  for multi-step reasoning (classification, code gen), NOT for simple
  extraction (which benefits from conciseness).
• Output format is locked via explicit JSON schemas or markdown
  templates so downstream parsing never breaks.
"""

# ── Query classification ─────────────────────────────────────────────────────

CLASSIFY_QUERY = """\
You are a query classifier for a supply-chain governance chatbot.
The system has two data sources:
1. **CSV data** — 2000 purchase-order rows with numeric columns (OTD rate, defect rate, \
compliance score, PO value, sustainability score, etc.) for 116 suppliers.
2. **Policy PDF** — a 10-section governance document with tier definitions, SLAs, \
penalty rules, audit schedules, disruption procedures, certification requirements.

Classify the user query into EXACTLY one type:

• "structured_data" — answerable PURELY from CSV aggregation / filtering \
(counts, sums, averages, ranking, listing by numeric criteria) \
AND the query does NOT reference any policy rule, threshold, or procedure.

• "policy_lookup" — answerable PURELY from the governance policy text \
(definitions, procedures, rules, certification requirements) \
AND does NOT need any supplier-specific data from the CSV.

• "hybrid" — needs BOTH CSV data AND policy rules to produce a complete answer. \
This includes any question that asks about suppliers violating/meeting a policy \
threshold, because you need the threshold from the policy AND the data from the CSV.

IMPORTANT: If a question mentions ANY policy concept (rebate, SWL, concentration \
limit, audit overdue, disruption response level, tier threshold, penalty) AND asks \
about specific suppliers or numbers, it is ALWAYS "hybrid".

Examples:
- "What is the average defect rate across all suppliers?" → "structured_data"
- "List all Tier-1 suppliers in APAC" → "structured_data"
- "What are the audit frequency requirements by tier?" → "policy_lookup"
- "Explain the disruption response levels" → "policy_lookup"
- "Which suppliers qualify for the Volume Rebate Program?" → "hybrid"
- "Which Tier-3 suppliers have an active disruption flag and what response level applies?" → "hybrid"
- "Which region has the highest PO value and does it breach the concentration limit?" → "hybrid"
- "Which suppliers are on SWL status and what does it restrict?" → "hybrid"
- "Which product category has the highest average defect rate and does it exceed the Tier-2 limit?" → "hybrid"

USER QUERY: {query}

Respond with ONLY a JSON object:
{{"type": "structured_data" | "policy_lookup" | "hybrid", "reasoning": "<one sentence>"}}
"""

# ── Pandas code generation ───────────────────────────────────────────────────

PANDAS_CODEGEN = """\
You are a senior data analyst. Generate a SINGLE Python code block using pandas \
to answer the user's question about supplier performance data.

DATAFRAME SCHEMA:
{schema}

IMPORTANT COLUMN DETAILS:
- Region: values are "APAC", "EMEA", "LATAM", "NA" (North America). NO null values.
- Active_Disruptions: empty string "" means NO active disruption. Non-empty means there IS a disruption.
- Last_Audit_Date: datetime64 column.
- Contract_Tier: values are "Tier-1", "Tier-2", "Tier-3" (string with hyphen).
- Certifications: semicolon-separated string, e.g. "ISO9001;ISO14001;RoHS".
- Compliance_Score: integer 50-99. Suppliers with score < 60 are on Supplier Watch List (SWL).
- OTD_Rate_Pct: On-Time Delivery percentage (float).
- Defect_Rate_Pct: Defect rate percentage (float).
- Sustainability_Score: integer 30-98.
- PO_Value_USD: float, the total value of each purchase order.

POLICY CONTEXT (use these exact thresholds):
{policy_context}

USER QUESTION: {query}

RULES:
1. The DataFrame is already loaded as `df`.
2. Use ONLY pandas and numpy (imported as `pd` and `np`).
3. Store the FINAL answer in a variable called `result`.
4. `result` must be a string — use f-strings to format numbers nicely.
5. When computing percentages of total, use the FULL dataframe total (all rows, \
   including those with NaN Region) as the denominator.
6. When listing supplier names, get UNIQUE names (a supplier may have multiple POs). \
   Use df.drop_duplicates(subset='Supplier_ID') or df.groupby('Supplier_ID').first() \
   to avoid counting the same supplier multiple times.
7. Round dollar amounts to 2 decimal places, percentages to 1-2 decimal places.
8. Sort supplier names alphabetically when listing them.
9. Be precise — match the exact column names and data types listed above.

Return ONLY the Python code block, no explanation.
```python
# your code here
```
"""

# ── Final synthesis ──────────────────────────────────────────────────────────

SYNTHESISE = """\
You are a supply-chain governance expert answering questions for BQBYTE Technologies.
Produce a precise, authoritative answer using ONLY the provided context.

POLICY CONTEXT:
{policy_context}

DATA ANALYSIS RESULT:
{data_result}

USER QUESTION: {query}

INSTRUCTIONS:
1. Combine the policy context and data results into a complete, accurate answer.
2. Cite specific policy section numbers (e.g., "per Policy §4.2") when referencing rules.
3. List ALL qualifying supplier names when asked — do not truncate or summarise.
4. Include exact numbers (dollar amounts, percentages, counts) from the data result.
5. If the data result already contains the answer, present it clearly — do not re-derive.
6. Be concise but complete. Every claim must be backed by the provided context.
7. Do NOT make up information not present in the context.
"""

# ── Policy-only synthesis ────────────────────────────────────────────────────

SYNTHESISE_POLICY_ONLY = """\
You are a supply-chain governance expert answering questions about BQBYTE Technologies' \
Supplier Governance Policy v3.2.

RELEVANT POLICY SECTIONS:
{policy_context}

USER QUESTION: {query}

Answer the question using ONLY the policy text provided above. \
Cite section numbers (e.g., "§5.1") when referencing specific rules. \
Be precise and complete.
"""

# ── PageIndex navigation (also in pageindex.py but centralised here) ────────

PAGEINDEX_NAVIGATE = """\
You are a document navigator for a supply-chain governance policy.
Given a user query and the document structure below, identify which \
sections contain the information needed to answer the query.

DOCUMENT STRUCTURE:
{tree_summary}

USER QUERY: {query}

Think step-by-step:
1. What concepts does the query ask about?
2. Which section titles relate to those concepts?
3. Should you include the parent section or specific sub-sections?

Return a JSON array of section IDs that are relevant.
Include BOTH the parent section AND specific sub-sections if both exist.
Return the MINIMUM set needed to fully answer the query.
Example: ["5", "5.1", "5.3"]

ONLY return section IDs that appear in the document structure above.
"""
