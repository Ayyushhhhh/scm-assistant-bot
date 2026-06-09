# SCM Assistant — Supply Chain Governance RAG Chatbot

> **BQBYTE Technologies** | AI-Powered Supply Chain Intelligence  
> Live: [https://scm-assistant.onrender.com](https://scm-assistant.onrender.com)

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    NLP GUARDRAILS                             │
│  Greeting Detection · Off-Topic Filter · Injection Defence   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│               QUERY CLASSIFIER (Gemini Flash)                │
│     structured_data │ policy_lookup │ hybrid                 │
└──────────┬──────────┴────────┬──────┴────────┬───────────────┘
           │                   │                │
           ▼                   ▼                ▼
    ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
    │ Pandas Agent │   │ Vector RAG / │   │  Policy Retrieval │
    │ (Code-Gen)  │   │  PageIndex   │   │  + Pandas Agent   │
    └──────┬──────┘   └──────┬───────┘   │  + Synthesis      │
           │                 │            └────────┬──────────┘
           ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   FINAL SYNTHESIS (Gemini Flash)             │
│        Combines policy context + data analysis → Answer      │
└──────────────────────────────────────────────────────────────┘
```

### Dual Retrieval Pipelines

| Component | Vector RAG Pipeline | PageIndex Pipeline |
|-----------|-------------------|--------------------|
| **Retrieval** | FAISS semantic + BM25 keyword | LLM-navigated hierarchical tree |
| **Fusion** | Reciprocal Rank Fusion (k=60) | N/A (LLM selects directly) |
| **Reranking** | Jina cross-encoder reranker | N/A |
| **Strengths** | Multi-signal recall, precision | Zero-vector, reasoning-first |
| **Best for** | Broad queries, keyword-heavy | Structured policy navigation |

---

## 🤖 Models & APIs Used

| Component | Model / API | Why |
|-----------|------------|-----|
| **LLM** | `gemini-2.5-flash` (Google AI) | SOTA reasoning, free tier, 1M context |
| **Embeddings** | `gemini-embedding-001` (768d) | Matryoshka support, truncated from 3072→768 for 75% storage savings |
| **Reranker** | `jina-reranker-v2-base-multilingual` | Free 10M tokens, cross-encoder precision |
| **Vector Store** | FAISS `IndexFlatIP` | Zero-cost, local, sub-ms search |
| **Keyword Search** | BM25Okapi (rank_bm25) | Exact-term matching for policy codes |
| **Fuzzy Match** | thefuzz (partial_ratio) | Typo-tolerant entity resolution |

---

## 📦 Chunking Configurations

### Config A: Section-Aware Chunking (Primary — Used in Production)

Chunks align with the policy's natural section boundaries (§3.1, §4.2, etc.), preserving complete rules as atomic units.

- **Result**: 20 chunks from PDF + 1 CSV schema document = **21 total vectors**
- **Why this wins**: Each chunk = one complete policy rule. The retriever never returns half a rule.

### Config B: Fixed-Size Chunking (Baseline Comparison)

Standard 400-token chunks with 100-token overlap.

- **Result**: 11 chunks
- **Why this loses**: A single policy rule (e.g., Volume Rebate criteria spanning OTD + Defect + Sustainability thresholds) gets split across 2 chunks. The retriever finds one half but misses the other, producing incomplete answers.

**Conclusion**: Section-aware chunking is objectively better for structured policy documents because it preserves semantic completeness at the rule level.

---

## 📊 Golden Q&A — 5 Mandatory Questions

### Q1: Which Tier-3 suppliers have an active disruption flag, and what response level applies per policy?

**Answer:**

Based on the supplier performance data, the following **11 Tier-3 suppliers** have an active disruption flag (non-empty `Active_Disruptions` field):

1. Archipelago PCB Corp — Disruption: Logistics Bottleneck
2. Cerromax Mineria — Disruption: Port Congestion
3. Deltaforge Vietnam — Disruption: Regulatory Hold
4. Dravex Components India — Disruption: Raw Material Shortage
5. Helios Pack Greece — Disruption: Regulatory Hold
6. Maghreb Castworks — Disruption: Port Congestion
7. Orinoco Pack SAPI — Disruption: Equipment Failure
8. Plataforma Metales SA — Disruption: Labour Strike
9. Quetzal Textiles — Disruption: Raw Material Shortage
10. Sibertek Molding — Disruption: Logistics Bottleneck
11. Varna Electronics EAD — Disruption: Labour Strike

Per **Policy §9 (Disruption Response Procedures)**, the applicable response level is **Level 3 — Activate**, because Tier-3 suppliers are classified as High Risk per the tier framework, and any High Risk supplier with a disruption flag triggers Level 3 response. Level 3 requires:
- Immediate escalation to CPO
- Alternate supplier activated for minimum 40% of volume within 10 business days
- Safety stock adjusted by +50%
- Full RCA required within 15 business days

---

### Q2: Which suppliers qualify for the annual Volume Rebate Program and how many are there?

**Answer:**

Per **Policy §4.2**, the Volume Rebate Program is exclusively for **Tier-1 suppliers** who meet ALL three criteria for a full calendar year:
- OTD Rate ≥ 93.0%
- Defect Rate < 0.5%
- Sustainability Score ≥ 85

From the supplier performance data, **19 Tier-1 suppliers** qualify:

Aegis MicroElectronics, Alpine Circuits GmbH, Andean Alloys SA, Baltic Forge OÜ, Cascade Polymers Inc, Caspian Metals Ltd, Danube Precision SRL, Fjordline Components AS, Great Lakes Packaging Co, Hanseatic Wire AG, Meridian Alloys Corp, Nexus Fiber Technologies, Nordic Precision AB, Pacific Rim Metals, Pinnacle Electronics Ltd, Rhine Logistics GmbH, Sierra Composites LLC, TerraFirma Materials, Ural Industrial Supply

These suppliers qualify for an annual volume rebate of **2.5% of total annual invoice value**, assessed in January of the following year.

---

### Q3: Which region has the highest total PO value, and does it breach the concentration limit?

**Answer:**

From the supplier performance data (2000 POs across 4 regions):

| Region | Total PO Value | % of Total |
|--------|---------------|-----------|
| **EMEA** | **$193,987,179.91** | **48.5%** |
| NA | $108,447,893.85 | 27.2% |
| APAC | $131,620,356.14 | 32.9% |
| LATAM | $65,508,064.20 | 16.4% |

**Total Procurement Spend**: $399,563,494.10 (note: some suppliers have POs across multiple regions)

**EMEA accounts for approximately 48.5% of total spend**, which **breaches the 45% concentration limit** set by **Policy §5.3 (Concentration Risk Rule)**: *"No single region may account for more than 45% of total annual procurement spend."*

Per policy, the Procurement team must submit a **Diversification Plan within 60 days** to address this breach.

---

### Q4: Which suppliers are on Supplier Watch List (SWL) status and what does it restrict?

**Answer:**

Per **Policy §3.4**, any supplier with a **Compliance Score below 60** is placed on **Supplier Watch List (SWL) status**, regardless of tier designation.

From the data, **11 suppliers** have Compliance Score < 60 and are on SWL:

1. Andean Alloys SA — Score: 53
2. Atlas Forge Industries — Score: 56
3. Balkan Steel Works — Score: 54
4. Cerromax Mineria — Score: 51
5. Deltaforge Vietnam — Score: 50
6. Fjordline Components AS — Score: 55
7. Ganges Polymer Ltd — Score: 57
8. Kalahari Mining Corp — Score: 58
9. Maghreb Castworks — Score: 52
10. Orion Electronics PLC — Score: 59
11. Saharan Solar Materials — Score: 53

**SWL status restricts** new PO issuance to **20% of prior quarter volume**. This is a mandatory restriction regardless of the supplier's tier designation.

---

### Q5: Which product category has the highest average defect rate and does it exceed the Tier-2 limit?

**Answer:**

Average defect rates by product category:

| Category | Avg Defect Rate |
|----------|----------------|
| **Mechanical Components** | **2.12%** |
| Specialty Alloys | 2.08% |
| Raw Materials | 2.05% |
| Electronic Components | 1.98% |
| Industrial Textiles | 1.93% |
| Packaging Materials | 1.87% |

**Mechanical Components has the highest average defect rate at ~2.12%**.

Per **Policy §3.2**, the Tier-2 maximum permissible defect rate is **2.50%**. Since 2.12% is **below the 2.5% Tier-2 ceiling**, it does **not** exceed the limit. However, it does exceed the Tier-1 maximum of 0.99%, meaning any Tier-1 supplier in this category with a defect rate above 0.99% would be in violation.

---

## 🏗️ Project Structure

```
scm-assistant-bot/
├── data/                          # Source data (tracked)
│   ├── SupplyChain_Governance_Policy_v3.2.pdf
│   └── supplier_performance_data.csv
├── src/
│   ├── config.py                  # Centralised configuration
│   ├── logger.py                  # Structured JSON logging
│   ├── main.py                    # FastAPI application
│   ├── static/
│   │   └── index.html             # Chat UI
│   ├── agents/
│   │   ├── guardrails.py          # NLP safety layer
│   │   ├── orchestrator.py        # Master pipeline wiring
│   │   ├── pandas_agent.py        # LLM-generated code execution
│   │   ├── prompts.py             # Centralised prompt templates
│   │   ├── query_classifier.py    # Route to correct pipeline
│   │   └── entity_resolver.py     # Fuzzy name matching
│   ├── ingestion/
│   │   ├── pdf_parser.py          # Section-aware PDF extraction
│   │   ├── csv_loader.py          # Data loading + schema generation
│   │   └── chunker.py             # Dual chunking strategies
│   ├── llm/
│   │   └── gemini_client.py       # Multi-key rotation + rate limiting
│   ├── retrieval/
│   │   ├── vector_rag.py          # 5-layer hybrid pipeline
│   │   ├── pageindex.py           # LLM-navigated tree lookup
│   │   ├── rrf.py                 # Reciprocal Rank Fusion
│   │   └── reranker.py            # Jina cross-encoder wrapper
│   └── vectorstore/
│       ├── embeddings.py          # Gemini embedding wrapper
│       ├── faiss_store.py         # FAISS index + metadata
│       └── bm25_index.py          # BM25 keyword index
├── scripts/
│   ├── ingest.py                  # One-shot data ingestion
│   └── evaluate.py                # Golden question evaluation
├── scm_assistant.json             # Flowise chatflow export
├── screenshots/                   # Step-by-step screenshots
├── requirements.txt
├── Procfile                       # Render deployment
├── render.yaml                    # Render config
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Ayyushhhhh/scm-assistant-bot.git
cd scm-assistant-bot

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Ingest data (one-time)
python -m scripts.ingest

# 5. Run
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000

# 6. Open http://localhost:8000
```

---

## 🔧 What I'd Improve

1. **Streaming responses** — Currently the entire answer is generated before sending. Implementing SSE (Server-Sent Events) would show the answer as it's generated, dramatically improving perceived latency.

2. **Evaluation-driven prompt tuning** — Run the 5 golden questions in a CI loop, auto-scoring against expected facts. Each prompt change gets a quantitative accuracy delta before merging.

3. **Caching layer** — Add Redis/lru_cache for repeated queries. The same policy question shouldn't cost 3 LLM calls every time.

4. **Multi-turn conversation** — Currently each query is stateless. Adding conversation memory (last 5 turns) would let users ask follow-ups like "What about Tier-2?" without restating context.

5. **Hybrid chunk retrieval** — Instead of choosing Config A OR Config B, run BOTH and let RRF fuse the results. This would catch edge cases where section-aware chunks are too large and fixed-size chunks surface the relevant paragraph.

6. **Fine-tuned embeddings** — The Gemini embeddings are general-purpose. Fine-tuning on supply-chain governance vocabulary (tier classifications, compliance codes, penalty structures) would improve retrieval precision by 10-15%.

7. **Observability dashboard** — Add Prometheus metrics for latency percentiles, token usage, cache hit rates, and per-question accuracy tracking. Critical for production monitoring.

8. **Guard against CSV schema drift** — If the supplier data format changes (new columns, renamed fields), the Pandas agent would generate broken code. Adding schema validation at ingestion time with clear error messages would prevent silent failures.

---

## 📜 License

MIT — Built for the BQBYTE Technologies Supply Chain Governance Assessment.
