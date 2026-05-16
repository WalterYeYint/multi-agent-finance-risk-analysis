# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent finance/risk analysis system built on **LangGraph**. Six specialist agents (data, sentiment, valuation, fundamental, risk, writer) run sequentially in a "chain" workflow, then a separate "debate" workflow runs round-robin debate rounds (fundamental/sentiment/valuation) supervised by a `debate_manager` that converges on a buy/hold/sell recommendation. Flask backend ([backend/app.py](backend/app.py)) exposes a React frontend in [frontend/](frontend/).

## Common commands

### Run the full stack locally
```bash
# Terminal 1 — Ollama (only needed when MODEL_PROVIDER=ollama, the default when no OPENAI_API_KEY)
ollama serve

# Terminal 2 — Flask backend on port 8000 (NOT 5000, despite older docs)
source .venv/bin/activate
python -m backend.app

# Terminal 3 — React frontend on port 3000 (proxies /api → http://localhost:8000)
cd frontend && npm start
```

### Run the multi-agent workflow directly (no UI)
```bash
python -m src.main          # runs chain → debate, writes final_state.json, final_state_with_debate.json, and AnalysisReport_<TICKER>.pdf
```

### Tests
```bash
# Integration test using pre-saved JSON outputs in src/tests/json/
cd src/tests && pytest integration_test.py -v -s

# Single parametrized case
cd src/tests && pytest integration_test.py -v -s -k "GOOGL_1mo_2025-4-1"

# Regenerate the five JSON fixtures by re-running the full workflow, then run pytest
bash test_program.sh
```

The integration test compares the debate's BUY/SELL/HOLD recommendation against the actual average close-price delta over `horizon_days` after `end_date` — it only passes if the recommendation aligned with what subsequently happened in the market.

### Setup
```bash
bash setup.sh               # installs frontend (npm), backend, and root Python deps
```

### Streamlit alternative UI (separate from the React app)
```bash
streamlit run chatgpt_ui.py
```

## Architecture

### Two LangGraph workflows in [src/main.py](src/main.py)

1. **`build_chain_graph()`** — linear: `data → sentiment → valuation → fundamental → risk → writer → END`. Each node is a function in [src/agents.py](src/agents.py) that takes and returns a `State` (pydantic model with optional `MarketData`, `NewsBundle`, `SentimentSummary`, `ValuationMetrics`, `FundamentalAnalysis`, `RiskMetrics`, `RiskReport`, `DebateReport` fields).
2. **`build_final_recommendation_graph()`** — debate loop: `debate_manager` is the entry and the hub; `route_debate` conditionally dispatches to whichever specialist (`debate_fundamental`, `debate_sentiment`, `debate_valuation`) has the lowest turn count, until either consensus is reached (`terminated == "END"`) or `agent_max_turn` (default 5) is hit (`terminated == "ENDMAX"`), then routes to `writer`.

The backend's `/api/analyze` endpoint runs the chain graph, then constructs a `DebateReport`, attaches it to the resulting `State`, and runs the debate graph on top of it (with `recursion_limit=100`). The full run executes inside a watchdog thread with `ANALYZE_TIMEOUT_SECS` (default 900s).

### LLM provider abstraction

[src/utils/config.py](src/utils/config.py) `get_llm()` auto-selects a provider in priority order: OpenAI (`OPENAI_API_KEY` → `gpt-4o`) → Anthropic → Google → Ollama (default `llama3.2:3b`, or `OLLAMA_MODEL`) → `MockLLM` fallback. Override with `MODEL_PROVIDER` env var. The backend's `_detect_model_provider()` mirrors this logic at the HTTP layer (and resolves `MODEL_PROVIDER=auto` to either openai or ollama based on key presence).

`get_embeddings()` follows the same pattern for RAG (OpenAI embeddings or Ollama `nomic-embed-text`).

### Fundamental RAG

[src/utils/rag_utils.py](src/utils/rag_utils.py) `FundamentalRAG` stores filings in **Postgres + pgvector** (not Chroma anymore): filing metadata in the `filings` table, chunk text + embeddings in `filing_chunks`. Connection and schema live in [src/utils/db.py](src/utils/db.py); `ensure_schema()` creates the `vector` extension and tables idempotently. Configure via `DATABASE_URL` (a `postgresql://` URL — a `postgresql+psycopg://` prefix is also accepted). A local Postgres with pgvector is required: `docker run -e POSTGRES_USER=finance -e POSTGRES_PASSWORD=finance -e POSTGRES_DB=finance_rag -p 5432:5432 pgvector/pgvector:pg16`.

The fundamental agent calls the `query_10k_documents` tool ([src/utils/tools.py](src/utils/tools.py)) via a LangGraph `create_react_agent`; that tool calls `retrieve_relevant_chunks()`, which does a cosine-distance (`<=>`) search filtered by ticker and filing date range. Chunks are namespaced by an `embedding_model` tag (provider + vector dimension), so OpenAI/Ollama/mock embeddings can coexist without a dimension clash — retrieval only ever compares same-dimension vectors. Ingestion is idempotent (a filing already present, by SEC accession number or natural key, is skipped). To rebuild, `DROP TABLE filings, filing_chunks` (the `ON DELETE CASCADE` clears chunks with the parent filing).

Two ingestion paths:
- **PDFs** — drop files in [data/filings/](data/filings/) named `TICKER-FILINGTYPE-Q#-STARTMONTH-ENDMONTH-YEAR.pdf` (e.g. `AAPL-10Q-Q3-4-6-2025.pdf`); `batch_ingest_documents()` parses the metadata from the filename. The fundamental agent auto-ingests this directory if a ticker has no stored filings.
- **SEC EDGAR auto-ingest** — `python -m src.utils.edgar_ingest --tickers AAPL,MSFT --forms 10-K,10-Q --limit 4` downloads filings straight from the free EDGAR API, extracts text, and ingests via `ingest_text()`. Cron-safe (skips already-stored filings). Set `SEC_USER_AGENT` to a real contact string or SEC throttles requests.

### Agent response parsing

LLM agents (sentiment, fundamental) are instructed to emit free-text analysis followed by a `STRUCTURED DATA` marker and a fenced ` ```json ` block. [`parse_agent_response()`](src/agents.py) splits on the marker, extracts the JSON, and feeds it into the corresponding pydantic schema in [src/utils/schemas.py](src/utils/schemas.py). If parsing fails, an empty default schema is used. When editing prompts, preserve the `STRUCTURED DATA` marker and the JSON shape, or downstream `**structured_data` unpacking will break.

### State mutation pattern

Agents construct a **new** `State` each step rather than mutating in place — they explicitly carry forward all unchanged fields. When adding a field to `State`, every agent function that returns a new `State` in [src/agents.py](src/agents.py) must be updated to propagate it, or the field will be silently dropped mid-pipeline.

## Environment

Required env vars (set in `.env`, loaded via `python-dotenv`):
- `OPENAI_API_KEY` *or* run Ollama locally (one of the two)
- `DATABASE_URL` — Postgres+pgvector connection for the RAG store (default `postgresql://finance:finance@localhost:5432/finance_rag`)
- `SEC_USER_AGENT` — contact string for SEC EDGAR auto-ingest; `FILINGS_RAW_DIR` — where extracted filing text is archived (default `./data/filings_raw`)
- `POLYGON_API_KEY` — for real news; falls back to synthetic briefs if missing
- `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT` — for LangSmith tracing (optional)
- `ANALYSIS_MODE` — `chain` (default) or `debate`; can also be passed per-request as `mode` in the `/api/analyze` body
- `ANALYZE_TIMEOUT_SECS` — backend watchdog (default 900)
- `HOST`, `PORT` — Flask bind (defaults 127.0.0.1:8000)

## Notes for editing

- The backend serializes `State` into a large hand-rolled dict in `/api/analyze` — when adding fields to a schema, also extend the `safe_get` block in [backend/app.py](backend/app.py) or the frontend won't see them.
- [langgraph.json](langgraph.json) registers two graph entrypoints (`chain` and `debate`) for the LangGraph CLI / Studio; `debate` points at `src/debate_agents.py:build_debate_graph` (separate from the `build_final_recommendation_graph` used at runtime by the backend).
- The writer agent renders a fixed Markdown template; it is not LLM-generated. The `## Investment Final Recommendation` section is appended only when `state.debate.consensus_summary` is present.
- Docker: `Dockerfile.backend` + `Dockerfile.frontend` + `docker-compose.yml` run backend on 8000 and frontend on 3000. AWS deployment via App Runner + S3/CloudFront is documented in [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).
