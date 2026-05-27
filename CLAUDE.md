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

### Eval suite

```bash
python -m src.evals.eval_suite --ticker AAPL --runs 3
```

[src/evals/eval_suite.py](src/evals/eval_suite.py) is a standalone measurement script (not pytest — these are slow, non-deterministic quality metrics). It runs the full pipeline N times and reports: (a) structured-output success rate, (b) number grounding (fundamental-analysis numbers vs. retrieved filing chunks — an approximate hallucination signal), (c) BUY/HOLD/SELL recommendation stability across runs. Needs Postgres + an LLM provider; minutes per run.

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

1. **`build_chain_graph()`** — `data → (sentiment ‖ valuation ‖ fundamental) → risk → writer → END`. `data` fans out to the three independent analyst agents, which run in parallel; `risk` has three incoming edges so LangGraph runs it once, after all three finish. Each node is a function in [src/agents.py](src/agents.py) taking a `State` (pydantic model with optional `MarketData`, `NewsBundle`, `SentimentSummary`, `ValuationMetrics`, `FundamentalAnalysis`, `RiskMetrics`, `RiskReport`, `DebateReport` fields).
2. **`build_final_recommendation_graph()`** — debate loop: `debate_manager` is the entry and the hub; `route_debate` conditionally dispatches to whichever specialist (`debate_fundamental`, `debate_sentiment`, `debate_valuation`) has the lowest turn count, until either consensus is reached (`terminated == "END"`) or `agent_max_turn` (default 5) is hit (`terminated == "ENDMAX"`), then routes to `writer`.

The backend's `/api/analyze` endpoint runs the chain graph, then constructs a `DebateReport`, attaches it to the resulting `State`, and runs the debate graph on top of it (with `recursion_limit=100`). The full run executes inside a watchdog thread with `ANALYZE_TIMEOUT_SECS` (default 900s).

### Horizon presets + snapshots

The product surface no longer accepts a user-supplied `period` / `horizon_days`. Every pipeline run is parameterised by one of three baked-in **horizon presets** defined in [src/utils/horizons.py](src/utils/horizons.py): `SHORT` (1mo lookback / 7d forecast / 24h freshness), `MID` (6mo / 30d / 72h), `LONG` (2y / 90d / 168h). The `HORIZONS` dict is the single source of truth — agents still read `state.period` / `state.horizon_days` as before; only the *caller* (worker / API) maps a horizon name into those fields.

`src/main.py:run_pipeline_for_horizon(ticker, horizon_name)` is the programmatic entry point for the worker and the on-demand API — it reuses the chain + debate graphs, times the run, persists a row to the `snapshots` table, and produces no file IO (unlike `run_all_graphs`, which is kept as the file-writing script for test-data generation).

Two DB tables live alongside `filings`/`filing_chunks` in [src/utils/db.py](src/utils/db.py) `SCHEMA_DDL`: `snapshots` (append-only — `(ticker, horizon, generated_at)`; "latest" is just `ORDER BY generated_at DESC LIMIT 1`, the same table drives the run/risk history timeseries) and `jobs` (queued on-demand requests + worker coordination; status `queued | running | ready | failed`). [src/utils/snapshots.py](src/utils/snapshots.py) holds the CRUD helpers (`save_snapshot`, `get_latest_snapshot`, `is_fresh`, `list_snapshot_history`, `create_job`, `get_job`, `update_job_status`) — that file is storage-only and never invokes the pipeline.

### LLM provider abstraction

[src/utils/config.py](src/utils/config.py) `get_llm()` auto-selects a provider in priority order: OpenAI (`OPENAI_API_KEY` → `gpt-4o`) → Anthropic → Google → Ollama (default `llama3.2:3b`, or `OLLAMA_MODEL`) → `MockLLM` fallback. Override with `MODEL_PROVIDER` env var. The backend's `_detect_model_provider()` mirrors this logic at the HTTP layer (and resolves `MODEL_PROVIDER=auto` to either openai or ollama based on key presence).

`get_embeddings()` follows the same pattern for RAG (OpenAI embeddings or Ollama `nomic-embed-text`).

### Fundamental RAG

[src/utils/rag_utils.py](src/utils/rag_utils.py) `FundamentalRAG` stores filings in **Postgres + pgvector** (not Chroma anymore): filing metadata in the `filings` table, chunk text + embeddings in `filing_chunks`. Connection and schema live in [src/utils/db.py](src/utils/db.py); `ensure_schema()` creates the `vector` extension and tables idempotently. Configure via `DATABASE_URL` (a `postgresql://` URL — a `postgresql+psycopg://` prefix is also accepted). A local Postgres with pgvector is required: `docker run -e POSTGRES_USER=finance -e POSTGRES_PASSWORD=finance -e POSTGRES_DB=finance_rag -p 5432:5432 pgvector/pgvector:pg16`.

The fundamental agent calls the `query_10k_documents` tool ([src/utils/tools.py](src/utils/tools.py)) via a LangGraph `create_react_agent`; that tool calls `retrieve_relevant_chunks()`, which does a cosine-distance (`<=>`) search filtered by ticker and filing date range. Chunks are namespaced by an `embedding_model` tag (provider + vector dimension), so OpenAI/Ollama/mock embeddings can coexist without a dimension clash — retrieval only ever compares same-dimension vectors. Ingestion is idempotent **per embedding model**: a filing already ingested under the active embedding model is skipped, but re-running ingest under a different provider adds that provider's chunks alongside the existing ones (one `filings` row, multiple providers' vectors in `filing_chunks`). To rebuild, `DROP TABLE filings, filing_chunks` (the `ON DELETE CASCADE` clears chunks with the parent filing).

Two ingestion paths:
- **PDFs** — drop files in [data/filings/](data/filings/) named `TICKER-FILINGTYPE-Q#-STARTMONTH-ENDMONTH-YEAR.pdf` (e.g. `AAPL-10Q-Q3-4-6-2025.pdf`); `batch_ingest_documents()` parses the metadata from the filename. The fundamental agent auto-ingests this directory if a ticker has no stored filings.
- **SEC EDGAR auto-ingest** — `python -m src.utils.edgar_ingest --tickers AAPL,MSFT --forms 10-K,10-Q --limit 4` downloads filings straight from the free EDGAR API, extracts text, and ingests via `ingest_text()`. Cron-safe (skips already-stored filings). Set `SEC_USER_AGENT` to a real contact string or SEC throttles requests.

### Agent structured output

The `sentiment` and `fundamental` agents are LangGraph `create_react_agent`s built with `response_format=(structuring_prompt, ExtractSchema)` — after the ReAct loop, langgraph runs a separate constrained-decoding call (`model.with_structured_output(...)`) and the parsed pydantic object lands in `result["structured_response"]`. This works identically for GPT-4o and small local models (e.g. llama3.x) because the schema is enforced by the decoder, not by prompt instructions.

The structuring step uses dedicated **`SentimentExtract` / `FundamentalExtract`** schemas ([src/utils/schemas.py](src/utils/schemas.py)) whose fields are **all required (no defaults)**. This is load-bearing: a field with a default is *optional* in the generated JSON schema, and smaller models silently skip optional fields — so the storage schemas (`SentimentSummary` / `FundamentalAnalysis`, whose fields all have defaults) cannot be used directly as the response format. The agent functions in [src/agents.py](src/agents.py) read `result.get("structured_response")`, fall back to an empty storage schema if it is absent (agent/structuring error), otherwise map the `*Extract` object into the storage schema while filling the system-owned fields (`ticker`, `methodology`, `filing_*`, `analysis_date`, `news_items_analyzed`) themselves. Schema list fields use the `StrList` type ([schemas.py](src/utils/schemas.py) `_coerce_str_list`), which flattens dict-shaped list items as defense-in-depth. When adding a field the model should produce, add it to both the `*Extract` schema and its storage counterpart, and extend the mapping in [src/agents.py](src/agents.py).

### State update pattern

Each chain agent returns a **partial dict** of only the field(s) it produces (e.g. `sentiment_agent` returns `{"sentiment": ...}`) — LangGraph merges that into the graph state, leaving every other field intact. This is what makes the parallel fan-out safe: the three parallel agents write disjoint keys, so there is no channel-write conflict. Do **not** return a full `State(...)` from a chain agent — two parallel nodes writing the same key (`ticker`, etc.) would raise `InvalidUpdateError`, and a partial `State(...)` construction silently drops the unset fields. `graph.invoke()` returns a plain dict; callers wrap it with `State(**result)`.

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
