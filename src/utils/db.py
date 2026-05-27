"""
Postgres + pgvector connection helpers and schema management.

This is the single store for the RAG system: filing metadata lives in `filings`
and chunk text + embeddings live in `filing_chunks`. Run history (Phase 1, 18.5)
will be added as additional tables in this same database.

Connection is configured via env:
- DATABASE_URL  e.g. postgresql://finance:finance@localhost:5432/finance_rag
  (a `postgresql+psycopg://` SQLAlchemy-style prefix is also accepted and
  normalised). If DATABASE_URL is unset, PG* vars / sensible localhost
  defaults are used.

Bring up a local Postgres with the pgvector extension available before use,
e.g. `docker run -e POSTGRES_PASSWORD=finance -e POSTGRES_USER=finance \
 -e POSTGRES_DB=finance_rag -p 5432:5432 pgvector/pgvector:pg16`.
"""

from __future__ import annotations

import os

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError as e:  # pragma: no cover - clear message if deps missing
    raise ImportError(
        "Postgres RAG backend requires `psycopg[binary]` and `pgvector`. "
        "Install them with: pip install 'psycopg[binary]' pgvector"
    ) from e


# DDL is idempotent — safe to run on every process start.
# The `embedding` column is an unconstrained `vector`, so chunks embedded by
# different providers (OpenAI 1536-d, Ollama 768-d, mock 16-d) can coexist.
# Retrieval always filters by `embedding_model`, so distance ops only ever
# compare same-dimension vectors. With a small corpus a sequential scan is
# fine; add an HNSW index per fixed dimension later if the corpus grows.
SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filings (
    id                  BIGSERIAL PRIMARY KEY,
    ticker              TEXT NOT NULL,
    filing_type         TEXT NOT NULL,
    filing_year         INT  NOT NULL,
    filing_start_month  INT  NOT NULL,
    filing_end_month    INT  NOT NULL,
    period_start        DATE,
    period_end          DATE,
    accession_no        TEXT,
    company             TEXT,
    source              TEXT,
    num_chunks          INT  NOT NULL DEFAULT 0,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migration for databases created before period_start/period_end
-- existed. `filing_year/filing_start_month/filing_end_month` are kept as the
-- natural dedup key + metadata; `period_start/period_end` drive date filtering.
ALTER TABLE filings ADD COLUMN IF NOT EXISTS period_start DATE;
ALTER TABLE filings ADD COLUMN IF NOT EXISTS period_end   DATE;

CREATE UNIQUE INDEX IF NOT EXISTS filings_accession_uniq
    ON filings (accession_no) WHERE accession_no IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS filings_natural_uniq
    ON filings (ticker, filing_type, filing_year, filing_start_month, filing_end_month);

CREATE INDEX IF NOT EXISTS filings_ticker_idx ON filings (ticker);

CREATE INDEX IF NOT EXISTS filings_period_idx ON filings (period_start, period_end);

CREATE TABLE IF NOT EXISTS filing_chunks (
    id              BIGSERIAL PRIMARY KEY,
    filing_id       BIGINT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    ticker          TEXT NOT NULL,
    chunk_index     INT  NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector NOT NULL,
    embedding_model TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS filing_chunks_filing_idx ON filing_chunks (filing_id);
CREATE INDEX IF NOT EXISTS filing_chunks_ticker_idx ON filing_chunks (ticker);

-- Append-only history of pipeline runs per (ticker, horizon). The "latest"
-- snapshot is just the newest row for the pair; the same table also drives
-- the run / risk history timeseries.
CREATE TABLE IF NOT EXISTS snapshots (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    horizon         TEXT NOT NULL,    -- 'SHORT' | 'MID' | 'LONG'
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    sentiment       JSONB,
    fundamental     JSONB,
    valuation       JSONB,
    metrics         JSONB,
    debate          JSONB,
    report_markdown TEXT,
    cost_usd        NUMERIC(10, 4),
    latency_ms      INTEGER
);

CREATE INDEX IF NOT EXISTS snapshots_lookup_idx
    ON snapshots (ticker, horizon, generated_at DESC);

-- On-demand pipeline requests + worker coordination. The polling status
-- endpoint (23.5) reads from here; the worker (22.5) consumes queued rows.
CREATE TABLE IF NOT EXISTS jobs (
    id            BIGSERIAL PRIMARY KEY,
    ticker        TEXT NOT NULL,
    horizon       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued | running | ready | failed
    progress      TEXT,
    snapshot_id   BIGINT REFERENCES snapshots(id) ON DELETE SET NULL,
    error         TEXT,
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS jobs_status_idx         ON jobs (status, requested_at);
CREATE INDEX IF NOT EXISTS jobs_ticker_horizon_idx ON jobs (ticker, horizon);
"""

_schema_ready = False


def get_conninfo() -> str:
    """Return a libpq connection string from env, with localhost defaults."""
    url = os.getenv("DATABASE_URL")
    if url:
        # psycopg wants a plain libpq URL; strip a SQLAlchemy driver suffix.
        return url.replace("postgresql+psycopg://", "postgresql://", 1)

    user = os.getenv("POSTGRES_USER", "finance")
    password = os.getenv("POSTGRES_PASSWORD", "finance")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "finance_rag")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def connect(register_types: bool = True) -> "psycopg.Connection":
    """
    Open a psycopg connection.

    With register_types=True (default) the pgvector type adapter is registered,
    which requires the `vector` extension to already exist — so ensure_schema()
    must have run first. ensure_schema() itself connects with register_types=False
    precisely because it is the call that creates the extension.
    """
    try:
        conn = psycopg.connect(get_conninfo())
    except psycopg.OperationalError as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"Could not connect to Postgres at {get_conninfo()!r}. "
            "Is the database running and DATABASE_URL set correctly?"
        ) from e
    if register_types:
        register_vector(conn)
    return conn


def ensure_schema(force: bool = False) -> None:
    """Create the pgvector extension and RAG tables if absent (once per process)."""
    global _schema_ready
    if _schema_ready and not force:
        return
    # No type registration here: this call creates the `vector` extension,
    # so the type does not exist yet at connection time.
    with connect(register_types=False) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)
        conn.commit()
    _schema_ready = True
