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
import re

try:
    import psycopg
    from psycopg import sql
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
# compare same-dimension vectors. Because the column is dimensionless, a plain
# `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` is impossible
# (HNSW needs a fixed dimension). Instead `ensure_ann_indexes()` builds one
# PARTIAL HNSW index per distinct `embedding_model` over the fixed-dimension
# expression `embedding::vector(N)` — see that function below.
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
    latency_ms      INTEGER,
    prices          JSONB             -- persisted daily-close series (Polygon),
                                      -- served by /api/price so the request path
                                      -- never has to call Polygon live.
);

-- Idempotent migration for snapshots created before the `prices` column existed.
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS prices JSONB;
-- N2: public-analyzer insights (technical/valuation/analyst summary) per snapshot.
ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS insights JSONB;

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

-- At most one in-flight job per (ticker, horizon): lets create_job upsert with
-- ON CONFLICT DO NOTHING so concurrent on-demand requests can't double-queue.
CREATE UNIQUE INDEX IF NOT EXISTS jobs_pending_uniq
    ON jobs (ticker, horizon) WHERE status IN ('queued', 'running');
"""

# HNSW build parameters for the ANN indexes on `filing_chunks.embedding`.
# These are pgvector's own defaults and a sensible balance of build time, index
# size and recall for a filings corpus (≲10^6 vectors):
#   m               = max edges per node in the graph. Higher → better recall &
#                     bigger/slower-to-build index. 16 is the standard sweet spot.
#   ef_construction = candidate-list size while building. Higher → better recall
#                     at higher build cost. 64 is the default.
# Query-time recall/latency is tuned separately and at runtime via
# `SET hnsw.ef_search = N` (default 40); it is NOT baked into the index, so it
# needs no rebuild to change. Bump ef_search if recall on the ANN path is low.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

_schema_ready = False


def _index_suffix(embedding_model: str) -> str:
    """Sanitise an embedding_model tag into a safe SQL identifier suffix.

    Tags look like 'openai:text-embedding-ada-002-1536d' / 'ollama:nomic-embed-text-768d'
    / 'mock-16d'; collapse everything non-alphanumeric to underscores.
    """
    return re.sub(r"[^0-9a-z]+", "_", embedding_model.lower()).strip("_") or "unknown"


def ensure_ann_indexes(cur: "psycopg.Cursor") -> None:
    """Create a per-embedding_model HNSW cosine index on `filing_chunks`.

    Runs on an already-open cursor (inside the caller's transaction). Idempotent:
    every index is created `IF NOT EXISTS`, so this is cheap to call repeatedly.

    Why one index per embedding_model instead of a single table-wide index:
    `filing_chunks.embedding` is a dimensionless `vector` on purpose, so vectors
    from different providers (1536-d / 768-d / 16-d) share the table. pgvector's
    HNSW requires a *fixed* dimension, so a single index over the raw column is
    not possible. We therefore build one PARTIAL index per distinct
    `embedding_model`, keyed on the expression `embedding::vector(N)` (N = that
    model's actual dimension, read from the data via `vector_dims`) and scoped by
    `WHERE embedding_model = '<model>'`.

    This matches `retrieve_relevant_chunks`, which (a) always filters
    `embedding_model = <active>` and (b) orders by cosine distance `<=>`, so
    `vector_cosine_ops` is the correct operator class and exactly one partial
    index is eligible per query. The retrieval SQL casts its ORDER BY expression
    to the same `::vector(N)` so the planner can match this expression index.
    """
    # One representative row per model is enough to learn its dimension.
    cur.execute(
        "SELECT embedding_model, vector_dims(embedding) AS dim FROM ("
        "  SELECT DISTINCT ON (embedding_model) embedding_model, embedding"
        "  FROM filing_chunks ORDER BY embedding_model"
        ") s"
    )
    for embedding_model, dim in cur.fetchall():
        if not dim or dim <= 0:
            continue
        index_name = f"filing_chunks_hnsw_{_index_suffix(embedding_model)}"
        # NB: the partial predicate must be a literal (index predicates cannot be
        # parameterised), so `embedding_model` is inlined via sql.Literal — safe
        # against injection and correct quoting. `dim` is an int from the DB.
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON filing_chunks "
                "USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
                "WITH (m = {m}, ef_construction = {efc}) "
                "WHERE embedding_model = {model}"
            ).format(
                name=sql.Identifier(index_name),
                dim=sql.SQL(str(int(dim))),
                m=sql.SQL(str(HNSW_M)),
                efc=sql.SQL(str(HNSW_EF_CONSTRUCTION)),
                model=sql.Literal(embedding_model),
            )
        )


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

    Connection discipline: one short-lived connection PER call (opened in a
    `with` block, closed on exit) — deliberately NO app-side connection pool.
    Pooling is delegated to the managed pooler in front of Postgres (Supabase's
    pgbouncer / RDS Proxy) that the deployment already runs; a second in-process
    pool would just fight it. The worker is serial and the backend read path is
    TTL-cached, so per-call connect volume stays low. Running multiple workers is
    safe because claim_next_job() uses FOR UPDATE SKIP LOCKED.
    """
    from utils.retry import retry_call
    try:
        # prepare_threshold=None disables psycopg3's automatic server-side
        # prepared statements. Bulk ingestion fires hundreds of identical
        # INSERTs, which would otherwise get promoted to prepared statements and
        # then fail behind a transaction-mode connection pooler (Supabase's
        # pooler on :6543, or any pgbouncer in transaction mode), which routes
        # each query to a different backend that never saw the PREPARE — the
        # symptom is "sending prepared query failed: SSL error: bad length" /
        # "SSL SYSCALL error: EOF detected". Disabling auto-prepare is safe on a
        # direct connection too, so this is unconditional.
        #
        # Retry transient connection failures (pooler blips, brief network
        # drops) with bounded backoff before giving up.
        conn = retry_call(
            lambda: psycopg.connect(get_conninfo(), prepare_threshold=None),
            retry_on=(psycopg.OperationalError,), max_delay=4.0, label="db-connect")
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
            # Build ANN indexes for whatever embedding models already have data.
            # (A model ingested for the first time later in this process gets its
            # index from ingestion's own ensure_ann_indexes call — see rag_utils.)
            ensure_ann_indexes(cur)
        conn.commit()
    _schema_ready = True
