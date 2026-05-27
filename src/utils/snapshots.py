"""
DB helpers for the `snapshots` and `jobs` tables.

A *snapshot* is one pipeline-output row per `(ticker, horizon, generated_at)` —
they're append-only, so the same table doubles as the history timeseries. A
*job* is an on-demand pipeline request; the worker (22.5) consumes queued rows
and the polling status endpoint (23.5) reads progress / final snapshot id.

This module is intentionally storage-only — it does not run pipelines or know
about agents. The pipeline entry point lives in `src/main.py`
(`run_pipeline_for_horizon`) and calls `save_snapshot` once the State is built.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from psycopg.types.json import Jsonb

from utils.db import connect, ensure_schema
from utils.horizons import Horizon, HorizonName


# ----------------------------------------------------------------- snapshots
_SNAPSHOT_COLS = (
    "id", "ticker", "horizon", "generated_at",
    "sentiment", "fundamental", "valuation", "metrics", "debate",
    "report_markdown", "cost_usd", "latency_ms",
)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return dict(zip(_SNAPSHOT_COLS, row))


def _to_jsonb(model: Any) -> Optional[Jsonb]:
    """Pydantic model -> Jsonb param, or None."""
    return Jsonb(model.model_dump(mode="json")) if model is not None else None


def save_snapshot(*, ticker: str, horizon: HorizonName, state: Any,
                  latency_ms: int, cost_usd: Optional[float] = None) -> int:
    """Persist one snapshot row from a completed pipeline `State`. Returns the
    new snapshot id. `state.report.markdown_report` is stored as plain text;
    everything else as JSONB."""
    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO snapshots
               (ticker, horizon, sentiment, fundamental, valuation, metrics,
                debate, report_markdown, cost_usd, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                ticker.upper(), horizon,
                _to_jsonb(getattr(state, "sentiment", None)),
                _to_jsonb(getattr(state, "fundamental", None)),
                _to_jsonb(getattr(state, "valuation", None)),
                _to_jsonb(getattr(state, "metrics", None)),
                _to_jsonb(getattr(state, "debate", None)),
                state.report.markdown_report if getattr(state, "report", None) else None,
                cost_usd, latency_ms,
            ),
        )
        snapshot_id = cur.fetchone()[0]
        conn.commit()
    return snapshot_id


def get_latest_snapshot(ticker: str, horizon: HorizonName) -> Optional[dict]:
    """Return the most recent snapshot row for (ticker, horizon), or None."""
    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_SNAPSHOT_COLS)} FROM snapshots "
            "WHERE ticker = %s AND horizon = %s "
            "ORDER BY generated_at DESC LIMIT 1",
            (ticker.upper(), horizon),
        )
        row = cur.fetchone()
    return _row_to_dict(row) if row else None


def list_tracked_tickers() -> list[dict]:
    """Distinct tickers that have at least one snapshot, with the most recent
    `generated_at` across all horizons and a total snapshot count. Powers the
    ticker-list landing page."""
    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, MAX(generated_at) AS last_updated, COUNT(*) AS snapshot_count "
            "FROM snapshots GROUP BY ticker ORDER BY ticker"
        )
        rows = cur.fetchall()
    return [
        {"ticker": r[0], "last_updated": r[1], "snapshot_count": r[2]}
        for r in rows
    ]


def is_fresh(snapshot: Optional[dict], horizon: Horizon) -> bool:
    """A snapshot is fresh if its `generated_at` is within
    `horizon.freshness_hours`."""
    if not snapshot or not snapshot.get("generated_at"):
        return False
    age = datetime.now(timezone.utc) - snapshot["generated_at"]
    return age < timedelta(hours=horizon.freshness_hours)


def list_snapshot_history(ticker: str, horizon: HorizonName, *,
                          days: int = 90) -> list[dict]:
    """Return snapshots for (ticker, horizon) within the last `days`, oldest first.
    Powers the run / risk history chart."""
    ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_SNAPSHOT_COLS)} FROM snapshots "
            "WHERE ticker = %s AND horizon = %s AND generated_at >= %s "
            "ORDER BY generated_at ASC",
            (ticker.upper(), horizon, cutoff),
        )
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------- jobs
_JOB_COLS = (
    "id", "ticker", "horizon", "status", "progress", "snapshot_id", "error",
    "requested_at", "started_at", "finished_at",
)


def _job_row_to_dict(row: tuple) -> dict[str, Any]:
    return dict(zip(_JOB_COLS, row))


def create_job(ticker: str, horizon: HorizonName) -> int:
    """Insert a queued job; returns its id. The worker (22.5) picks it up."""
    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (ticker, horizon) VALUES (%s, %s) RETURNING id",
            (ticker.upper(), horizon),
        )
        job_id = cur.fetchone()[0]
        conn.commit()
    return job_id


def get_job(job_id: int) -> Optional[dict]:
    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_JOB_COLS)} FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    return _job_row_to_dict(row) if row else None


def update_job_status(job_id: int, status: str, *,
                      progress: Optional[str] = None,
                      snapshot_id: Optional[int] = None,
                      error: Optional[str] = None) -> None:
    """Patch a job row. Sets started_at on first transition to 'running' and
    finished_at on transitions to 'ready' / 'failed'."""
    ensure_schema()
    sets: list[str] = ["status = %s"]
    params: list[Any] = [status]
    if progress is not None:
        sets.append("progress = %s")
        params.append(progress)
    if snapshot_id is not None:
        sets.append("snapshot_id = %s")
        params.append(snapshot_id)
    if error is not None:
        sets.append("error = %s")
        params.append(error)
    if status == "running":
        sets.append("started_at = COALESCE(started_at, now())")
    if status in ("ready", "failed"):
        sets.append("finished_at = now()")
    params.append(job_id)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s",
            tuple(params),
        )
        conn.commit()
