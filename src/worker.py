"""
Background worker — does the heavy pipeline runs out of the request path.

Two responsibilities:
  1. On-demand: claim queued `jobs` (created by the API on a cache miss), run the
     chain + debate pipeline, persist the snapshot, mark the job ready/failed.
  2. Scheduled refresh: every WORKER_REFRESH_SCAN_SECONDS, enqueue refresh jobs
     for any (ticker, horizon) whose latest snapshot is past its freshness target,
     so popular tickers stay warm.

Run it alongside the Flask backend:
    python -m src.worker

In production this is a long-running Fargate task; the scheduled-refresh scan
can alternatively be triggered by an EventBridge cron hitting
enqueue_stale_refreshes(). Single-worker / serial by design — one pipeline run at
a time to avoid hammering the LLM provider.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from main import run_pipeline_for_horizon  # noqa: E402
from utils.db import ensure_schema  # noqa: E402
from utils.edgar_ingest import refresh_tracked_filings  # noqa: E402
from utils.snapshots import (  # noqa: E402
    claim_next_job, enqueue_stale_refreshes, list_tracked_tickers,
    update_job_status,
)

POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "3"))
REFRESH_SCAN_SECONDS = int(os.getenv("WORKER_REFRESH_SCAN_SECONDS", "300"))
# Weekly sweep: re-check every tracked ticker for NEW SEC filings (default 7 days).
FILING_SCAN_SECONDS = int(os.getenv("WORKER_FILING_SCAN_SECONDS", str(7 * 24 * 3600)))


def process_job(job: dict) -> None:
    """Run the pipeline for one claimed job and record the outcome."""
    job_id, ticker, horizon = job["id"], job["ticker"], job["horizon"]
    print(f"▶️  job {job_id}: {ticker}/{horizon} starting", flush=True)
    try:
        def _progress(msg: str) -> None:
            update_job_status(job_id, "running", progress=msg)
            print(f"   job {job_id}: {msg}", flush=True)

        _, snapshot_id = run_pipeline_for_horizon(
            ticker, horizon, progress_cb=_progress)
        update_job_status(job_id, "ready", snapshot_id=snapshot_id)
        print(f"✅ job {job_id}: ready (snapshot {snapshot_id})", flush=True)
    except Exception as e:  # noqa: BLE001
        update_job_status(job_id, "failed", error=str(e))
        print(f"❌ job {job_id}: failed — {e}", flush=True)
        traceback.print_exc()


def main() -> int:
    ensure_schema()
    print(f"🛠️  worker started (poll={POLL_SECONDS}s, "
          f"refresh-scan={REFRESH_SCAN_SECONDS}s, "
          f"filing-scan={FILING_SCAN_SECONDS}s). Ctrl-C to stop.", flush=True)
    last_scan = 0.0
    last_filing_scan = 0.0
    try:
        while True:
            job = claim_next_job()
            if job is not None:
                process_job(job)
                continue  # drain the queue before idling

            now = time.time()
            if now - last_scan >= REFRESH_SCAN_SECONDS:
                try:
                    n = enqueue_stale_refreshes()
                    if n:
                        print(f"🔄 enqueued {n} stale refresh job(s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️  stale-refresh scan failed: {e}", flush=True)
                last_scan = now

            if now - last_filing_scan >= FILING_SCAN_SECONDS:
                try:
                    tickers = [t["ticker"] for t in list_tracked_tickers()]
                    if tickers:
                        n = refresh_tracked_filings(tickers)
                        print(f"📰 filing sweep: ingested {n} new filing(s) "
                              f"across {len(tickers)} ticker(s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️  filing sweep failed: {e}", flush=True)
                last_filing_scan = now

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n👋 worker stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
