"""
Day 4 (Sprint 1) — concurrency primitives + failed-job surfacing (no live DB).

  - structural: the multi-worker / dedup / cache-locking guarantees are actually
    in the code (FOR UPDATE SKIP LOCKED, ON CONFLICT DO NOTHING, partial unique
    index, cache Lock),
  - behavioural: /api/snapshot surfaces a recent *failed* job instead of silently
    re-enqueuing it forever, and ?retry=1 forces a fresh job.
"""

import inspect
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import db, snapshots  # noqa: E402


# ---------------------------------------------------- concurrency primitives
def test_claim_next_job_uses_skip_locked():
    src = inspect.getsource(snapshots.claim_next_job)
    assert "FOR UPDATE SKIP LOCKED" in src  # two workers never grab the same job


def test_create_job_dedup_on_conflict():
    src = inspect.getsource(snapshots.create_job)
    assert "ON CONFLICT" in src and "DO NOTHING" in src


def test_schema_has_pending_unique_index():
    # At most one in-flight job per (ticker, horizon).
    assert "jobs_pending_uniq" in db.SCHEMA_DDL
    assert "WHERE status IN ('queued', 'running')" in db.SCHEMA_DDL


def test_get_latest_job_exists():
    assert hasattr(snapshots, "get_latest_job")


# ------------------------------------------------------ failed-job surfacing
import backend.app as appmod  # noqa: E402

_client = appmod.app.test_client()

_FAILED = {
    "id": 5, "ticker": "AAPL", "horizon": "MID", "status": "failed",
    "progress": None, "snapshot_id": None, "error": "boom: provider down",
    "requested_at": None, "started_at": None, "finished_at": None,
}
_QUEUED = {**_FAILED, "id": 6, "status": "queued", "error": None}


def test_failed_job_surfaced_not_reenqueued():
    with patch.object(appmod, "_cached_latest_snapshot", return_value=None), \
         patch.object(appmod, "get_latest_snapshot", return_value=None), \
         patch.object(appmod, "get_latest_job", return_value=_FAILED), \
         patch.object(appmod, "get_or_create_pending_job") as enqueue:
        r = _client.get("/api/snapshot/AAPL/MID")
        assert r.status_code == 202
        body = r.get_json()
        assert body["status"] == "failed"
        assert body["error"].startswith("boom")
        enqueue.assert_not_called()  # no perpetual re-enqueue


def test_retry_forces_new_job_past_failure():
    with patch.object(appmod, "_cached_latest_snapshot", return_value=None), \
         patch.object(appmod, "get_latest_snapshot", return_value=None), \
         patch.object(appmod, "get_latest_job", return_value=_FAILED), \
         patch.object(appmod, "get_or_create_pending_job", return_value=_QUEUED) as enqueue:
        r = _client.get("/api/snapshot/AAPL/MID?retry=1")
        assert r.status_code == 202
        assert r.get_json()["status"] == "queued"
        enqueue.assert_called_once()
