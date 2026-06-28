"""Day 2 fix, committed: ticker validation, JSON error handlers, NaN-safe JSON.

Validation rejects before any DB call, so these need no live database.
"""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))        # src/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))  # repo root (backend pkg)

from backend.app import app  # noqa: E402

client = app.test_client()


def test_invalid_ticker_with_space_400():
    r = client.post("/api/analyze", json={"ticker": "AAPL MSFT"})
    assert r.status_code == 400
    assert "Invalid ticker" in r.get_json()["error"]


def test_injection_ticker_400():
    r = client.get("/api/snapshot/%27%3B%20DROP/MID")
    assert r.status_code == 400


def test_missing_ticker_400():
    assert client.post("/api/analyze", json={}).status_code == 400


def test_overlong_ticker_400():
    assert client.get("/api/price/TOOLONGTICKERXYZ").status_code == 400


def test_bad_horizon_400():
    assert client.get("/api/snapshot/AAPL/NONSENSE").status_code == 400


def test_unknown_route_returns_json_404():
    r = client.get("/no/such/route")
    assert r.status_code == 404
    assert "application/json" in r.headers.get("Content-Type", "")
    assert r.get_json()["status"] == 404


def test_nan_safe_json_provider():
    out = app.json.dumps({"a": float("nan"), "b": float("inf"), "c": 1.5, "d": [float("nan"), 2]})
    assert "NaN" not in out and "Infinity" not in out
    assert "null" in out
