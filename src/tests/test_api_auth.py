"""I1: API auth + enqueue rate limiting.

These exercise the enqueue guard directly (no DB): the guard is the only gated
path, and auth is opt-in (enforced only when API_KEYS is set), while the
per-identity rate limit always applies.
"""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))        # src/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))  # repo root (backend pkg)

import pytest  # noqa: E402
from backend import app as A  # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Clean env + rate-limit state before each test."""
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("ENQUEUE_RATE_LIMIT_PER_HOUR", raising=False)
    A._enqueue_hits.clear()
    yield
    A._enqueue_hits.clear()


def _authorize(headers=None):
    with A.app.test_request_context(headers=headers or {}):
        r = A._authorize_enqueue()
    return "ALLOW" if r is None else r[1]


def test_dev_mode_no_keys_allows(monkeypatch):
    monkeypatch.setenv("ENQUEUE_RATE_LIMIT_PER_HOUR", "30")
    assert _authorize() == "ALLOW"


def test_keys_set_missing_key_401(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret1,secret2")
    assert _authorize() == 401


def test_keys_set_wrong_key_401(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret1")
    assert _authorize({"X-API-Key": "nope"}) == 401


def test_correct_x_api_key_allows(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret1")
    assert _authorize({"X-API-Key": "secret1"}) == "ALLOW"


def test_correct_bearer_allows(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret1")
    assert _authorize({"Authorization": "Bearer secret1"}) == "ALLOW"


def test_rate_limit_trips_after_quota(monkeypatch):
    monkeypatch.setenv("ENQUEUE_RATE_LIMIT_PER_HOUR", "3")
    with A.app.test_request_context(headers={"X-Forwarded-For": "1.2.3.4"}):
        statuses = ["ALLOW" if A._authorize_enqueue() is None else A._authorize_enqueue()[1]
                    for _ in range(3)]
        over = A._authorize_enqueue()
    assert statuses == ["ALLOW", "ALLOW", "ALLOW"]
    assert over is not None and over[1] == 429


def test_rate_limit_is_per_identity(monkeypatch):
    monkeypatch.setenv("ENQUEUE_RATE_LIMIT_PER_HOUR", "1")
    assert _authorize({"X-Forwarded-For": "1.1.1.1"}) == "ALLOW"
    assert _authorize({"X-Forwarded-For": "1.1.1.1"}) == 429
    assert _authorize({"X-Forwarded-For": "2.2.2.2"}) == "ALLOW"  # different IP unaffected


def test_rate_limit_disabled_with_zero(monkeypatch):
    monkeypatch.setenv("ENQUEUE_RATE_LIMIT_PER_HOUR", "0")
    with A.app.test_request_context(headers={"X-Forwarded-For": "1.2.3.4"}):
        assert all(A._authorize_enqueue() is None for _ in range(50))


def test_429_body_has_code_and_retry_after(monkeypatch):
    monkeypatch.setenv("ENQUEUE_RATE_LIMIT_PER_HOUR", "1")
    with A.app.test_request_context(headers={"X-Forwarded-For": "5.5.5.5"}):
        A._authorize_enqueue()
        resp, code = A._authorize_enqueue()
        body = resp.get_json()
    assert code == 429
    assert body["code"] == "rate_limited"
    assert body["retry_after"] > 0
