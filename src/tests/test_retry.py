"""Day 1 fix, committed: bounded retry/backoff (utils.retry)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.retry import retry_call, with_retry  # noqa: E402


def test_success_first_try():
    assert retry_call(lambda: 42) == 42


def test_retry_then_succeed():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert retry_call(flaky, base_delay=0, label="flaky") == "ok"
    assert calls["n"] == 3


def test_exhaust_reraises_last_exception():
    def always():
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="always"):
        retry_call(always, attempts=2, base_delay=0)


def test_non_retryable_propagates_without_retry():
    hits = {"n": 0}

    def boom():
        hits["n"] += 1
        raise KeyError("nope")

    with pytest.raises(KeyError):
        retry_call(boom, retry_on=(ValueError,), attempts=5, base_delay=0)
    assert hits["n"] == 1  # called once, never retried


def test_with_retry_decorator():
    calls = {"n": 0}

    @with_retry(base_delay=0)
    def f():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return "done"

    assert f() == "done" and calls["n"] == 2
