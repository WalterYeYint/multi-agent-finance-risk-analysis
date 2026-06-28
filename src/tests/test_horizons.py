"""Unit tests for the horizon preset module (pure Python — no DB / no network)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.horizons import HORIZONS, get_horizon  # noqa: E402
from utils.tools import period_to_datetime_range  # noqa: E402


# Preset (period, horizon_days, freshness_hours). Freshness was unified to 168h
# (weekly) across all three horizons — see src/utils/horizons.py HORIZONS.
EXPECTED = {
    "SHORT": ("1mo", 7,  168),
    "MID":   ("6mo", 30, 168),
    "LONG":  ("2y",  90, 168),
}


@pytest.mark.parametrize("name,expected", list(EXPECTED.items()))
def test_preset_matches_spec(name, expected):
    h = HORIZONS[name]
    assert (h.period, h.horizon_days, h.freshness_hours) == expected
    assert h.name == name


def test_get_horizon_case_insensitive():
    assert get_horizon("short").name == "SHORT"
    assert get_horizon("MID").name == "MID"
    assert get_horizon(" Long ".strip()).name == "LONG"


def test_get_horizon_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown horizon"):
        get_horizon("YEARLY")
    with pytest.raises(ValueError):
        get_horizon("")


@pytest.mark.parametrize("name,min_days,max_days", [
    # Approximate lookback windows from period_to_datetime_range; pad ±5 days
    # so calendar months / leap years don't make the test brittle.
    ("SHORT", 25,    37),
    ("MID",   175,   190),
    ("LONG",  725,   735),
])
def test_period_string_resolves_to_expected_window(name, min_days, max_days):
    """The yfinance period strings we picked must round-trip through
    period_to_datetime_range to a plausible lookback window."""
    h = get_horizon(name)
    end = datetime(2026, 1, 15)
    start, end_out = period_to_datetime_range(h.period, end)
    span = (end_out - start).days
    assert min_days <= span <= max_days, (
        f"{name} period={h.period!r} -> {span}d, expected {min_days}-{max_days}d")
