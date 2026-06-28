"""Day 1 fix, committed: Polygon price source (utils.prices) + get_price_history
producing a Date,Close CSV that the metrics consume to a non-null return."""

import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.prices as prices  # noqa: E402
import utils.tools as tools  # noqa: E402


def _resp(status, results=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json = lambda: {"results": results or []}
    return r


def test_fetch_no_key_returns_empty():
    assert prices.fetch_price_rows_polygon(
        "AAPL", date(2024, 1, 1), date(2024, 1, 5), api_key=None) == []


def test_fetch_retries_503_then_200():
    seq = [_resp(503, text="busy"),
           _resp(200, results=[{"c": 185.6, "t": 1704153600000}])]
    with patch("utils.prices.requests.get", side_effect=seq) as g, patch("time.sleep"):
        rows = prices.fetch_price_rows_polygon(
            "AAPL", date(2024, 1, 1), date(2024, 1, 5), api_key="X")
    assert rows == [{"date": "2024-01-02", "close": 185.6}]
    assert g.call_count == 2  # retried the 503 once


def test_slice_period_tails():
    full = [{"date": f"2024-01-{d:02d}", "close": float(d)} for d in range(1, 29)]
    assert prices.slice_period(full, "1mo") == full[-22:]
    assert prices.slice_period(full, "2y") == full  # None tail == full series


def test_get_price_history_csv_feeds_non_null_metrics():
    fake = [{"date": f"2024-01-{d:02d}", "close": 180 + d * 0.5} for d in range(1, 29)]
    with patch("utils.prices.fetch_price_rows_polygon", return_value=fake):
        csv = tools.get_price_history.invoke(
            {"ticker": "AAPL", "period": "1mo", "interval": "1d", "end_date": None})
    assert csv.splitlines()[0] == "Date,Close"
    vm = tools.compute_valuation_metrics(csv, "AAPL", "1mo")
    assert vm["trading_days"] > 0
    assert vm["cumulative_return"] != 0.0  # the bug was a null/zero return
