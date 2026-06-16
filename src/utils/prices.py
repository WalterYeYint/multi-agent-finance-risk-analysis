"""
Single source of truth for fetching a daily-close price series from Polygon
aggregates.

Both the Flask backend (`/api/price`) and the pipeline (`run_pipeline_for_horizon`,
which persists the series into the snapshot) import `fetch_price_series_polygon`
so the Polygon URL / response-parsing logic lives in exactly one place.

Polygon is used instead of yfinance on purpose: yfinance pulls in curl_cffi (a
native extension that crashes the backend container on python:slim images) and
Yahoo IP-blocks AWS datacenter ranges. Polygon is a plain HTTP call via
`requests`, so it works everywhere.

The function is failure-tolerant by contract: it returns `[]` (never raises)
when `POLYGON_API_KEY` is missing or the request fails, so callers render an
empty chart rather than a 500 / crash.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

# Slice a full ~2y daily series into per-period tails so the three horizons are
# consistent windows of one fetch. None = full series. Shared by the backend's
# /api/price slicing and the snapshot read path.
PERIOD_TAIL = {"1mo": 22, "6mo": 124, "2y": None}

# How far back to fetch when building the full series (covers the LONG 2y window
# with a little slack for non-trading days).
_FULL_LOOKBACK_DAYS = 365 * 2 + 7


def fetch_price_rows_polygon(ticker: str, start: date, end: date, *,
                             api_key: Optional[str] = None) -> list[dict]:
    """Daily close rows for `ticker` over the inclusive [start, end] date range
    from Polygon aggregates.

    Returns a list of {"date": "YYYY-MM-DD", "close": float}, oldest first.
    Returns [] (never raises) on missing key / non-200 / network error so the
    caller can fall back gracefully.
    """
    api_key = api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("⚠️  price fetch: POLYGON_API_KEY not set — returning empty series")
        return []

    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
    try:
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"⚠️  price fetch: Polygon {resp.status_code} for {ticker}: {resp.text[:200]}")
            return []
        results = (resp.json() or {}).get("results") or []
        rows: list[dict] = []
        for bar in results:
            close, ts = bar.get("c"), bar.get("t")
            if close is None or ts is None:
                continue
            try:
                close = float(close)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(close):
                continue
            d = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).date()
            rows.append({"date": d.isoformat(), "close": close})
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  price fetch failed for {ticker}: {e}")
        return []


def fetch_price_series_polygon(ticker: str, *, years: float = 2.0,
                               api_key: Optional[str] = None) -> list[dict]:
    """Daily close series for `ticker` over the trailing `years`, from Polygon.

    Thin wrapper over `fetch_price_rows_polygon` (today − years → today). Returns
    a list of {"date": "YYYY-MM-DD", "close": float}, oldest first; [] on failure.
    """
    today = date.today()
    start = today - timedelta(days=int(365 * years) + 7)
    return fetch_price_rows_polygon(ticker, start, today, api_key=api_key)


def slice_period(full: list[dict], period: str) -> list[dict]:
    """Slice a full series into a per-period tail using PERIOD_TAIL."""
    tail = PERIOD_TAIL.get(period)
    if tail is None or len(full) <= tail:
        return full
    return full[-tail:]
