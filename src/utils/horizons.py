"""
Horizon presets — the single source of truth for what Short / Mid / Long mean.

The product surface no longer accepts a user-supplied `period`/`horizon_days`;
every pipeline run is parameterised by one of three baked-in horizon names.
The worker (cron refresh + on-demand jobs), the read API, and the new
`run_pipeline_for_horizon` entry point all import from here.

| Horizon | Lookback (price/news) | Forecast | Cache freshness |
|---------|-----------------------|----------|-----------------|
| SHORT   | 1 month               | 7 days   | 24 h            |
| MID     | 6 months              | 30 days  | 72 h (3 d)      |
| LONG    | 2 years               | 90 days  | 168 h (7 d)     |
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

HorizonName = Literal["SHORT", "MID", "LONG"]


class Horizon(BaseModel):
    """One horizon preset; immutable values plugged into State at run time."""
    name: HorizonName
    period: str            # yfinance lookback string — "1mo" / "6mo" / "2y"
    horizon_days: int      # forecast window the LLM agents reason over
    freshness_hours: int   # cached snapshot is "fresh" if newer than this


HORIZONS: dict[HorizonName, Horizon] = {
    "SHORT": Horizon(name="SHORT", period="1mo", horizon_days=7,  freshness_hours=24),
    "MID":   Horizon(name="MID",   period="6mo", horizon_days=30, freshness_hours=72),
    "LONG":  Horizon(name="LONG",  period="2y",  horizon_days=90, freshness_hours=168),
}


def get_horizon(name: str) -> Horizon:
    """Look up a horizon preset by name (case-insensitive). Raises on unknowns."""
    key = (name or "").upper()
    if key not in HORIZONS:
        raise ValueError(
            f"Unknown horizon {name!r}. Valid: {sorted(HORIZONS)}")
    return HORIZONS[key]  # type: ignore[index]
