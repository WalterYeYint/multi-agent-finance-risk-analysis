"""N2: public-analyzer insights.

Surfaces the kind of metrics you'd see on a public stock analyzer, computed from
data the pipeline already has or can fetch for free:

  • Technical & momentum  — derived from the Polygon daily-close series
                            (SMA 50/200, RSI-14, 52-week range, trailing returns).
  • Valuation ratios       — from SEC XBRL companyfacts (P/E, P/S, margins, ROE,
                            debt/equity) combined with the latest price.
  • Analyst-style summary  — a short, grounded LLM synthesis of the two.

Everything is best-effort: each section is computed independently and swallows
its own errors, returning None for any metric whose inputs are missing, so
insights can never fail a pipeline run. Computed in the worker (off the request
path) and persisted in the snapshot.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.schemas import (
    PublicInsights, TechnicalSignals, ValuationRatios,
)


# ------------------------------------------------------------------ helpers
def _finite(x: Any) -> Optional[float]:
    """Return a plain finite float, or None for NaN/inf/None/non-numeric."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    x = _finite(x)
    return round(x, n) if x is not None else None


def _prices_to_series(prices: Any) -> Optional[pd.Series]:
    """Coerce a [{"date","close"}, ...] list (or a Date,Close CSV string) into a
    date-indexed, ascending, deduped close Series. None if unusable."""
    if prices is None:
        return None
    try:
        if isinstance(prices, str):
            from io import StringIO
            df = pd.read_csv(StringIO(prices))
            df = df.rename(columns={"Date": "date", "Close": "close"})
        else:
            df = pd.DataFrame(list(prices))
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        df = df[df["close"] > 0]
        if df.empty:
            return None
        s = df.set_index("date")["close"]
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return None


def _return_over(s: pd.Series, days: int) -> Optional[float]:
    """Trailing return over the last `days` calendar days (uses the close at/just
    before the cutoff), as a fraction. None if not enough history."""
    if s.empty:
        return None
    cutoff = s.index[-1] - timedelta(days=days)
    past = s[s.index <= cutoff]
    if past.empty:
        return None
    p0, p1 = float(past.iloc[-1]), float(s.iloc[-1])
    return (p1 / p0 - 1.0) if p0 > 0 else None


def _rsi_14(s: pd.Series) -> Optional[float]:
    """Wilder's RSI(14) on the latest bar. None if <15 points."""
    if len(s) < 15:
        return None
    delta = s.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    if not math.isfinite(avg_gain) or not math.isfinite(avg_loss):
        return None
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ------------------------------------------------------------- technical
def compute_technical_signals(prices: Any) -> Optional[TechnicalSignals]:
    """Momentum/technical signals from a daily-close price series. Returns None
    only if the series is unusable; individual metrics degrade to None."""
    s = _prices_to_series(prices)
    if s is None or len(s) < 2:
        return None

    last = float(s.iloc[-1])
    sma50 = float(s.tail(50).mean()) if len(s) >= 50 else None
    sma200 = float(s.tail(200).mean()) if len(s) >= 200 else None

    def pct_vs(sma):
        return (last / sma - 1.0) if sma else None

    # Trend from the SMA relationship (golden/death cross), else price-vs-SMA200.
    trend = None
    if sma50 is not None and sma200 is not None:
        trend = "golden_cross" if sma50 >= sma200 else "death_cross"
    elif sma200 is not None:
        trend = "uptrend" if last >= sma200 else "downtrend"
    elif sma50 is not None:
        trend = "uptrend" if last >= sma50 else "downtrend"

    rsi = _rsi_14(s)
    rsi_zone = None
    if rsi is not None:
        rsi_zone = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"

    win_52w = s[s.index >= (s.index[-1] - timedelta(days=365))]
    hi = float(win_52w.max()) if not win_52w.empty else None
    lo = float(win_52w.min()) if not win_52w.empty else None

    r1m = _return_over(s, 30)
    r3m = _return_over(s, 91)
    r6m = _return_over(s, 182)
    r1y = _return_over(s, 365)

    # Momentum bucket off the 3-month return (falls back to the longest we have).
    ref = next((r for r in (r3m, r6m, r1m, r1y) if r is not None), None)
    momentum = None
    if ref is not None:
        momentum = ("strong_up" if ref >= 0.20 else "up" if ref >= 0.05
                    else "flat" if ref > -0.05 else "down" if ref > -0.20
                    else "strong_down")

    return TechnicalSignals(
        as_of=s.index[-1].date().isoformat(),
        last_close=_round(last, 4),
        sma_50=_round(sma50, 4),
        sma_200=_round(sma200, 4),
        pct_vs_sma_50=_round(pct_vs(sma50)),
        pct_vs_sma_200=_round(pct_vs(sma200)),
        trend=trend,
        rsi_14=_round(rsi, 2),
        rsi_zone=rsi_zone,
        high_52w=_round(hi, 4),
        low_52w=_round(lo, 4),
        pct_from_52w_high=_round((last / hi - 1.0) if hi else None),
        pct_from_52w_low=_round((last / lo - 1.0) if lo else None),
        return_1m=_round(r1m), return_3m=_round(r3m),
        return_6m=_round(r6m), return_1y=_round(r1y),
        momentum=momentum,
    )


# ------------------------------------------------------------- valuation
# us-gaap concept candidates (companies tag the same figure differently); tried
# in order, first hit wins.
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues", "SalesRevenueNet",
]
_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
_EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
_LIABILITIES_TAGS = ["Liabilities"]
_EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]


def _latest_fy_fact(facts: Dict[str, Any], tags: List[str], unit: str):
    """From a companyfacts 'us-gaap'/'dei' block, return (value, fiscal_year) for
    the most recent annual (FY / 10-K, 20-F) data point among `tags`, or (None, None)."""
    best_val = best_fy = best_end = None
    for tag in tags:
        node = facts.get(tag)
        if not node:
            continue
        for u in node.get("units", {}).get(unit, []):
            if u.get("fp") != "FY":
                continue
            if u.get("form") not in ("10-K", "10-K/A", "20-F", "20-F/A"):
                continue
            end = u.get("end")
            val = u.get("val")
            if end is None or val is None:
                continue
            if best_end is None or end > best_end:
                best_end, best_val, best_fy = end, val, u.get("fy")
        if best_val is not None:
            break  # first tag that yielded data wins
    return best_val, best_fy


def fetch_key_financials(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch key annual financials from SEC XBRL companyfacts. Best-effort:
    returns a dict of whatever concepts resolved (may be partial), or None on
    any failure (no CIK, network error, unexpected shape)."""
    try:
        from utils import edgar_ingest as E
        cik = E.load_cik_map().get(ticker.upper())
        if not cik:
            return None
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        data = E._get(url, as_json=True)
        if not isinstance(data, dict):
            return None
        gaap = (data.get("facts") or {}).get("us-gaap", {})
        dei = (data.get("facts") or {}).get("dei", {})

        revenue, rev_fy = _latest_fy_fact(gaap, _REVENUE_TAGS, "USD")
        net_income, ni_fy = _latest_fy_fact(gaap, _NET_INCOME_TAGS, "USD")
        equity, _ = _latest_fy_fact(gaap, _EQUITY_TAGS, "USD")
        liabilities, _ = _latest_fy_fact(gaap, _LIABILITIES_TAGS, "USD")
        eps, _ = _latest_fy_fact(gaap, _EPS_TAGS, "USD/shares")

        # Shares outstanding: prefer the dei cover-page instantaneous count.
        shares = None
        for u in dei.get("EntityCommonStockSharesOutstanding", {}).get(
                "units", {}).get("shares", []):
            if u.get("val"):
                if shares is None or (u.get("end") or "") > (shares[1] or ""):
                    shares = (u["val"], u.get("end"))
        shares_val = shares[0] if shares else None

        return {
            "revenue": _finite(revenue),
            "net_income": _finite(net_income),
            "equity": _finite(equity),
            "liabilities": _finite(liabilities),
            "eps_diluted": _finite(eps),
            "shares_outstanding": _finite(shares_val),
            "fiscal_year": rev_fy or ni_fy,
            "company": (data.get("entityName") or None),
        }
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  insights: could not fetch XBRL financials for {ticker}: {e}")
        return None


def compute_valuation_ratios(fin: Optional[Dict[str, Any]],
                             latest_price: Optional[float]) -> Optional[ValuationRatios]:
    """Combine XBRL fundamentals with the latest price into valuation ratios.
    Best-effort: omit any ratio whose inputs are missing and note why."""
    if not fin:
        return None
    price = _finite(latest_price)
    revenue = fin.get("revenue")
    net_income = fin.get("net_income")
    equity = fin.get("equity")
    liabilities = fin.get("liabilities")
    eps = fin.get("eps_diluted")
    shares = fin.get("shares_outstanding")

    notes: List[str] = []
    market_cap = (price * shares) if (price and shares) else None
    pe = (price / eps) if (price and eps and eps > 0) else None
    if pe is None and eps is not None and eps <= 0:
        notes.append("P/E omitted (non-positive EPS)")
    ps = (market_cap / revenue) if (market_cap and revenue and revenue > 0) else None
    net_margin = (net_income / revenue) if (net_income is not None and revenue) else None
    roe = (net_income / equity) if (net_income is not None and equity and equity > 0) else None
    d_to_e = (liabilities / equity) if (liabilities is not None and equity and equity > 0) else None

    if market_cap is None:
        notes.append("market cap / P/S omitted (shares outstanding unavailable)")

    have_any = any(v is not None for v in (pe, ps, net_margin, roe, d_to_e, market_cap))
    if not have_any:
        return None

    return ValuationRatios(
        as_of_fy=fin.get("fiscal_year"),
        price=_round(price, 4),
        market_cap=_round(market_cap, 0),
        eps_diluted=_round(eps, 4),
        revenue=_round(revenue, 0),
        net_income=_round(net_income, 0),
        pe_ratio=_round(pe, 2),
        ps_ratio=_round(ps, 2),
        net_margin=_round(net_margin, 4),
        roe=_round(roe, 4),
        debt_to_equity=_round(d_to_e, 2),
        source="SEC XBRL companyfacts",
        notes=notes,
    )


# --------------------------------------------------------- analyst summary
def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _fmt_num(x: Optional[float]) -> str:
    return f"{x:,.2f}" if isinstance(x, (int, float)) else "n/a"


def analyst_summary(ticker: str,
                    technical: Optional[TechnicalSignals],
                    valuation: Optional[ValuationRatios],
                    recommendation: str = "") -> str:
    """Short, grounded LLM synthesis of the computed signals. Returns "" on any
    failure (incl. a degraded/mock LLM) so it never blocks the pipeline."""
    try:
        from utils.config import get_llm, is_degraded_llm
        if is_degraded_llm():
            return ""
        t = technical.model_dump() if technical else {}
        v = valuation.model_dump() if valuation else {}
        facts = []
        if technical:
            facts.append(
                f"Technical: trend={t.get('trend')}, RSI14={t.get('rsi_14')} "
                f"({t.get('rsi_zone')}), 1m/3m/6m/1y returns="
                f"{_fmt_pct(t.get('return_1m'))}/{_fmt_pct(t.get('return_3m'))}/"
                f"{_fmt_pct(t.get('return_6m'))}/{_fmt_pct(t.get('return_1y'))}, "
                f"{_fmt_pct(t.get('pct_from_52w_high'))} from 52w high.")
        if valuation:
            facts.append(
                f"Valuation (FY{v.get('as_of_fy')}): P/E={_fmt_num(v.get('pe_ratio'))}, "
                f"P/S={_fmt_num(v.get('ps_ratio'))}, net margin={_fmt_pct(v.get('net_margin'))}, "
                f"ROE={_fmt_pct(v.get('roe'))}, debt/equity={_fmt_num(v.get('debt_to_equity'))}.")
        if not facts:
            return ""
        block = "\n".join(facts)
        rec = f"\nThe multi-agent recommendation is: {recommendation}." if recommendation else ""

        prompt = (
            f"You are a concise equity analyst. Using ONLY the metrics below for "
            f"{ticker}, write 3-4 plain-English sentences on what they imply about "
            f"valuation and momentum. Do not invent numbers not shown. End with "
            f"'Not financial advice.'\n\n{block}{rec}"
        )
        llm = get_llm(max_tokens=400)
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        return text.strip()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  insights: analyst summary skipped for {ticker}: {e}")
        return ""


# ------------------------------------------------------------- orchestration
def build_insights(ticker: str, *, prices: Any = None,
                   price_csv: Optional[str] = None,
                   recommendation: str = "") -> PublicInsights:
    """Assemble PublicInsights for a ticker. `prices` (the persisted 2y series)
    is preferred for technicals; `price_csv` (the horizon window) is a fallback.
    Each section is independent and best-effort."""
    technical = None
    try:
        technical = compute_technical_signals(prices if prices else price_csv)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  insights: technical signals failed for {ticker}: {e}")

    latest_price = technical.last_close if technical else None
    valuation = None
    try:
        valuation = compute_valuation_ratios(fetch_key_financials(ticker), latest_price)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  insights: valuation ratios failed for {ticker}: {e}")

    summary = analyst_summary(ticker, technical, valuation, recommendation)

    return PublicInsights(
        ticker=ticker.upper(),
        technical=technical,
        valuation=valuation,
        analyst_summary=summary,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
