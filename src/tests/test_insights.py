"""N2: public-analyzer insights — deterministic technical + valuation math.

Network (SEC XBRL) and LLM (analyst summary) paths are best-effort and excluded
here; these cover the pure computation that must be correct.
"""

import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # src/

from utils.insights import compute_technical_signals, compute_valuation_ratios  # noqa: E402


def _uptrend(n=500, start_price=100.0, drift=1.0008):
    d = date(2023, 7, 1)
    out, p = [], start_price
    for i in range(n):
        p *= drift
        out.append({"date": (d + timedelta(days=i)).isoformat(), "close": round(p, 2)})
    return out


def test_technical_uptrend_ordering_and_cross():
    t = compute_technical_signals(_uptrend())
    assert t.last_close > t.sma_50 > t.sma_200      # rising series
    assert t.trend == "golden_cross"
    assert t.return_1y and t.return_1y > 0
    assert t.pct_from_52w_high is not None and t.pct_from_52w_high <= 0
    assert t.pct_from_52w_low is not None and t.pct_from_52w_low >= 0


def test_technical_rsi_bounds():
    t = compute_technical_signals(_uptrend())
    assert 0 <= t.rsi_14 <= 100
    # a pure uptrend has no down days → RSI saturates high (overbought)
    assert t.rsi_zone == "overbought"


def test_technical_short_series_degrades():
    t = compute_technical_signals(_uptrend(n=20))
    assert t is not None
    assert t.last_close is not None
    assert t.sma_200 is None and t.return_1y is None   # not enough history


def test_technical_downtrend_death_cross():
    t = compute_technical_signals(_uptrend(drift=0.999))   # declining
    assert t.trend == "death_cross"
    assert t.return_1y is not None and t.return_1y < 0


def test_technical_empty_or_garbage_is_none():
    assert compute_technical_signals(None) is None
    assert compute_technical_signals([]) is None
    assert compute_technical_signals("not,a,valid,csv") is None


def test_technical_accepts_csv_string():
    csv = "Date,Close\n2024-01-01,100\n2024-02-01,110\n2024-03-01,120\n"
    t = compute_technical_signals(csv)
    assert t is not None and t.last_close == 120.0


_FIN = {"revenue": 100e9, "net_income": 25e9, "equity": 50e9, "liabilities": 40e9,
        "eps_diluted": 6.0, "shares_outstanding": 4e9, "fiscal_year": 2024}


def test_valuation_ratios_correct():
    v = compute_valuation_ratios(_FIN, latest_price=180.0)
    assert abs(v.pe_ratio - 30.0) < 0.01           # 180 / 6
    assert abs(v.market_cap - 720e9) < 1           # 180 * 4e9
    assert abs(v.ps_ratio - 7.2) < 0.01            # 720e9 / 100e9
    assert abs(v.net_margin - 0.25) < 1e-9
    assert abs(v.roe - 0.5) < 1e-9
    assert abs(v.debt_to_equity - 0.8) < 1e-9
    assert v.as_of_fy == 2024 and v.source


def test_valuation_negative_eps_omits_pe():
    v = compute_valuation_ratios({**_FIN, "eps_diluted": -2.0}, 180.0)
    assert v.pe_ratio is None
    assert any("P/E" in n for n in v.notes)


def test_valuation_missing_shares_omits_marketcap_but_keeps_margin():
    v = compute_valuation_ratios({**_FIN, "shares_outstanding": None}, 180.0)
    assert v.market_cap is None and v.ps_ratio is None
    assert v.roe is not None and v.net_margin is not None


def test_valuation_no_financials_is_none():
    assert compute_valuation_ratios(None, 180.0) is None


def test_valuation_all_inputs_missing_is_none():
    assert compute_valuation_ratios({"fiscal_year": 2024}, None) is None
