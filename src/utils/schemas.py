from typing import List, Literal, Dict, Any, Optional, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, BeforeValidator
from collections import defaultdict


def _coerce_str_list(value: Any) -> Any:
    """
    Flatten an LLM-produced list field into a list of plain strings.

    Different models emit list items in different shapes: GPT-4o returns plain
    strings, while smaller local models (e.g. llama3.1) often return per-item
    dicts like {"summary": ..., "potential_market_impact": ...}. To keep schema
    validation model-agnostic, a dict item is collapsed to its values joined by
    "; ", and any other non-string item is stringified. A plain list of strings
    passes through unchanged.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return value  # let pydantic raise its normal type error
    coerced: List[str] = []
    for item in value:
        if isinstance(item, str):
            coerced.append(item)
        elif isinstance(item, dict):
            coerced.append("; ".join(str(v) for v in item.values()))
        else:
            coerced.append(str(item))
    return coerced


# A List[str] field that tolerates model-dependent output shape (see above).
StrList = Annotated[List[str], BeforeValidator(_coerce_str_list)]


class MarketData(BaseModel):
    ticker: str
    period: str
    interval: str
    price_csv: str


class NewsItem(BaseModel):
    date: str
    headline: str
    sentiment: Literal["positive", "neutral", "negative"]
    content: str


class NewsBundle(BaseModel):
    ticker: str
    window_days: int
    items: List[NewsItem] = Field(default_factory=list)


class RiskMetrics(BaseModel):
    ticker: str
    horizon_days: int
    annual_vol: float
    max_drawdown: float
    daily_var_95: float
    sharpe_like: Optional[float] = None
    notes: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class SentimentSummary(BaseModel):
    ticker: str = ""
    news_items_analyzed: int = 0
    overall_sentiment: str = ""
    confidence_score: float = 0
    summary: str = ""
    investment_recommendation: str = ""
    key_insights: StrList = []
    methodology: str = ""


class SentimentExtract(BaseModel):
    """
    Output schema for the sentiment agent's structured-output (structuring)
    step. Every field is REQUIRED (no defaults) — a field with a default is
    optional in the generated JSON schema, and smaller local models simply
    skip optional fields, leaving them empty. The agent maps this into the
    storage schema (SentimentSummary), adding the system-owned fields.
    """
    overall_sentiment: Literal["bullish", "bearish", "neutral"]
    confidence_score: float
    summary: str
    investment_recommendation: str
    key_insights: StrList


class ValuationMetrics(BaseModel):
    ticker: str
    analysis_period: str
    trading_days: int
    cumulative_return: float
    annualized_return: float
    daily_volatility: float
    annualized_volatility: float
    price_trend: Literal["upward", "downward", "sideways"]
    volatility_regime: Literal["low", "medium", "high"]
    valuation_insights: StrList = Field(default_factory=list)
    trend_analysis: str
    risk_assessment: str
    methodology: str = "Computational analysis with 252 trading days assumption"


class RiskReport(BaseModel):
    ticker: str
    as_of: str
    summary: str
    key_findings: List[str]
    metrics_table: Dict[str, Any]
    risk_flags: List[str]
    methodology: str
    markdown_report: str


class FundamentalAnalysis(BaseModel):
    ticker: str = ""
    filing_type: str = ""  # "10-K" or "10-Q"
    filing_date: str = ""
    analysis_date: str = ""
    executive_summary: str = ""
    key_financial_metrics: StrList = []
    business_highlights: StrList = []
    risk_factors: StrList = []
    competitive_position: str = ""
    growth_prospects: str = ""
    financial_health_score: float = 0
    investment_thesis: str = ""
    concerns_and_risks: StrList = []
    methodology: str = "RAG-enhanced 10-K/10-Q document analysis"


class FundamentalExtract(BaseModel):
    """
    Output schema for the fundamental agent's structured-output step. Every
    field is REQUIRED (see SentimentExtract for the rationale). The agent maps
    this into the storage schema (FundamentalAnalysis), adding the system-owned
    fields (ticker, filing_type, filing_date, analysis_date, methodology).
    """
    executive_summary: str
    key_financial_metrics: StrList
    business_highlights: StrList
    risk_factors: StrList
    competitive_position: str
    growth_prospects: str
    financial_health_score: float
    investment_thesis: str
    concerns_and_risks: StrList

# Debate
class DebateReport(BaseModel):
    agent_list: List[str] = []
    agent_max_turn: int = 3
    agent_turn_count : Dict[str, int] = None
    agent_arguments: Dict[str, List[str]] = defaultdict(list)  # key: agent name, value: list of arguments made
    preliminary_report: str = ""
    consensus_summary: str = ""
    terminated: str = ""
    markdown_report: str = ""


# --- N2: public-analyzer insights -------------------------------------------
# Public-stock-analyzer-style metrics surfaced alongside the multi-agent report.
# All fields Optional/defaulted: any metric whose inputs are unavailable is left
# None (the pipeline computes best-effort and never fails on a missing input).

class TechnicalSignals(BaseModel):
    """Price-derived momentum/technical signals (from the Polygon daily series)."""
    as_of: Optional[str] = None
    last_close: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    pct_vs_sma_50: Optional[float] = None      # % above/below the 50-day SMA
    pct_vs_sma_200: Optional[float] = None
    trend: Optional[str] = None                # golden_cross | death_cross | uptrend | downtrend | neutral
    rsi_14: Optional[float] = None
    rsi_zone: Optional[str] = None             # overbought | oversold | neutral
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = None  # negative = below the 52w high
    pct_from_52w_low: Optional[float] = None
    return_1m: Optional[float] = None          # trailing returns, as fractions
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    return_1y: Optional[float] = None
    momentum: Optional[str] = None             # strong_up | up | flat | down | strong_down


class ValuationRatios(BaseModel):
    """Fundamental valuation ratios from SEC XBRL (companyfacts) + latest price."""
    as_of_fy: Optional[int] = None             # fiscal year the fundamentals are from
    currency: str = "USD"
    price: Optional[float] = None
    market_cap: Optional[float] = None
    eps_diluted: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    pe_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    net_margin: Optional[float] = None         # net income / revenue
    roe: Optional[float] = None                # net income / shareholders' equity
    debt_to_equity: Optional[float] = None     # total liabilities / equity (approx)
    source: Optional[str] = None               # e.g. "SEC XBRL companyfacts", else None
    notes: StrList = Field(default_factory=list)  # e.g. which ratios were omitted + why


class PublicInsights(BaseModel):
    ticker: str = ""
    technical: Optional[TechnicalSignals] = None
    valuation: Optional[ValuationRatios] = None
    analyst_summary: str = ""                   # LLM plain-English synthesis (grounded on the above)
    generated_at: Optional[str] = None