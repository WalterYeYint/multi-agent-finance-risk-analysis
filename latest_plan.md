# Plan — Multi-Agent Finance + Quant Research Platform

> **Vision:** evolve the current point-in-time stock analyzer into a hosted, multi-tenant **quant research platform** where the multi-agent system is one signal generator among many, evaluated honestly against baselines on historical data. AWS-hosted, freemium SaaS, positioned as a research/education tool (not investment advice).

> **Positioning rule (non-negotiable):** every output carries a "research tool, not investment advice" disclaimer. No recommendation-as-a-service. This shapes every product decision below.

---

## Phase 1 — Foundation (15–28 May 2026)

Goal: turn the current prototype into a hostable, multi-tenant service with run history and parallel agents.

| Date | Deliverable | Notes |
|---|---|---|
| **15.5** | RAG: Chroma → Postgres + pgvector | Single store for embeddings, filings metadata, run history. Run locally via docker-compose Postgres for now. |
| **15.5** | SEC EDGAR auto-ingest script | Replaces manual `data/filings/` drops. Cron-able. Stores raw PDFs in `s3://` (or local for now), parsed chunks in pgvector. |
| **16.5** | Parallelize `sentiment ‖ valuation ‖ fundamental` | All three only depend on `data` output. Use LangGraph parallel branches. Expect 2–3× wall-clock improvement. |
| **16.5** | Eval suite v0 | (a) schema-validity rate of agent JSON, (b) hallucinated-number detection (numbers in fundamental output must appear in retrieved chunks), (c) recommendation stability across 3 reruns of same ticker. |
| **17.5** | User accounts via Cognito (local dev: Cognito-emulated or Supabase) | Foundation for watchlists + tier gating. JWT in Flask. |
| **18.5** | Run history schema | `runs(id, user_id, ticker, mode, as_of, signals_json, recommendation, cost_usd, latency_ms)`. Every chain/debate run persists. |
| **19.5** | SSE / WebSocket progress streaming | Replace the 15-min blank-screen UX. Each agent emits "started/finished" events. |
| **20.5** | Frontend polish | Run history page, per-ticker timeline, "previous analyses" sidebar. |
| **22–23.5** | AWS deploy (Option B-lite, see Architecture) | API Gateway + Lambda + ECS Fargate worker + Aurora Serverless v2 + S3+CloudFront frontend. Stretch: Bedrock provider behind `MODEL_PROVIDER=bedrock`. |

**Phase 1 exit criteria:**
- A user can sign up, run an analysis, see it stream, view past analyses, and the report links to the exact filings cited.
- One-command deploy via Terraform or CDK. AWS bill < $90/mo idle.

---

## Phase 2 — Backtest engine + qfx integration (29 May – 26 Jun 2026)

Goal: prove (or disprove) that the multi-agent system has edge by running it over history alongside dumb baselines. This is the qfx integration.

### Step 1 — Signal abstraction (29 May – 5 Jun)

Refactor the chain + debate workflows behind a common interface so the backtester treats them as black boxes:

```python
class SignalGenerator(Protocol):
    name: str
    asset_class: Literal["equity", "fx"]
    def generate(self, ticker: str, as_of: date) -> Signal: ...

class Signal(BaseModel):
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    rationale: str
    sources: list[str]  # filing IDs, news URLs
    cost_usd: float
    latency_ms: int
```

Implementations to ship in Phase 2:
- `MultiAgentChainSignal` (wraps current chain workflow)
- `MultiAgentDebateSignal` (wraps current debate workflow)
- `BuyAndHoldSignal` (baseline)
- `SMACrossoverSignal` (50/200 — port from qfx)
- `MeanReversionSignal` (port from qfx)
- `BedrockMacroNewsSignal` (LLM-only, no agent debate — port from qfx)

### Step 2 — Backtest engine (6–12 Jun)

- Walk-forward only (no look-ahead). Engine iterates `as_of` over a date range, calls `signal.generate(ticker, as_of)`, simulates portfolio actions, records equity curve.
- Metrics: Sharpe, Sortino, max drawdown, win rate, Calmar, hit rate vs forward avg-close (current integration test becomes one assertion in a suite).
- Persist backtest runs to Postgres alongside live runs.
- Cache LLM outputs by `(signal_name, ticker, as_of)` so re-running a backtest is cheap.

### Step 3 — Dashboard + leaderboard (13–19 Jun)

- Equity-curve overlay: multi-agent debate vs SMA crossover vs buy-and-hold, picked tickers, picked years.
- Per-strategy stats card. A/B leaderboard.
- Per-recommendation deep-dive: click a trade → see the agents' arguments at that `as_of` → links to the cited filings.

### Step 4 — FX angle (20–26 Jun)

- Add OANDA demo-account ingestion Lambda → S3 parquet.
- One FX strategy (SMA crossover on EUR/USD) running through the same engine.
- **Not** porting the multi-agent system to FX — different fundamental data shape. The shared substrate is the engine + dashboard.

**Phase 2 exit criteria:**
- Public dashboard URL showing multi-agent debate vs 3+ baselines on 5 tickers over 3 years.
- Honest blog post draft: "Does a 6-agent debate beat buy-and-hold? Here's 3 years of walk-forward."

---

## Phase 3 — Monetize + launch (27 Jun – 24 Jul 2026)

### Step 1 — Billing (27 Jun – 3 Jul)

- Stripe Checkout + customer portal.
- Tiers gated by Cognito groups + Postgres `subscription` table:

| Tier | $/mo | Gates |
|---|---|---|
| Free | 0 | 3 tickers/day, Haiku/Ollama, watermarked PDF, 7-day history |
| Pro | 19 | Unlimited tickers, Sonnet, no watermark, watchlists w/ daily re-run, email alerts on recommendation flip |
| Power | 79 | API access (rate-limited), run backtests on your watchlist, agent-weight customization, Slack webhook |
| Enterprise | custom | White-label embed, SSO, compliance audit export |

### Step 2 — Watchlists + alerts (4–10 Jul)

- EventBridge cron → SQS → Fargate worker re-runs each user's watchlist nightly.
- SES email when recommendation flips (BUY→HOLD, HOLD→SELL, etc.).
- "Why did this change?" diff view in the UI.

### Step 3 — Public API + rate limits (11–17 Jul)

- API Gateway usage plans + API keys keyed to Stripe customer ID.
- OpenAPI spec, simple docs page, request examples.

### Step 4 — Launch (18–24 Jul)

- Blog post (the Phase 2 honest evaluation post).
- Show HN / r/algotrading / LinkedIn for AU job-hunt visibility.
- Affiliate links to IBKR / Stake (disclosed).
- Cost dashboard: per-user $-spent vs $-earned. Kill the free tier if unit economics break.

---

## Architecture — AWS Option B-lite

```
                  CloudFront + WAF
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
   S3: frontend    API Gateway        Cognito
                   + Lambda            (auth)
                        │
            ┌───────────┼────────────┐
            │           │            │
         SQS jobs   DynamoDB    Aurora Serverless v2
                    (hot cache) Postgres + pgvector
            │
   ECS Fargate workers (LangGraph chain/debate + backtest)
            │
   ┌────────┼─────────────┐
   │        │             │
Bedrock   S3 filings   yfinance/OANDA/Polygon
(Claude)  + parquet    egress

EventBridge cron → EDGAR pull, watchlist re-runs, backtest refresh
Observability: CloudWatch + LangSmith + Sentry
```

**Why this shape:**
- Long debate runs (multi-minute) bypass API Gateway's 29s ceiling via SQS+Fargate.
- pgvector colocates embeddings with run history → cheap joins for the dashboard.
- Bedrock removes the "host Ollama on a server" cost trap and gives the AU-portfolio AWS story.
- Aurora Serverless v2 idles at ~$45/mo floor — if that's too steep pre-revenue, swap to RDS t4g.micro until first paying user.

---

## Open questions / risks

- **Legal positioning.** Need a real disclaimer + ToS before charging. Possibly consult an AU/US fintech lawyer before Phase 3.
- **Backtest honesty trap.** It's easy to leak future info into walk-forward (using filings dated *after* `as_of`, news from after `as_of`). Phase 2 Step 2 needs strict `as_of` enforcement at every data source — yfinance, news, EDGAR. This is the single most likely failure mode.
- **Aurora floor cost.** If month-1 has zero paying users, $45/mo Aurora + Bedrock token spend may bleed. Mitigation: start on RDS t4g.micro, migrate when first $100 MRR.
- **Multi-agent vs baselines result.** If the multi-agent system loses to buy-and-hold, that's a great blog post but kills the "Power" tier value prop. Plan: lean monetization on **convenience + speed + watchlists**, not "our signal is better."
- **Scope creep into FX.** Phase 2 Step 4 is the smallest possible FX wedge. Resist building a full qfx system inside this repo — keep that as a separate `qfx` repo that imports the shared engine package, or defer FX entirely to Phase 4+.

---

## What this plan deliberately doesn't do

- Real-money trading (qfx hard rule applies).
- Selling signals as a service (regulatory minefield).
- Sponsored/paid ticker coverage (kills credibility).
- LangGraph Platform managed hosting (vendor lock-in; ECS Fargate is portable).
