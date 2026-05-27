# Plan — Multi-Agent Finance + Quant Research Platform

> **Vision:** evolve the current point-in-time stock analyzer into a hosted,
> AWS-deployed **research database** of pre-computed per-ticker, per-horizon
> snapshots. Users browse a curated-but-growing list of tickers; clicking a
> ticker shows three horizon views (Short / Mid / Long) backed by snapshots a
> worker keeps warm. The multi-agent system is one signal generator among many,
> evaluated honestly against baselines on historical data (Phase 2).

> **Positioning rule (non-negotiable):** every output carries a "research tool,
> not investment advice" disclaimer. No recommendation-as-a-service.

---

## Product model (decided 20 May 2026)

- **No user-supplied `period`.** Every ticker has three baked-in horizon presets:
  | Horizon | Lookback (price/news) | Forecast | Cache freshness target |
  |---|---|---|---|
  | **Short** | 1 month | 7 days | 24 h |
  | **Mid** | 6 months | 30 days | 3 days |
  | **Long** | 2 years | 90 days | 7 days |
- **Snapshots are the cache.** A `snapshots` table holds the latest pipeline
  output per `(ticker, horizon)`. Anything ever requested ends up cached.
- **Worker keeps things warm.** A cron'd job scans for snapshots older than
  their freshness target and refreshes them. The list of warm tickers
  self-curates: it's whatever has ever been requested, minus eventual
  cold-aging.
- **On-demand for new tickers.** Click an uncached ticker → API enqueues a job,
  worker runs the pipeline, snapshot lands; client polls a JSON status endpoint
  (no SSE / WebSocket — plain HTTP polling is sufficient).
- **No auth in v1.** Cognito is deferred to Phase 3 (alerts, API keys,
  custom-ticker requests) — browsing is anonymous.

---

## Phase 1 — Foundation (15–28 May 2026)

| Date | Deliverable | Status |
|---|---|---|
| **15.5** | RAG: Chroma → Postgres + pgvector | ✅ Done |
| **15.5** | SEC EDGAR auto-ingest script | ✅ Done |
| **16.5** | Parallelize `sentiment ‖ valuation ‖ fundamental` | ✅ Done |
| **16.5** | Eval suite v0 (schema validity, number grounding, recommendation stability) | ✅ Done |
| **20–21.5** | Snapshots table + horizon presets + refactor pipeline runner | Next |
| **22.5** | Worker service (scheduled refresh + on-demand jobs) | |
| **23.5** | Read API (no auth) + polling status endpoint | |
| **24–25.5** | Frontend pivot (ticker list → per-ticker 3-tab + history) | |
| **26–27.5** | AWS deploy | |
| **28.5** | Buffer / polish | |

### 20–21.5 — Snapshots schema + pipeline-by-horizon

- New `snapshots(ticker, horizon, generated_at, sentiment_json, fundamental_json,
  valuation_json, metrics_json, debate_json, report_markdown, cost_usd,
  latency_ms)`; primary key on `(ticker, horizon)` for latest, plus a separate
  table or non-deleted history rows for the time series.
- New `jobs(id, ticker, horizon, status, requested_at, started_at, finished_at,
  error)` for the on-demand path and worker coordination.
- A `HORIZONS` constant module (Short/Mid/Long → lookback period, forecast days,
  freshness target). The agent code that currently reads `state.period` /
  `state.horizon_days` continues to work — the *caller* (worker / API) just
  feeds the preset values instead of accepting them from the user.

### 22.5 — Worker

- One Fargate-runnable Python process (locally: `python -m src.worker`).
  Continuously polls `jobs` for queued work and `snapshots` for rows past
  their horizon's freshness target. Runs the chain + debate graphs for each
  `(ticker, horizon)`, persists the snapshot, writes `cost_usd` / `latency_ms`.
- Scheduled refresh = a cron entry that inserts "refresh stale" jobs at a fixed
  cadence (e.g. every 2 h Short, every 12 h Mid, daily Long). On-demand =
  API inserts a job row, the same worker picks it up within seconds.

### 23.5 — Read API + polling

- `GET /api/tickers` — list of cached/known tickers with latest summary per
  horizon.
- `GET /api/snapshot/{ticker}/{horizon}` — returns the latest snapshot if
  fresh; otherwise inserts a job and returns `{status: "running", job_id}`.
- `GET /api/snapshot/{ticker}/{horizon}/history?days=N` — timeseries of past
  snapshots (recommendation flips, risk metric drift). Powers the "risk
  history" chart.
- `GET /api/jobs/{id}` — `{status: "queued" | "running" | "ready" | "failed",
  progress?: "fundamental_agent..."}` for the client poller.

### 24–25.5 — Frontend pivot

- Landing page: ticker list, summary card per ticker showing
  Short-horizon recommendation + last-updated.
- Per-ticker page: three tabs (Short / Mid / Long), each showing the snapshot
  fields + an inline "risk history" sparkline. Cache-miss state: spinner +
  progress text + polling loop (~3s) until status flips to `ready`.
- Drop the old "ticker + period + interval" form.

### 26–27.5 — AWS deploy

See the Architecture section below. Target: < $80/mo idle.

**Phase 1 exit criteria**
- Any visitor (no login) can browse the ticker list and view three-horizon
  snapshots. A new ticker request resolves to a populated snapshot within
  pipeline runtime + polling cadence.
- One-command deploy. AWS bill measured and tracked.

---

## Phase 2 — Backtest engine + qfx integration (29 May – 26 Jun 2026)

Goal: prove (or disprove) the multi-agent system has edge, by running it over
history alongside dumb baselines. The Phase 1 worker is structurally close to a
backtester — it's already "run the pipeline at an `as_of` point and persist the
result." Reuse heavily.

### Step 1 — Signal abstraction (29 May – 5 Jun)

```python
class SignalGenerator(Protocol):
    name: str
    asset_class: Literal["equity", "fx"]
    def generate(self, ticker: str, as_of: date, horizon: Horizon) -> Signal: ...

class Signal(BaseModel):
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    rationale: str
    sources: list[str]   # filing IDs, news URLs
    cost_usd: float
    latency_ms: int
```

Implementations: `MultiAgentChainSignal`, `MultiAgentDebateSignal`,
`BuyAndHoldSignal`, `SMACrossoverSignal`, `MeanReversionSignal`,
`BedrockMacroNewsSignal`.

### Step 2 — Backtest engine (6–12 Jun)

- Walk-forward only (no look-ahead). Engine iterates `as_of` over a date range,
  calls `signal.generate(ticker, as_of, horizon)`, simulates portfolio actions,
  records equity curve.
- Strict `as_of` enforcement on every data source (yfinance, news, EDGAR).
- Metrics: Sharpe, Sortino, max drawdown, win rate, Calmar, hit-rate.
- Reuse the Phase 1 snapshots schema: each backtest sample = one snapshot row
  with a back-dated `as_of`. Cache by `(signal_name, ticker, as_of, horizon)`
  so re-running a backtest is cheap.

### Step 3 — Dashboard + leaderboard (13–19 Jun)

- Equity-curve overlay: multi-agent debate vs. SMA crossover vs. buy-and-hold.
- A/B leaderboard per strategy.
- Per-recommendation deep-dive: click a trade → agents' arguments at that `as_of`
  → linked filings.

### Step 4 — FX wedge (20–26 Jun)

- OANDA demo ingestion Lambda → S3 parquet.
- One FX strategy (SMA crossover on EUR/USD) through the same engine.
- Not porting the multi-agent system to FX — different fundamental shape.

**Phase 2 exit criteria**
- Public dashboard showing multi-agent debate vs. 3+ baselines on the curated
  ticker list over 3 years.
- Honest blog post: "Does a 6-agent debate beat buy-and-hold?"

---

## Phase 3 — Monetize + launch (27 Jun – 24 Jul 2026)

Without per-click pipeline runs to meter, monetization shifts to **value-add
features**, not "unlimited usage." Auth (Cognito) returns here.

### Tier structure

| Tier | $/mo | Gates |
|---|---|---|
| **Free** | 0 | Browse the curated list, see snapshots, see history (read-only, no login) |
| **Pro** | 19 | Add custom tickers to the list, email/Slack alerts on recommendation flips, daily digest, no PDF watermark |
| **Power** | 79 | API access (rate-limited), backtests on your watchlist, agent-weight customization |
| **Enterprise** | custom | White-label embed, SSO, audit-trail export |

### Step 1 — Auth + billing (27 Jun – 3 Jul)
Cognito for auth (gated to login-required endpoints), Stripe Checkout + customer
portal for tiering.

### Step 2 — Alerts (4–10 Jul)
EventBridge cron → diff today's snapshot vs. yesterday's → if `recommendation`
or `risk_flags` changed for a user's tracked ticker → SES email. "Why did this
change?" diff view links the two snapshots.

### Step 3 — Public API (11–17 Jul)
API Gateway usage plans + API keys per Stripe customer. OpenAPI spec.

### Step 4 — Launch (18–24 Jul)
Phase 2 blog post → Show HN / r/algotrading / LinkedIn. Affiliate links to
IBKR / Stake (disclosed). Per-user cost dashboard; kill Free if unit economics break.

---

## Architecture — AWS Option B-lite (revised)

```
                  CloudFront + WAF
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
   S3: frontend    API Gateway       (Cognito — Phase 3 only)
                   + Lambda
                        │
            ┌───────────┴────────────┐
            │                        │
      DynamoDB / cache          Aurora Serverless v2
      (hot snapshot index)      Postgres + pgvector
                                  - snapshots
                                  - snapshots_history
                                  - jobs
                                  - filings + filing_chunks
                                  - (Phase 3) users, subs, alerts
                        │
            ┌───────────┴───────────────────────┐
            │                                   │
      ECS Fargate worker                  EventBridge cron
       (continuous; polls jobs +           - per-horizon refresh
        stale snapshots; runs chain         - EDGAR daily pull
        + debate; writes snapshot)          - (Phase 3) alert diffs
            │
   ┌────────┼─────────────┐
   │        │             │
 Bedrock  S3 raw         yfinance / Polygon / OANDA
 (Claude)  filings +     egress
           parquet

Observability: CloudWatch + LangSmith + Sentry
```

**Why this shape**
- **Snapshots table is the read store.** API endpoints are simple SELECTs.
  Cache-miss path is the only place the pipeline runs synchronously (for the
  client); everything else is the worker.
- **No SSE / WebSocket.** Client polls a JSON status endpoint — much cheaper to
  build and deploy than streaming.
- **No Cognito in v1.** API Gateway routes are public; cuts deploy complexity
  significantly.
- **Worker IS the backtester.** Phase 2 reuses the same pipeline-by-`as_of`
  shape — just with historical `as_of` and the snapshot stored under the
  backtest's signal name.
- Aurora Serverless v2 idles at ~$45/mo floor — swap to RDS `t4g.micro`
  (~$15/mo) until first paying user if needed.

---

## Open questions / risks

- **Cache-miss UX.** First user to click a new ticker waits ~5 min staring at a
  polling spinner. Mitigations: keep the popular set well-covered so misses are
  rare; show a clear "this analysis is being generated; usually takes 3–5
  minutes" copy + the visible agent-progress text from the polling endpoint.
- **Worker compute budget.** With ~100 tickers × 3 horizons × the per-horizon
  refresh cadence, expect tens of pipeline runs per day. Bedrock / OpenAI cost
  scales linearly. Solution if it bleeds: tighter freshness windows for Long,
  cheaper model for Long-horizon runs, or cap the warm set.
- **No auth = no per-user metering for v1.** Public API has to be rate-limited
  by IP / WAF, not by user. Phase 3 is when Cognito returns; until then there's
  no Pro tier.
- **Backtest honesty trap.** Easy to leak future info into walk-forward
  (filings dated after `as_of`, post-`as_of` news). Phase 2 must enforce
  `as_of` at every data source — most likely failure mode.
- **If multi-agent loses to buy-and-hold** in Phase 2 backtests, that's the
  honest result + a great blog post, but it kills the "Power tier =
  better signals" pitch. Monetization leans on convenience / alerts / custom
  tickers, not "our signals are better."

---

## What this plan deliberately doesn't do

- Real-money trading.
- Selling signals as a service (regulatory minefield).
- Sponsored ticker coverage.
- User-typed `period` parameter (replaced by the three horizon presets).
- SSE / WebSocket streaming (polling is enough).
- Auth in v1 (deferred to Phase 3 with monetization).
