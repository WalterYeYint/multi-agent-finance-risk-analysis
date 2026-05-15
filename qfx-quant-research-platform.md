---
title: qfx — Quantitative FX Research Platform — Idea
created: 2026-05-05
last_verified: 2026-05-05
status: idea
tags: [idea, project, fx, quant, trading, aws, research, portfolio]
---

# qfx — Quantitative FX Research Platform

## What "quant" means

Quantitative = math/stats/code-driven, not gut-feel. A quant trading platform uses backtesting, statistical models, and algorithmic strategies — the engineering side of trading. This idea is the **engineering / research / showcase** version of FX trading, NOT the "let Claude trade my money" version (see [[ideas/projects/claude-fx-trading-bot]] for why that one is parked).

## The motivation (Yang's actual itch)

- Test whether I can crack market patterns
- Improve programming + infrastructure + AWS skills
- Build a serious portfolio piece for AU job applications
- Combine AI/LLM + AWS + finance domain — rare and impressive combo
- NOT "make money trading" (index funds already winning that game at 7-10%/yr)

## Why this beats the "FX bot with real money" version

| | qfx (research platform) | Live FX bot |
|---|---|---|
| Capital at risk | $0 | Real money |
| Engineering complexity | Same | Same |
| Public showcase | ✅ GitHub + blog | ❌ Private |
| Portfolio value | High (AU job hunt) | None |
| Stress | Low | High |
| Learning value | High | High but blunted by stress |
| If strategy fails | Engineering still wins | Lose money |
| If strategy works | Validated → can graduate to real money later | Just proves luck |

## Architecture sketch

```
                    ┌──────────────────┐
                    │  OANDA / Polygon │ ← free APIs
                    │  Dukascopy       │ ← historical data
                    └────────┬─────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Lambda (cron)      │
                  │  data ingestion     │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │  S3 raw + Athena    │
                  │  DynamoDB state     │
                  └──────────┬──────────┘
                             │
            ┌────────────────┼─────────────────┐
            │                │                 │
   ┌────────▼─────┐  ┌───────▼────────┐  ┌────▼──────────┐
   │ Backtesting  │  │ Live paper     │  │ LLM signal    │
   │ engine       │  │ trading        │  │ (Bedrock)     │
   │ (Python)     │  │ (OANDA demo)   │  │               │
   └────────┬─────┘  └───────┬────────┘  └────┬──────────┘
            │                 │                │
            └─────────────────┼────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Next.js dashboard │
                    │  - Equity curve    │
                    │  - Win rate        │
                    │  - Strategy A/B    │
                    └────────────────────┘
```

## Components

### 1. Data ingestion
- Scheduled Lambda pulling FX OHLCV data (OANDA free API)
- Historical data backfill from Dukascopy or Polygon
- Storage: S3 (raw parquet) + DynamoDB (latest state)
- Tools to demonstrate: Lambda, EventBridge, S3, Athena, parquet

### 2. Backtesting engine (Python)
- Pluggable strategy interface
- Walk-forward analysis (avoid overfitting)
- Metrics: Sharpe ratio, Sortino, max drawdown, win rate, profit factor, Calmar ratio
- Output: HTML report + JSON for dashboard

### 3. Strategy library
Start simple, layer complexity:
- **Baseline**: Buy-and-hold (proves your benchmarks work)
- **Technical**: Moving average crossover, RSI mean reversion, breakout (well-known, easy to verify)
- **Statistical**: Pairs trading, cointegration, mean reversion
- **ML**: LSTM price prediction (likely fails — useful negative result)
- **LLM**: Claude analyzes daily macro news → BUY/SELL/HOLD signal with confidence

### 4. Paper trading runtime
- Connects to OANDA demo account (free, real-time data)
- Executes trades based on strategy signals
- Logs every decision: rationale, entry, exit, P&L
- Continuous live evaluation alongside backtests

### 5. Dashboard (Next.js)
- Equity curves per strategy
- Live position viewer
- A/B leaderboard between strategies
- Trade journal with rationale
- Public read-only mode for portfolio showcase

### 6. LLM integration angle
- Claude reads daily news (Reuters, Bloomberg headlines via free RSS)
- Outputs structured analysis: macro factors, sentiment, BUY/SELL/HOLD per major pair
- Stored as one signal among many — competes with technical strategies
- Honest evaluation: does LLM add edge? Likely no, but proving it is the point.

## Hard rules (non-negotiable)

1. **NO REAL MONEY for 12 months minimum.** If unable to follow this rule, don't start the project.
2. **All paper trading.** OANDA demo. Pretend money. Real engineering.
3. **Public from day 1** — GitHub repo open, blog post per milestone. Skin in the game.
4. **Honest reporting** — publish failed strategies too. The bad results are scientific.
5. **Track real costs**: AWS bill, data API, time spent. Be honest about ROI.
6. **Time-boxed**: 4 hours/week max. Weekend mornings. Not nightly.

## Why it's a strong portfolio piece for AU

- AWS at scale (Lambda, EventBridge, DynamoDB, S3, Athena, Bedrock)
- Python + ML + LLM integration
- Time-series analysis
- Financial domain (impresses Optiver / Citadel / Macquarie / Two Sigma)
- GitHub stars + blog SEO
- Demonstrates "ships things, not just talks"
- Different from 90% of bootcamp portfolios (CRUD apps)

## Why it might not work

- 8th project on Yang's stack — bandwidth competition is real
- Quant research is a rabbit hole — easy to spend 1000 hours, find nothing
- Even if interesting, 12-month-no-money rule means no validation of "does it actually work"
- AU companies care about delivered production systems, not personal research projects

## Bandwidth honest check

Already on plate:
1. AU move (priority)
2. Xperisus freelance (income)
3. OurValue (until May 31)
4. Kusala (planning, parked)
5. Mujō YouTube (idea, pre-launch)
6. claude-speak (active dev)
7. doujin-kakaku (idea)
8. Burmese-in-Japan info hub (idea, parked)
9. **qfx (this — idea, parked)**
10. Claude FX trading bot (idea, superseded by this)

This is **#9 in priority**. Realistic: park as documented idea now. Revisit when:
- AU move stabilizes (post-Sydney trip + offer / no-offer clarity)
- Or as 1 of 2 weekend-protected projects (alongside Kusala or Mujō)

## Next steps if pursuing

- [ ] Create GitHub repo `qfx` with README
- [ ] Open OANDA demo account (free, real-time)
- [ ] Pull 10 years of EUR/USD daily data, store as parquet
- [ ] Implement first backtest: buy-and-hold baseline
- [ ] Implement second backtest: 50/200 SMA crossover
- [ ] Compare both. Document results honestly.
- [ ] Then layer LLM signal as third strategy.

Each step is small + concrete. No commitment to "becoming a trader" — just an engineering project with finance flavor.

## Honest 1-year forecast

Most likely outcome:
- 6 months of weekend coding
- 3-5 strategies tested
- All underperform buy-and-hold
- LLM signal slightly worse than technical strategies
- Result: "I built a quant FX research platform, here's what I learned about why retail FX is hard" → blog post + GitHub repo with stars
- **Portfolio value: real, even with no profitable strategy.**

Less likely outcome:
- A specific quirky pattern (e.g. weekend gap trading) shows persistent edge
- Paper-trade it for 12 months
- THEN consider risking small real money
- If still profitable, scale slowly

Most likely + secondary scenario both are good engineering experiences.

## Related

- [[ideas/projects/_index]]
- [[ideas/projects/claude-fx-trading-bot]] — superseded by this (research-first framing instead of money-first)
- [[wiki/concepts/career]] — portfolio angle
