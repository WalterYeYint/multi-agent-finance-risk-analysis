# ROADMAP — multi-agent-finance-risk-analysis

Plan to a **stable v1.0** release. "Bug-free" isn't a real milestone (software is
never provably bug-free); the target is a *release candidate*: known issues
closed, evals passing, monitoring in place so new bugs surface fast.

_Last updated: 2026-06-16_

---

## Known backlog (~a dozen items)

| # | Type | Item | Status |
|---|------|------|--------|
| 1 | 🔴 Bug | ~~**Valuation/risk still use yfinance** (`src/utils/tools.py:get_price_history`) → crashes / IP-blocked on AWS → null `cumulative_return` in snapshots.~~ **Fixed** — now sources from the shared `utils/prices.py` Polygon helper; yfinance demoted to a lazy local-only fallback. | ✅ |
| 2 | 🔴 Verify | Today's changes (prices-in-snapshot, EDGAR auto-ingest, SSL `prepare_threshold` fix, duplicate-job fix) **not yet end-to-end tested on AWS**. | ☐ |
| 3 | 🟠 Ops | Worker task def must actually have `POLYGON_API_KEY` + `SEC_USER_AGENT` populated. | ☐ |
| 4 | 🟠 Tests | No automated tests for new paths (`ensure_filings`, prices persistence, dedup race, `prepare_threshold`). Integration suite is only 5 fixtures. | ☐ |
| 5 | 🟠 Reliability | No retry/backoff on transient LLM/Polygon/EDGAR failures; failed jobs aren't surfaced or alerted. | ☐ |
| 6 | 🟠 Scale | Single serial worker is a throughput ceiling (mitigation: run N workers — `claim_next_job` already uses `FOR UPDATE SKIP LOCKED`); DB connection pooling needs a real answer. | ☐ |
| 7 | 🟡 Quality | Hallucination guardrail — the eval's number-grounding signal has no enforcement. | ☐ |
| 8 | 🟡 UX | Writer is a fixed template; no failed-job state on the frontend. | ☐ |
| 9 | 🟡 Security | API has no auth / rate-limiting / ticker input validation if exposed publicly. | ☐ |
| 10 | 🟡 Debate | Debate effectiveness is **unmeasured** (see "Debate graph" below). | ☐ |
| 11 | 🟢 Hygiene | Duplicate AAPL snapshots already in the DB from the pre-fix race. | ☐ |
| 12 | 🟡 Observability | No prod metrics/tracing (LangSmith optional/off). | ☐ |

**Blockers: #1 and #2.** The rest is hardening.

### Recently fixed (this cycle)
- ✅ **Valuation/risk price source moved off yfinance** — `get_price_history` now fetches daily closes from Polygon via `utils/prices.py:fetch_price_rows_polygon` (date-range fetch), builds a `Date,Close` CSV that `compute_valuation_metrics` / `compute_risk` consume unchanged. yfinance is a lazy fallback (local/offline only), so curl_cffi never loads on the AWS path. Fixes null `cumulative_return`. **Requires `POLYGON_API_KEY` on the worker.**
- ✅ Price series persisted in `snapshots.prices` (worker fetches Polygon once; `/api/price` reads from DB, Polygon = fallback).
- ✅ SEC EDGAR auto-ingest: on-demand per-ticker in the pipeline + weekly worker sweep (idempotent by accession).
- ✅ SSL "connection is lost" during filing ingestion behind the Supabase transaction pooler (pgbouncer), in two parts: (a) disabled psycopg3 auto-prepared-statements (`prepare_threshold=None` in `utils/db.py`); (b) replaced the chunk-insert `cur.executemany()` — which uses libpq **pipeline mode** the pooler rejects (`Pipeline [BAD]` / `EOF detected`) — with batched single multi-row `INSERT`s in `rag_utils.py`. Prepared statements and pipeline mode are *separate* psycopg3 features, hence the two-part fix.
- ✅ Duplicate-job race — `_resolve_snapshot` does a fresh DB read before enqueuing instead of trusting the 60s-stale cache.

---

## 🛠️ Sprint 1 (Days 1–5): fix-out — close the known bugs

**Scope note:** code-level fixes + **local** verification (no AWS yet). The big
correctness fixes are already in the working tree; Day 1 commits them as a
baseline, then this sprint closes the remaining robustness/quality gaps from the
backlog so Sprint 2 deploys a solid tree. ~6 hrs/day, **~30 hrs total**.

### Day 1 (2026-06-26) — Baseline + failure handling (#5) ✅
- [x] Baseline already committed (the earlier fixes landed in commits
      `e6019344`/`c799b508`/`c869b9f3`/`54386ae6`).
- [x] Bounded retry/backoff + timeouts on transient external calls — new
      `utils/retry.py` (`retry_call`/`with_retry`), applied to DB `connect()`
      (retry `OperationalError`), Polygon `fetch_price_rows_polygon` (retry
      429/5xx, then graceful `[]`), EDGAR `_get` (retry 429/5xx/conn/timeout, not
      permanent 4xx), and `LLM_MAX_RETRIES` on the OpenAI/Anthropic/Google LLMs.
- [x] Job-failure surfacing — `process_job` already writes `failed` + `error` to
      `jobs`; hardened the worker loop so a transient `claim_next_job()` error
      backs off instead of crashing the loop.
- **Exit:** a forced provider outage degrades gracefully; failed jobs report a reason. ✅

### Day 2 (2026-06-27) — Input validation & API robustness (#9 partial) ✅
- [x] `_clean_ticker()` — uppercases/strips + regex (`^[A-Z][A-Z0-9.\-]{0,9}$`,
      covers BRK.B/BRK-B); rejects garbage/injection/over-long with a clean 400
      before any DB/pipeline work. Applied to analyze, snapshot, history, price.
- [x] JSON on every error path — `HTTPException` handler (404/405/400→JSON, not
      Werkzeug HTML) + catch-all `Exception` handler (→ JSON 500, no stack leak);
      `get_json(silent=True)`; clamped `days` (1–365) in history.
- [x] NaN/Inf safe app-wide — `_SafeJSONProvider` runs `_sanitize_for_json` on
      every response, so no endpoint can emit invalid JSON regardless of the value.
- **Exit:** verified via test client — malformed input → clean 400/404 JSON;
      NaN/Inf → `null`. ✅

### Day 3 (2026-06-28) — RAG / ingestion correctness (#7 light)
- [ ] Stress the new multi-row chunk INSERT on a large 10-K (hundreds of chunks);
      confirm batch sizing + embedding-namespace tagging are correct and idempotent
      on re-run.
- [ ] Verify retrieval date-range filtering returns the right filings per horizon.
- [ ] Add a lightweight number-grounding check (the eval's hallucination signal) as
      a guard/log in the fundamental path — flag a quoted number not in any chunk.
- **Exit:** ingest is idempotent + correct under load; grounding mismatches are visible.

### Day 4 (2026-06-29) — Concurrency, DB hygiene & UX (#6, #8)
- [ ] Settle DB connection discipline (per-call vs a small pool); confirm 2 workers
      run safely side-by-side (`claim_next_job` `SKIP LOCKED`).
- [ ] Re-verify the dedup-race fix + cache correctness under concurrent polling.
- [ ] Frontend: a clear failed-job state (not an infinite spinner). Stretch:
      LLM-synthesize the debate section in the writer instead of the fixed template.
- **Exit:** concurrent workers + pollers behave; a failed job renders as failed in the UI.

### Day 5 (2026-06-30) — Tests + local full-run regression (#4)
- [ ] Unit/integration tests for every fix above + the earlier ones (`ensure_filings`
      no-op, prices persist+read, dedup enqueues once, price→CSV→non-null metrics,
      retry/fallback, input validation).
- [ ] Full **local** pipeline run for 2–3 tickers × 3 horizons; fix stragglers.
- [ ] Tidy the untracked spike test files.
- **Exit:** `pytest` green locally; the tree is correct & robust → ready for Sprint 2.

> Sprint 1 = correctness & robustness (local). Sprint 2 = ship it & verify in prod.

---

## 🚀 Sprint 2 (Days 6–10): deploy & bug-free sign-off

**Scope note:** "bug-free" here = **known-bug-free and verified working
end-to-end on the live AWS stack** — a deployment sign-off, not a mathematical
guarantee of zero bugs. With Sprint 1 done, the tree is already fixed; this sprint
is **deploy → verify → prod-only edge cases → sign off**, not new feature work.
Assumes ~full days (~6 hrs/day, **~30–35 hrs total**).

### Day 6 (2026-07-01) — Land & deploy
- [ ] Ensure all Sprint 1 fixes are merged to `main` → PR → final pre-deploy review.
- [ ] Push; confirm `build-and-push.yml` (backend + worker) and `deploy-frontend.yml`
      all go green.
- [ ] Set **`POLYGON_API_KEY` + `SEC_USER_AGENT`** on the worker ECS task def (both
      are now load-bearing). Redeploy the worker.
- [ ] Purge the duplicate AAPL snapshots (#11).
- **Exit:** backend `/`, worker, and frontend all live on the deployed stack; CI green.

### Day 7 (2026-07-02) — End-to-end verification on AWS (the core day)
- [ ] Run a real ticker through the deployed flow: enter AAPL → **exactly one** job
      per horizon (no dup) → worker pulls filings from EDGAR with **no SSL/pipeline
      errors** → snapshot has **non-null `cumulative_return` + `prices`** → chart
      renders → second `/api/price` call serves from snapshot (no live Polygon hit).
- [ ] Repeat across **all 3 horizons** and **2–3 tickers** (incl. one never seen).
- [ ] Fix whatever breaks. This is the "does it actually work in prod" gate.
- **Exit:** a clean full run for ≥3 tickers × 3 horizons, logs free of the known errors.

### Day 8 (2026-07-03) — Prod-only edge cases
- [ ] Re-test the Sprint 1 resilience fixes **on AWS**: invalid ticker, EDGAR
      unreachable, missing `POLYGON_API_KEY`, concurrent same-ticker requests, a
      forced worker-task restart mid-job.
- [ ] Confirm every failure **degrades, never crashes** the worker loop or backend
      in the real environment (managed Postgres, ALB, Fargate timeouts).
- **Exit:** no crash path in prod; failed jobs show a clear status, not a hang.

### Day 9 (2026-07-04) — Deployed-stack regression
- [ ] Re-run the Sprint 1 test suite against the deployed config (managed Postgres
      via the pooler, real Polygon/EDGAR) — catch anything that only breaks in prod.
- [ ] Soak test: leave the worker running, confirm the weekly filing sweep + stale
      snapshot refresh fire without errors.
- **Exit:** tests green against prod config; the worker survives an idle soak.

### Day 10 (2026-07-05) — Final deploy, log check, sign-off (+ buffer)
- [ ] Clean redeploy from `main`; full user walkthrough on the live URL.
- [ ] Scan worker + backend logs end-to-end: **zero** SSL/pipeline errors, zero
      null-return snapshots, no duplicate jobs.
- [ ] Sanity-check health endpoints; write a 1-page runbook (URLs, env vars, "how to
      redeploy"). Tag a release.
- [ ] Reserve the back half of the day as **buffer** for straggler bugs.
- **Exit:** signed-off deployment — known-bug-free, verified, documented.

> Sprints 1–2 together (10 full days) ≈ Phase 0 + the critical slice of Phase 1,
> front-loaded: **fix locally, then deploy & verify in prod.** The remaining
> Phase 1–5 items below are the longer hardening roadmap beyond bug-free deploy.

---

## Timeline to v1.0

Total hands-on effort: **~110–165 hrs**. The calendar depends entirely on daily
cadence:

| Cadence | Calendar to v1.0 |
|---------|------------------|
| ~2–3 hrs/day (part-time, ~15 hrs/wk) | **6–7 weeks** ← default below |
| ~5 hrs/day | ~4–5 weeks |
| Full-time (~7 hrs/day) | ~3–3.5 weeks |

Per-phase effort (hrs) is on each heading. Two caveats so the hours aren't
misread: (a) much of Phase 2/3 is *wall-clock waiting* on eval runs / deploys,
not hands-on time — parallelize other work then; (b) most implementation can be
done in-session, so the real bottleneck is often *your review/decision* time,
not raw coding hours.

> Phase dates below assume the **~2–3 hrs/day** cadence. Rescale if yours differs.

### Phase 0 — Stabilize current main (this week) · ~10–15 hrs
Deploy today's work, verify end-to-end on AWS, fix yfinance-in-valuation (#1),
confirm worker env (#3), purge duplicate snapshots (#11).
**Exit:** a real AAPL run produces a complete snapshot with non-null returns + a
working chart, no SSL errors.

### Phase 1 — Correctness + tests (wk 1–2) · ~20–30 hrs
Kill yfinance everywhere, add automated tests for the new paths (#4), add
retry/backoff + job-failure visibility (#5).
**Exit:** `pytest` green; a forced provider outage degrades gracefully instead
of failing a job.

### Phase 2 — Evaluate & decide debate (wk 2–4) · ~25–40 hrs (eval runs are slow/iterative)
Run `eval_suite.py` **chain-only vs chain+debate** head-to-head on recommendation
accuracy + stability. Let data decide debate's fate, then improve the winner.
**Exit:** a documented decision backed by eval numbers.

### Phase 3 — Scale + ops (wk 4–5) · ~20–30 hrs
Real connection pooling, multi-worker test, EDGAR/Polygon/LLM rate-limit
handling, basic metrics + alerting (#6, #12), a small load test against the
"thousands of users" target.

### Phase 4 — Quality + UX polish (wk 5–6) · ~20–30 hrs
Hallucination guardrail (#7), LLM-synthesized writer + failed-job UX (#8), light
auth/rate-limit if public (#9).

### Phase 5 — Release candidate (wk 6–7) · ~15–20 hrs
Freeze, full regression, security pass, docs. Tag **v1.0**.

> Phases 0–2 are the irreducible core; 3–5 are hardening that can be traded off
> against a hard deadline.

---

## Debate graph — verdict: keep it, but prove it, and restructure it

**Honest assessment of the current implementation:** academically valuable but,
as built, probably doesn't improve the *answer* much — and may hurt.

**Why skeptical it helps accuracy:**
- **Adds no new evidence** — specialists re-argue over the *same* retrieved
  state. Without fresh tool calls between rounds it's rhetoric, not research.
- **Convergence ≠ truth** — in equity calls there's no ground truth to converge
  toward; LLMs converge *socially* (anchoring/sycophancy), so "consensus"
  manufactures confidence rather than correctness.
- **Cost/latency multiplier** on a pipeline that's already minutes long.
- 5 fixtures is far too small to claim debate beats chain-only.

**Why keep it anyway:**
- For a research project it's the **novel contribution** and the best narrative.
- Structured *disagreement* (bull + bear) improves the **report's usefulness**
  even if it doesn't move accuracy.
- Already toggleable via `ANALYSIS_MODE` — costs nothing to keep as an option.

**Improvement plan (do in Phase 2):**
1. **Measure first** — chain-only vs chain+debate via `eval_suite.py`; decide
   with data.
2. **Restructure round-robin → adversarial** bull vs bear + a judge.
3. **Give debaters tools** — let them re-query filings/prices when challenged so
   each round adds *evidence* (biggest single fix).
4. **Drop forced consensus** — output a recommendation with a **confidence score
   and recorded dissent** instead of fake agreement.
5. **Let the writer synthesize the debate** (LLM) instead of flattening it into
   the fixed template.

---

## Human-in-the-loop (LangGraph `interrupt()` + checkpointer) — verdict: defer

It's a **LangGraph** feature (not LangChain); you'd use the Postgres checkpointer.

**Don't add it now — it fights the current architecture:**
- The pipeline is **fire-and-forget**: a background worker grinds for minutes
  with no human watching. HITL pauses a graph to wait for human input — the two
  models are fundamentally at odds.
- To use it meaningfully you'd need a **synchronous "interactive analysis" mode**
  (separate UX from the cached-snapshot model) where an analyst approves/overrides
  the recommendation or corrects a hallucinated number. That's a genuine v2
  feature, not a bolt-on.

**If pursued (v2 / Phase 4+):** scope it narrowly as an optional **analyst review
gate** before a recommendation is published, in an interactive mode — not in the
background snapshot path.
