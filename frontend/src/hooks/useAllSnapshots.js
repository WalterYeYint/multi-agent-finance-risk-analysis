import { useEffect, useState, useRef, useCallback } from 'react';
import axios from 'axios';

const HORIZONS = ['SHORT', 'MID', 'LONG'];

/**
 * Fan-out version of useSnapshot. Starts three independent poll loops, one per
 * horizon, so visiting a TickerView eagerly warms all three caches in parallel
 * instead of lazily-on-tab-click. Returns:
 *
 *   {
 *     byHorizon: {
 *       SHORT: { snapshot, pending, error, job, status },
 *       MID:   { ... },
 *       LONG:  { ... },
 *     },
 *     allReady: boolean,
 *     retry: (horizon) => void,   // re-run a failed horizon
 *   }
 *
 * `status` is one of: 'loading' | 'ready' | 'pending' | 'failed' | 'error'.
 *   - 'failed' = the worker ran the job and it failed (job.error explains why);
 *     polling stops so the user sees the failure instead of a perpetual spinner,
 *     and `retry(horizon)` re-enqueues a fresh job.
 *   - 'error'  = the request itself failed (network / unexpected backend error)
 *     and kept failing past the transient-retry budget — see below.
 *
 * Transient gateway blips (502/503/504, request timeout, or a dropped
 * connection with no response) are EXPECTED while a job runs: the worker is
 * busy and the load balancer (the ECS Express Mode ALB) can briefly fail to
 * reach the backend. These are NOT surfaced as 'error' — the hook keeps the
 * current "Running" view and keeps polling, only giving up after MAX_TRANSIENT
 * consecutive blips so a genuinely-down backend still surfaces eventually.
 */
const MAX_TRANSIENT = 12; // consecutive gateway blips tolerated before 'error'

// A request error we should retry through rather than surface. No `response`
// at all = network drop / connection reset (container restart, cold path);
// 5xx / 408 / 429 = the gateway or backend was momentarily unavailable.
function isTransient(e) {
  const s = e?.response?.status;
  if (s == null) return true;
  return s === 408 || s === 429 || (s >= 500 && s <= 599);
}

export function useAllSnapshots(ticker, { intervalMs = 3000 } = {}) {
  const [byHorizon, setByHorizon] = useState(() => initialState());
  // pollOne lives inside the effect (closes over the live cancelled flag +
  // timers); expose the current one via a ref so retry() can re-trigger a
  // single horizon without resetting the other two.
  const pollRef = useRef(null);
  // Mirror of the latest state for non-render reads (refresh() needs the
  // current snapshot's generated_at without adding a state dependency).
  const stateRef = useRef(byHorizon);
  stateRef.current = byHorizon;

  useEffect(() => {
    if (!ticker) return undefined;
    let cancelled = false;
    const timers = {};
    const transientFails = {}; // per-horizon consecutive gateway-blip counter

    setByHorizon(initialState());

    // refreshCtx (set by refresh()) = { after, jobId }: a user-forced refresh
    // is in flight. The old snapshot may still be within freshness_hours, so
    // the GET below keeps returning it with 200 — a plain poll would snap back
    // to 'ready' showing STALE data and stop. While the ctx is set, a 200 whose
    // generated_at isn't newer than `after` counts as "still waiting": we poll
    // the job row instead (for progress + failure) and keep looping until the
    // GET finally serves a newer snapshot.
    const pollOne = async (horizon, force = false, refreshCtx = null) => {
      try {
        const url =
          `/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}` +
          (force ? '?retry=1' : '');
        const res = await axios.get(url, { validateStatus: (s) => s === 200 || s === 202 });
        if (cancelled) return;
        transientFails[horizon] = 0; // a real response → reset the blip budget
        if (res.status === 200) {
          const notNewer = refreshCtx?.after && res.data?.generated_at &&
            new Date(res.data.generated_at).getTime() <= new Date(refreshCtx.after).getTime();
          if (notNewer) {
            if (refreshCtx.jobId != null) {
              try {
                const jr = await axios.get(`/api/jobs/${refreshCtx.jobId}`);
                if (cancelled) return;
                if (jr.data?.status === 'failed') {
                  setByHorizon((prev) => ({
                    ...prev,
                    [horizon]: {
                      snapshot: null, pending: null, job: jr.data,
                      error: jr.data?.error || 'Analysis failed.', status: 'failed',
                    },
                  }));
                  return;
                }
                setByHorizon((prev) => ({
                  ...prev,
                  [horizon]: { snapshot: null, pending: jr.data, error: null, job: null, status: 'pending' },
                }));
              } catch { /* transient job-poll blip — just keep looping */ }
            }
            timers[horizon] = setTimeout(() => pollOne(horizon, false, refreshCtx), intervalMs);
            return;
          }
          setByHorizon((prev) => ({
            ...prev,
            [horizon]: { snapshot: res.data, pending: null, error: null, job: null, status: 'ready' },
          }));
          return;
        }
        // 202: a job payload. A terminal 'failed' job stops the loop.
        if (res.data?.status === 'failed') {
          setByHorizon((prev) => ({
            ...prev,
            [horizon]: {
              snapshot: null, pending: null, job: res.data,
              error: res.data?.error || 'Analysis failed.', status: 'failed',
            },
          }));
          return;
        }
        setByHorizon((prev) => ({
          ...prev,
          [horizon]: { snapshot: null, pending: res.data, error: null, job: null, status: 'pending' },
        }));
        timers[horizon] = setTimeout(() => pollOne(horizon, false, refreshCtx), intervalMs);
      } catch (e) {
        if (cancelled) return;
        // Definitive enqueue rejections from I1 (auth / rate limit) carry an
        // explicit code. Surface them right away instead of treating the 401/429
        // as a transient gateway blip to retry through.
        const code = e?.response?.data?.code;
        if (code === 'unauthorized' || code === 'rate_limited') {
          setByHorizon((prev) => ({
            ...prev,
            [horizon]: {
              snapshot: null, pending: null, job: null,
              error: e?.response?.data?.error || 'Request rejected.', status: 'error',
            },
          }));
          return;
        }
        // A transient gateway blip while a job is in flight: keep the current
        // view (loading/pending — no setState) and keep polling, backing off a
        // little to let the gateway recover. Only after MAX_TRANSIENT in a row
        // do we conclude the backend is actually down and surface the error.
        if (isTransient(e)) {
          transientFails[horizon] = (transientFails[horizon] || 0) + 1;
          if (transientFails[horizon] <= MAX_TRANSIENT) {
            const backoff = Math.min(intervalMs * 2, 8000);
            timers[horizon] = setTimeout(() => pollOne(horizon, force, refreshCtx), backoff);
            return;
          }
        }
        const msg = e?.response?.data?.error || e.message || 'Snapshot fetch failed';
        setByHorizon((prev) => ({
          ...prev,
          [horizon]: { snapshot: null, pending: null, error: msg, job: null, status: 'error' },
        }));
      }
    };

    pollRef.current = pollOne;
    HORIZONS.forEach((h) => pollOne(h));

    return () => {
      cancelled = true;
      Object.values(timers).forEach((t) => t && clearTimeout(t));
    };
  }, [ticker, intervalMs]);

  const retry = useCallback((horizon) => {
    if (!pollRef.current) return;
    setByHorizon((prev) => ({ ...prev, [horizon]: emptyState() }));
    pollRef.current(horizon, true); // force=true → backend enqueues past the failure
  }, []);

  // User-requested refresh of a READY horizon (the button). POSTs the force-
  // refresh endpoint; on 202 flips the horizon into 'pending' and hands off to
  // the normal poll loop. On rejection (too_fresh / rate_limited / unauthorized)
  // the ready snapshot is left untouched and the error message is RETURNED for
  // inline display — a refused refresh must not blank a perfectly good page.
  const refresh = useCallback(async (horizon) => {
    try {
      // Watermark: the poll loop only accepts a snapshot NEWER than this —
      // the backend keeps serving the current one (still within freshness)
      // with 200 while the refresh job runs.
      const after = stateRef.current?.[horizon]?.snapshot?.generated_at || null;
      const url =
        `/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}/refresh`;
      const res = await axios.post(url, null, { validateStatus: (s) => s === 202 });
      setByHorizon((prev) => ({
        ...prev,
        [horizon]: { snapshot: null, pending: res.data, error: null, job: null, status: 'pending' },
      }));
      if (pollRef.current) {
        pollRef.current(horizon, false, { after, jobId: res.data?.job_id ?? null });
      }
      return null;
    } catch (e) {
      return e?.response?.data?.error || e.message || 'Refresh failed.';
    }
  }, [ticker]);

  const allReady = HORIZONS.every((h) => byHorizon[h].status === 'ready');
  return { byHorizon, allReady, retry, refresh };
}

function initialState() {
  return { SHORT: emptyState(), MID: emptyState(), LONG: emptyState() };
}

function emptyState() {
  return { snapshot: null, pending: null, error: null, job: null, status: 'loading' };
}
