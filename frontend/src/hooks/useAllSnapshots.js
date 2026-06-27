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
 *   - 'error'  = the request itself failed (network / unexpected backend error).
 */
export function useAllSnapshots(ticker, { intervalMs = 3000 } = {}) {
  const [byHorizon, setByHorizon] = useState(() => initialState());
  // pollOne lives inside the effect (closes over the live cancelled flag +
  // timers); expose the current one via a ref so retry() can re-trigger a
  // single horizon without resetting the other two.
  const pollRef = useRef(null);

  useEffect(() => {
    if (!ticker) return undefined;
    let cancelled = false;
    const timers = {};

    setByHorizon(initialState());

    const pollOne = async (horizon, force = false) => {
      try {
        const url =
          `/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}` +
          (force ? '?retry=1' : '');
        const res = await axios.get(url, { validateStatus: (s) => s === 200 || s === 202 });
        if (cancelled) return;
        if (res.status === 200) {
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
        timers[horizon] = setTimeout(() => pollOne(horizon), intervalMs);
      } catch (e) {
        if (cancelled) return;
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

  const allReady = HORIZONS.every((h) => byHorizon[h].status === 'ready');
  return { byHorizon, allReady, retry };
}

function initialState() {
  return { SHORT: emptyState(), MID: emptyState(), LONG: emptyState() };
}

function emptyState() {
  return { snapshot: null, pending: null, error: null, job: null, status: 'loading' };
}
