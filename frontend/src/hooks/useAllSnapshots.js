import { useEffect, useState } from 'react';
import axios from 'axios';

const HORIZONS = ['SHORT', 'MID', 'LONG'];

/**
 * Fan-out version of useSnapshot. Starts three independent poll loops, one per
 * horizon, so visiting a TickerView eagerly warms all three caches in parallel
 * instead of lazily-on-tab-click. Returns:
 *
 *   {
 *     byHorizon: {
 *       SHORT: { snapshot, pending, error, status },
 *       MID:   { snapshot, pending, error, status },
 *       LONG:  { snapshot, pending, error, status },
 *     },
 *     allReady: boolean,
 *   }
 *
 * `status` is one of: 'loading' | 'ready' | 'pending' | 'error'.
 * Each horizon resolves independently; the UI can render a ready horizon
 * immediately even while the other two are still being computed.
 */
export function useAllSnapshots(ticker, { intervalMs = 3000 } = {}) {
  const [byHorizon, setByHorizon] = useState(() => initialState());

  useEffect(() => {
    if (!ticker) return undefined;
    let cancelled = false;
    const timers = {};

    setByHorizon(initialState());

    const pollOne = async (horizon) => {
      try {
        const res = await axios.get(
          `/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}`,
          { validateStatus: (s) => s === 200 || s === 202 },
        );
        if (cancelled) return;
        if (res.status === 200) {
          setByHorizon((prev) => ({
            ...prev,
            [horizon]: { snapshot: res.data, pending: null, error: null, status: 'ready' },
          }));
        } else {
          setByHorizon((prev) => ({
            ...prev,
            [horizon]: { snapshot: null, pending: res.data, error: null, status: 'pending' },
          }));
          timers[horizon] = setTimeout(() => pollOne(horizon), intervalMs);
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e?.response?.data?.error || e.message || 'Snapshot fetch failed';
        setByHorizon((prev) => ({
          ...prev,
          [horizon]: { snapshot: null, pending: null, error: msg, status: 'error' },
        }));
      }
    };

    HORIZONS.forEach(pollOne);

    return () => {
      cancelled = true;
      Object.values(timers).forEach((t) => t && clearTimeout(t));
    };
  }, [ticker, intervalMs]);

  const allReady = HORIZONS.every((h) => byHorizon[h].status === 'ready');
  return { byHorizon, allReady };
}

function initialState() {
  return {
    SHORT: emptyState(),
    MID: emptyState(),
    LONG: emptyState(),
  };
}

function emptyState() {
  return { snapshot: null, pending: null, error: null, status: 'loading' };
}
