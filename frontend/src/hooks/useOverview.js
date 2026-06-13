import { useEffect, useState } from 'react';
import axios from 'axios';

/**
 * Fetch `/api/overview` once (no polling) → `{ tickers, loading, error }`.
 * `tickers` is `[{ ticker, horizons: { SHORT|MID|LONG: { recommendation,
 * cumulative_return, generated_at } } }]`. The endpoint is read-only on the
 * backend — it never enqueues pipeline jobs.
 */
export function useOverview() {
  const [tickers, setTickers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    axios
      .get('/api/overview')
      .then((res) => {
        if (cancelled) return;
        setTickers(res.data?.tickers ?? []);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.response?.data?.error || e.message || 'Failed to load overview');
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { tickers, loading, error };
}
