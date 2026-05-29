import { useEffect, useState } from 'react';
import axios from 'axios';

/** Fetch the `/history` timeseries for one (ticker, horizon). */
export function useHistory(ticker, horizon, { days = 90 } = {}) {
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ticker || !horizon) return undefined;
    let cancelled = false;
    setLoading(true);
    setError(null);

    axios
      .get(`/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}/history`, {
        params: { days },
      })
      .then((res) => {
        if (cancelled) return;
        setHistory(res.data?.history ?? []);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.response?.data?.error || e.message || 'History fetch failed');
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [ticker, horizon, days]);

  return { history, error, loading };
}
