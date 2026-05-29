import { useEffect, useState } from 'react';
import axios from 'axios';

/** Fetch `/api/tickers` → `{ tickers: [...], horizons: [...] }`. */
export function useTickers() {
  const [tickers, setTickers] = useState([]);
  const [horizons, setHorizons] = useState(['SHORT', 'MID', 'LONG']);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    axios
      .get('/api/tickers')
      .then((res) => {
        if (cancelled) return;
        setTickers(res.data?.tickers ?? []);
        if (Array.isArray(res.data?.horizons) && res.data.horizons.length) {
          setHorizons(res.data.horizons);
        }
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.response?.data?.error || e.message || 'Failed to load tickers');
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { tickers, horizons, error, loading };
}
