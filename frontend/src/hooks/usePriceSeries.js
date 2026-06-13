import { useEffect, useState } from 'react';
import axios from 'axios';

/**
 * Fetch the daily-close price series for the chart. `period` is one of the
 * three horizon lookback windows: '1mo' | '6mo' | '2y'. Returns
 *   { series, loading, error }
 * where `series` is an array of { date, close }. Empty array means the backend
 * couldn't fetch data (e.g. yfinance hiccup); the UI should render an empty
 * chart state, not an error toast.
 */
export function usePriceSeries(ticker, period) {
  const [series, setSeries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ticker || !period) return undefined;
    let cancelled = false;
    setLoading(true);
    setError(null);

    axios
      .get(`/api/price/${encodeURIComponent(ticker)}`, { params: { period } })
      .then((res) => {
        if (cancelled) return;
        setSeries(res.data?.series || []);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.response?.data?.error || e.message || 'Price fetch failed');
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, [ticker, period]);

  return { series, loading, error };
}
