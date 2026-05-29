import { useEffect, useState } from 'react';
import axios from 'axios';

/**
 * Fetch a snapshot for (ticker, horizon). On a cache miss the backend returns
 * 202 + a job id; this hook polls the same URL every `intervalMs` until the
 * status flips to 200. Returns `{ snapshot, pending, error, loading }` —
 * exactly one of `snapshot` and `pending` is set at a time.
 */
export function useSnapshot(ticker, horizon, { intervalMs = 3000 } = {}) {
  const [snapshot, setSnapshot] = useState(null);
  const [pending, setPending] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ticker || !horizon) return undefined;
    let cancelled = false;
    let timer = null;

    setSnapshot(null);
    setPending(null);
    setError(null);
    setLoading(true);

    const poll = async () => {
      try {
        const res = await axios.get(
          `/api/snapshot/${encodeURIComponent(ticker)}/${encodeURIComponent(horizon)}`,
          { validateStatus: (s) => s === 200 || s === 202 },
        );
        if (cancelled) return;
        if (res.status === 200) {
          setSnapshot(res.data);
          setPending(null);
          setLoading(false);
        } else {
          setPending(res.data);
          setSnapshot(null);
          setLoading(false);
          timer = setTimeout(poll, intervalMs);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e?.response?.data?.error || e.message || 'Snapshot fetch failed');
        setLoading(false);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [ticker, horizon, intervalMs]);

  return { snapshot, pending, error, loading };
}
