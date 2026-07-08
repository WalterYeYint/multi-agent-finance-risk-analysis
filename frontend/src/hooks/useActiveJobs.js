import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

/**
 * Poll `/api/overview` for the list of in-flight analysis jobs so the landing
 * page can show what's currently being computed. Returns
 *   { active }
 * where `active` is `[{ ticker, horizon, status }]` (status: 'queued' |
 * 'running'), empty when nothing is running.
 *
 * The `active` field is served fresh (uncached) by the backend, so a short poll
 * interval keeps the indicator responsive as jobs start/finish. Failures are
 * swallowed — a missing indicator is never worth surfacing an error for.
 */
export function useActiveJobs({ intervalMs = 5000 } = {}) {
  const [active, setActive] = useState([]);
  const timerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await axios.get('/api/overview');
        if (!cancelled) setActive(res.data?.active ?? []);
      } catch {
        if (!cancelled) setActive([]);
      } finally {
        if (!cancelled) timerRef.current = setTimeout(poll, intervalMs);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [intervalMs]);

  return { active };
}
