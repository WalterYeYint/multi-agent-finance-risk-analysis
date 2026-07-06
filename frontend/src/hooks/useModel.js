import { useEffect, useState } from 'react';
import axios from 'axios';

/** Fetch `/api/model` → `{ provider, model, label }` for the header model badge. */
export function useModel() {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    axios
      .get('/api/model')
      .then((res) => {
        if (cancelled) return;
        setModel(res.data ?? null);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e?.message || 'Failed to load model');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { model, error };
}
