import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import axios from 'axios';
import { useOverview } from '../hooks/useOverview';
import { useActiveJobs } from '../hooks/useActiveJobs';
import OverviewTable from '../components/OverviewTable';

const HORIZON_LABEL = { SHORT: 'Short', MID: 'Mid', LONG: 'Long' };

function ActiveJobsStrip({ active }) {
  if (!active.length) return null;
  return (
    <div className="landing__active">
      <span className="landing__active-spinner" aria-hidden="true" />
      <span className="landing__active-label">Analyzing</span>
      <div className="landing__active-list">
        {active.map((j) => (
          <span key={`${j.ticker}-${j.horizon}`} className="landing__active-badge">
            {j.ticker}
            <span className="landing__active-badge-h">{HORIZON_LABEL[j.horizon] || j.horizon}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Landing() {
  const { tickers, loading, error } = useOverview();
  const { active } = useActiveJobs();
  const [query, setQuery] = useState('');
  const [checking, setChecking] = useState(false);
  const [lookupError, setLookupError] = useState('');
  const navigate = useNavigate();

  const go = (t) => navigate(`/t/${encodeURIComponent(t)}?horizon=SHORT`);

  const onSubmit = async (e) => {
    e.preventDefault();
    const t = query.trim().toUpperCase();
    if (!t) return;
    setLookupError('');
    setChecking(true);
    try {
      // Reject typos / non-existent tickers here so we never navigate to a
      // doomed page or enqueue a doomed job.
      await axios.get(`/api/ticker/${encodeURIComponent(t)}`);
      go(t);
    } catch (err) {
      const status = err?.response?.status;
      const code = err?.response?.data?.code;
      // Only block on a genuine rejection from THIS endpoint: an unknown-ticker
      // 404 (carries code:'unknown_ticker') or a format 400. Anything else — a
      // missing/stale endpoint (route-not-found 404), network drop, 5xx — must
      // NOT masquerade as "invalid ticker"; fail open and let the ticker page +
      // enqueue guard take over.
      if (status === 400 || (status === 404 && code === 'unknown_ticker')) {
        setLookupError(err?.response?.data?.error || `"${t}" is not a recognized ticker.`);
      } else {
        go(t);
      }
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="landing">
      <div className="landing__intro">
        <h1>Multi-agent finance research</h1>
        <p>
          Browse pre-computed snapshots across three horizons (Short / Mid / Long). Any ticker
          you look up is generated on demand and cached for next time.
        </p>
      </div>

      <form className="landing__search" onSubmit={onSubmit}>
        <Search size={16} />
        <input
          type="text"
          placeholder="Look up any ticker (e.g. NVDA)"
          value={query}
          onChange={(e) => { setQuery(e.target.value); if (lookupError) setLookupError(''); }}
          autoComplete="off"
          spellCheck={false}
        />
        <button type="submit" className="primary-btn" disabled={checking}>
          {checking ? 'Checking…' : 'Go'}
        </button>
      </form>

      {lookupError && <div className="landing__error">{lookupError}</div>}

      <ActiveJobsStrip active={active} />

      {loading && <div className="landing__muted">Loading tickers…</div>}
      {error && <div className="landing__error">Couldn't load tickers: {error}</div>}
      {!loading && !error && tickers.length === 0 && (
        <div className="landing__muted">
          No snapshots yet — look up a ticker above to generate the first one.
        </div>
      )}

      {tickers.length > 0 && <OverviewTable tickers={tickers} />}
    </div>
  );
}

export default Landing;
