import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { useOverview } from '../hooks/useOverview';
import OverviewTable from '../components/OverviewTable';

function Landing() {
  const { tickers, loading, error } = useOverview();
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const onSubmit = (e) => {
    e.preventDefault();
    const t = query.trim().toUpperCase();
    if (!t) return;
    navigate(`/t/${encodeURIComponent(t)}?horizon=SHORT`);
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
          onChange={(e) => setQuery(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <button type="submit" className="primary-btn">Go</button>
      </form>

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
