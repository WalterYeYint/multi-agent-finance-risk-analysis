import React, { useMemo } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ChevronLeft, AlertTriangle } from 'lucide-react';
import HorizonSummaryStrip from '../components/HorizonSummaryStrip';
import PriceChart from '../components/PriceChart';
import HistoryStrip from '../components/HistoryStrip';
import SnapshotView from '../components/SnapshotView';
import PendingView from '../components/PendingView';
import { useAllSnapshots } from '../hooks/useAllSnapshots';

const HORIZONS = ['SHORT', 'MID', 'LONG'];

function TickerView() {
  const { ticker } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const horizon = useMemo(() => {
    const raw = (searchParams.get('horizon') || 'SHORT').toUpperCase();
    return HORIZONS.includes(raw) ? raw : 'SHORT';
  }, [searchParams]);

  const upperTicker = (ticker || '').toUpperCase();
  // Eager: fan out to all three horizons at once. The user sees a ready horizon
  // immediately even while the other two are still being computed.
  const { byHorizon, retry } = useAllSnapshots(upperTicker);
  const active = byHorizon[horizon];

  const setHorizon = (h) => {
    const next = new URLSearchParams(searchParams);
    next.set('horizon', h);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="ticker-view">
      <div className="ticker-view__head">
        <Link to="/" className="ticker-view__back">
          <ChevronLeft size={16} /> All tickers
        </Link>
        <h1 className="ticker-view__symbol">{upperTicker}</h1>
      </div>

      <HorizonSummaryStrip
        byHorizon={byHorizon}
        active={horizon}
        onSelect={setHorizon}
      />

      <PriceChart ticker={upperTicker} horizon={horizon} />

      <HistoryStrip ticker={upperTicker} horizon={horizon} days={90} />

      <div className="ticker-view__body">
        {active.status === 'error' && (
          <div className="snapshot__error">
            <AlertTriangle size={18} className="snapshot__error-icon" />
            <div className="snapshot__error-body">
              <div className="snapshot__error-title">Couldn’t load the {horizon} snapshot</div>
              <div className="snapshot__error-msg">{active.error}</div>
            </div>
            <button type="button" className="snapshot__retry" onClick={() => retry(horizon)}>
              Try again
            </button>
          </div>
        )}
        {active.status === 'loading' && (
          <div className="snapshot__skeleton">
            <div className="snapshot__skeleton-bar" style={{ width: '40%' }} />
            <div className="snapshot__skeleton-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="snapshot__skeleton-cell" />
              ))}
            </div>
            <div className="landing__muted">Loading {horizon} snapshot…</div>
          </div>
        )}
        {active.status === 'failed' && (
          <div className="snapshot__failed">
            <div className="snapshot__failed-title">Analysis failed for {horizon}</div>
            <div className="snapshot__failed-msg">{active.error}</div>
            <button type="button" className="snapshot__retry" onClick={() => retry(horizon)}>
              Retry
            </button>
          </div>
        )}
        {active.status === 'pending' && <PendingView pending={active.pending} />}
        {active.status === 'ready' && <SnapshotView snapshot={active.snapshot} />}
      </div>
    </div>
  );
}

export default TickerView;
