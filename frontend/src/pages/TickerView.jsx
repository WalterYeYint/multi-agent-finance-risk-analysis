import React, { useMemo } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';
import HorizonTabs from '../components/HorizonTabs';
import HistoryStrip from '../components/HistoryStrip';
import SnapshotView from '../components/SnapshotView';
import PendingView from '../components/PendingView';
import { useSnapshot } from '../hooks/useSnapshot';

const HORIZONS = ['SHORT', 'MID', 'LONG'];

function TickerView() {
  const { ticker } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const horizon = useMemo(() => {
    const raw = (searchParams.get('horizon') || 'SHORT').toUpperCase();
    return HORIZONS.includes(raw) ? raw : 'SHORT';
  }, [searchParams]);

  const upperTicker = (ticker || '').toUpperCase();
  const { snapshot, pending, error, loading } = useSnapshot(upperTicker, horizon);

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

      <HorizonTabs horizons={HORIZONS} active={horizon} onChange={setHorizon} />

      <HistoryStrip ticker={upperTicker} horizon={horizon} days={90} />

      <div className="ticker-view__body">
        {error && <div className="snapshot__error">Could not load snapshot: {error}</div>}
        {!error && loading && !snapshot && !pending && <div className="landing__muted">Loading…</div>}
        {!error && pending && <PendingView pending={pending} />}
        {!error && snapshot && <SnapshotView snapshot={snapshot} />}
      </div>
    </div>
  );
}

export default TickerView;
