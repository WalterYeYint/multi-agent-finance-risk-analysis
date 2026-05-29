import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, Clock } from 'lucide-react';

function formatWhen(iso) {
  if (!iso) return '—';
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return iso;
  const diffMs = Date.now() - t.getTime();
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function TickerCard({ ticker }) {
  return (
    <Link to={`/t/${ticker.ticker}?horizon=SHORT`} className="ticker-card">
      <div className="ticker-card__head">
        <span className="ticker-card__sym">{ticker.ticker}</span>
        <ArrowUpRight size={16} className="ticker-card__arrow" />
      </div>
      <div className="ticker-card__meta">
        <Clock size={12} />
        <span>updated {formatWhen(ticker.last_updated)}</span>
      </div>
      <div className="ticker-card__count">
        {ticker.snapshot_count} snapshot{ticker.snapshot_count === 1 ? '' : 's'}
      </div>
    </Link>
  );
}

export default TickerCard;
