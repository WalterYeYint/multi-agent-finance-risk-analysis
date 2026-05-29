import React from 'react';
import Sparkline from './Sparkline';
import { useHistory } from '../hooks/useHistory';

const SENTIMENT_SCORE = { bullish: 1, neutral: 0, bearish: -1 };

function HistoryStrip({ ticker, horizon, days = 90 }) {
  const { history, error, loading } = useHistory(ticker, horizon, { days });

  if (loading) {
    return <div className="history-strip history-strip--muted">Loading history…</div>;
  }
  if (error) {
    return <div className="history-strip history-strip--muted">History unavailable: {error}</div>;
  }
  if (!history.length) {
    return (
      <div className="history-strip history-strip--muted">
        No history yet — the first snapshot you see is also the only one.
      </div>
    );
  }

  const sentimentSeries = history.map((p) => SENTIMENT_SCORE[p.overall_sentiment] ?? null);
  const volSeries = history.map((p) => (typeof p.annual_vol === 'number' ? p.annual_vol : null));
  const ddSeries = history.map((p) => (typeof p.max_drawdown === 'number' ? p.max_drawdown : null));
  const latest = history[history.length - 1];

  return (
    <div className="history-strip">
      <div className="history-strip__item">
        <span className="history-strip__label">Sentiment</span>
        <Sparkline values={sentimentSeries} />
        <span className="history-strip__value">{latest.overall_sentiment ?? '—'}</span>
      </div>
      <div className="history-strip__item">
        <span className="history-strip__label">Annual vol</span>
        <Sparkline values={volSeries} stroke="#d97706" fill="rgba(217, 119, 6, 0.12)" />
        <span className="history-strip__value">
          {typeof latest.annual_vol === 'number' ? `${(latest.annual_vol * 100).toFixed(1)}%` : '—'}
        </span>
      </div>
      <div className="history-strip__item">
        <span className="history-strip__label">Max drawdown</span>
        <Sparkline values={ddSeries} stroke="#dc2626" fill="rgba(220, 38, 38, 0.12)" />
        <span className="history-strip__value">
          {typeof latest.max_drawdown === 'number' ? `${(latest.max_drawdown * 100).toFixed(1)}%` : '—'}
        </span>
      </div>
      <div className="history-strip__meta">{history.length} snapshot{history.length === 1 ? '' : 's'} in the last {days} days</div>
    </div>
  );
}

export default HistoryStrip;
