import React from 'react';
import { CheckCircle2, Loader2, AlertCircle, Clock } from 'lucide-react';

const HORIZONS = ['SHORT', 'MID', 'LONG'];
const PERIOD_LABEL = { SHORT: '1mo', MID: '6mo', LONG: '2y' };

function pickRecommendation(text) {
  if (!text || typeof text !== 'string') return null;
  // The debate's consensus_summary embeds the BUY/HOLD/SELL token. We scan in
  // priority order — explicit "SELL" wins over "BUY" if both appear, because
  // recommendations are usually negated ("avoid a BUY") more than promoted.
  const upper = text.toUpperCase();
  if (/\bSELL\b/.test(upper)) return 'SELL';
  if (/\bBUY\b/.test(upper)) return 'BUY';
  if (/\bHOLD\b/.test(upper)) return 'HOLD';
  return null;
}

function tagClass(rec) {
  if (rec === 'BUY') return 'tag tag--positive';
  if (rec === 'SELL') return 'tag tag--negative';
  if (rec === 'HOLD') return 'tag tag--neutral';
  return 'tag';
}

function pct(v) {
  return typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—';
}

function StatusBadge({ status }) {
  if (status === 'ready') {
    return (
      <span className="hsum__badge hsum__badge--ok">
        <CheckCircle2 size={12} /> Ready
      </span>
    );
  }
  if (status === 'pending') {
    return (
      <span className="hsum__badge hsum__badge--running">
        <Loader2 size={12} className="hsum__spin" /> Running
      </span>
    );
  }
  if (status === 'failed') {
    return (
      <span className="hsum__badge hsum__badge--err">
        <AlertCircle size={12} /> Failed
      </span>
    );
  }
  if (status === 'error') {
    return (
      <span className="hsum__badge hsum__badge--err">
        <AlertCircle size={12} /> Error
      </span>
    );
  }
  return (
    <span className="hsum__badge hsum__badge--loading">
      <Clock size={12} /> Loading
    </span>
  );
}

function HorizonCard({ horizon, entry, active, onClick }) {
  const { snapshot, pending, error, status } = entry;
  const rec = snapshot ? pickRecommendation(snapshot.debate?.consensus_summary) : null;
  const cumReturn = snapshot?.valuation?.cumulative_return;

  return (
    <button
      type="button"
      className={`hsum__card ${active ? 'hsum__card--active' : ''}`}
      onClick={onClick}
      aria-pressed={active}
    >
      <div className="hsum__row">
        <span className="hsum__horizon">{horizon}</span>
        <span className="hsum__period">{PERIOD_LABEL[horizon]}</span>
        <StatusBadge status={status} />
      </div>
      <div className="hsum__row hsum__row--main">
        {rec ? (
          <span className={`${tagClass(rec)} hsum__rec`}>{rec}</span>
        ) : (
          <span className="hsum__rec hsum__rec--empty">—</span>
        )}
        <span className="hsum__return">{pct(cumReturn)}</span>
      </div>
      {pending && (
        <div className="hsum__sub">
          {pending.progress ? `agent: ${pending.progress}` : `job ${pending.job_id}`}
        </div>
      )}
      {error && <div className="hsum__sub hsum__sub--err">{error}</div>}
    </button>
  );
}

function HorizonSummaryStrip({ byHorizon, active, onSelect }) {
  return (
    <div className="hsum">
      {HORIZONS.map((h) => (
        <HorizonCard
          key={h}
          horizon={h}
          entry={byHorizon[h]}
          active={active === h}
          onClick={() => onSelect(h)}
        />
      ))}
    </div>
  );
}

export default HorizonSummaryStrip;
