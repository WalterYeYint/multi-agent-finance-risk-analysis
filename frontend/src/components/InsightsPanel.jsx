import React from 'react';
import { Activity, Calculator } from 'lucide-react';

// N2: public-analyzer insights — technical/momentum + valuation ratios + a short
// analyst-style summary. All fields are best-effort; anything missing renders '—'.

function pct(v, digits = 1) {
  return typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : '—';
}
function signedPct(v, digits = 1) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const s = `${(v * 100).toFixed(digits)}%`;
  return v > 0 ? `+${s}` : s;
}
function num(v, digits = 2) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
}
function money(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toFixed(0)}`;
}
function returnTone(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return undefined;
  return v > 0 ? 'positive' : v < 0 ? 'warning' : undefined;
}

function Cell({ label, value, tone }) {
  return (
    <div className={`kpi-cell${tone ? ` kpi-cell--${tone}` : ''}`}>
      <div className="kpi-cell__label">{label}</div>
      <div className="kpi-cell__value">{value}</div>
    </div>
  );
}

const TREND_LABEL = {
  golden_cross: 'Golden cross',
  death_cross: 'Death cross',
  uptrend: 'Uptrend',
  downtrend: 'Downtrend',
  neutral: 'Neutral',
};
const TREND_TONE = { golden_cross: 'positive', uptrend: 'positive', death_cross: 'negative', downtrend: 'negative' };
const MOMENTUM_LABEL = {
  strong_up: 'Strong up', up: 'Up', flat: 'Flat', down: 'Down', strong_down: 'Strong down',
};
const MOMENTUM_TONE = { strong_up: 'positive', up: 'positive', down: 'negative', strong_down: 'negative' };

function tagClass(tone) {
  if (tone === 'positive') return 'tag tag--positive';
  if (tone === 'negative') return 'tag tag--negative';
  return 'tag tag--neutral';
}

function InsightsPanel({ insights }) {
  if (!insights) return null;
  const t = insights.technical || null;
  const v = insights.valuation || null;
  const summary = insights.analyst_summary;
  if (!t && !v && !summary) return null;

  return (
    <section className="snapshot__section insights">
      <h3 className="snapshot__h3">Public-analyzer insights</h3>

      {t && (
        <div className="insights__block">
          <h4 className="snapshot__h4"><Activity size={14} /> Technical &amp; momentum</h4>
          <div className="kpi-grid">
            <Cell
              label="Trend"
              value={<span className={tagClass(TREND_TONE[t.trend])}>{TREND_LABEL[t.trend] || '—'}</span>}
            />
            <Cell
              label="Momentum"
              value={<span className={tagClass(MOMENTUM_TONE[t.momentum])}>{MOMENTUM_LABEL[t.momentum] || '—'}</span>}
            />
            <Cell
              label="RSI (14)"
              value={t.rsi_14 != null ? `${num(t.rsi_14, 0)} · ${t.rsi_zone || ''}` : '—'}
              tone={t.rsi_zone === 'overbought' || t.rsi_zone === 'oversold' ? 'warning' : undefined}
            />
            <Cell label="Last close" value={t.last_close != null ? `$${num(t.last_close)}` : '—'} />
            <Cell label="Return 1M" value={signedPct(t.return_1m)} tone={returnTone(t.return_1m)} />
            <Cell label="Return 3M" value={signedPct(t.return_3m)} tone={returnTone(t.return_3m)} />
            <Cell label="Return 6M" value={signedPct(t.return_6m)} tone={returnTone(t.return_6m)} />
            <Cell label="Return 1Y" value={signedPct(t.return_1y)} tone={returnTone(t.return_1y)} />
            <Cell label="vs 50-day SMA" value={signedPct(t.pct_vs_sma_50)} tone={returnTone(t.pct_vs_sma_50)} />
            <Cell label="vs 200-day SMA" value={signedPct(t.pct_vs_sma_200)} tone={returnTone(t.pct_vs_sma_200)} />
            <Cell label="From 52w high" value={signedPct(t.pct_from_52w_high)} tone={returnTone(t.pct_from_52w_high)} />
            <Cell label="From 52w low" value={signedPct(t.pct_from_52w_low)} tone={returnTone(t.pct_from_52w_low)} />
          </div>
        </div>
      )}

      {v && (
        <div className="insights__block">
          <h4 className="snapshot__h4">
            <Calculator size={14} /> Valuation ratios{v.as_of_fy ? ` · FY${v.as_of_fy}` : ''}
          </h4>
          <div className="kpi-grid">
            <Cell label="P/E" value={num(v.pe_ratio)} />
            <Cell label="P/S" value={num(v.ps_ratio)} />
            <Cell label="Net margin" value={pct(v.net_margin)} />
            <Cell label="ROE" value={pct(v.roe)} />
            <Cell label="Debt / equity" value={num(v.debt_to_equity)} />
            <Cell label="Market cap" value={money(v.market_cap)} />
            <Cell label="EPS (diluted)" value={v.eps_diluted != null ? `$${num(v.eps_diluted)}` : '—'} />
            <Cell label="Revenue" value={money(v.revenue)} />
          </div>
          {Array.isArray(v.notes) && v.notes.length > 0 && (
            <p className="insights__notes">{v.notes.join(' · ')}</p>
          )}
          {v.source && <p className="insights__source">Source: {v.source}</p>}
        </div>
      )}

      {summary && (
        <div className="insights__block">
          <h4 className="snapshot__h4">Analyst take</h4>
          <p className="snapshot__copy">{summary}</p>
        </div>
      )}
    </section>
  );
}

export default React.memo(InsightsPanel);
