import React from 'react';
import { Activity, Calculator } from 'lucide-react';
import KpiCell from './KpiCell';

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

const Cell = KpiCell;

// Hover/focus explanations. Thresholds mirror src/utils/insights.py
// (compute_technical_signals / compute_valuation_ratios) — keep in sync.
const HELP = {
  trend: 'Moving-average relationship: golden cross = 50-day average above the 200-day (bullish), death cross = below it (bearish). With under 200 days of history, uptrend/downtrend simply compares the last close to the longest available average.',
  momentum: 'Trailing 3-month return bucketed: strong up ≥ +20%, up ≥ +5%, flat within ±5%, down ≤ −5%, strong down ≤ −20%. Uses another window if 3 months of data is missing.',
  rsi: '14-day Relative Strength Index, 0–100. Above 70 = overbought (rally may be stretched); below 30 = oversold (sell-off may be stretched); in between = neutral.',
  lastClose: 'Most recent daily closing price in the stored price series (as of the last trading day the data covers).',
  return1m: 'Price change over the trailing 30 calendar days, excluding dividends.',
  return3m: 'Price change over the trailing 91 calendar days (about one quarter), excluding dividends.',
  return6m: 'Price change over the trailing 182 calendar days, excluding dividends.',
  return1y: 'Price change over the trailing 365 calendar days, excluding dividends.',
  sma50: 'How far the last close sits above (+) or below (−) its 50-day simple moving average — a short-term trend gauge. Far above can mean overextended; below can signal weakness.',
  sma200: 'How far the last close sits above (+) or below (−) its 200-day simple moving average — the classic long-term trend line. Below it is often read as a bear signal.',
  high52: 'Distance of the last close below its highest close of the past 52 weeks. 0% means the stock is at its yearly high.',
  low52: 'Distance of the last close above its lowest close of the past 52 weeks. 0% means the stock is at its yearly low.',
  pe: 'Price-to-earnings: last close ÷ diluted EPS from the latest annual SEC filing. Higher means investors pay more per dollar of profit. Omitted when EPS is zero or negative.',
  ps: 'Price-to-sales: market cap ÷ annual revenue. Useful for comparing companies whose earnings are negative or volatile.',
  netMargin: 'Net income ÷ revenue for the fiscal year — how many cents of profit each dollar of sales produces.',
  roe: 'Return on equity: net income ÷ shareholders\' equity. How efficiently the company turns its equity into profit; only shown when equity is positive.',
  de: 'Total liabilities ÷ shareholders\' equity (a broad leverage measure — all liabilities, not just borrowings). Higher means more of the business is funded by obligations rather than equity.',
  marketCap: 'Last close × diluted shares outstanding reported in the filing — the market\'s total valuation of the company.',
  eps: 'Diluted earnings per share for the fiscal year, from the company\'s XBRL financial data (SEC).',
  revenue: 'Total revenue for the fiscal year, from the latest annual SEC filing.',
};

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
              help={HELP.trend}
              value={<span className={tagClass(TREND_TONE[t.trend])}>{TREND_LABEL[t.trend] || '—'}</span>}
            />
            <Cell
              label="Momentum"
              help={HELP.momentum}
              value={<span className={tagClass(MOMENTUM_TONE[t.momentum])}>{MOMENTUM_LABEL[t.momentum] || '—'}</span>}
            />
            <Cell
              label="RSI (14)"
              help={HELP.rsi}
              value={t.rsi_14 != null ? `${num(t.rsi_14, 0)} · ${t.rsi_zone || ''}` : '—'}
              tone={t.rsi_zone === 'overbought' || t.rsi_zone === 'oversold' ? 'warning' : undefined}
            />
            <Cell label="Last close" help={HELP.lastClose} value={t.last_close != null ? `$${num(t.last_close)}` : '—'} />
            <Cell label="Return 1M" help={HELP.return1m} value={signedPct(t.return_1m)} tone={returnTone(t.return_1m)} />
            <Cell label="Return 3M" help={HELP.return3m} value={signedPct(t.return_3m)} tone={returnTone(t.return_3m)} />
            <Cell label="Return 6M" help={HELP.return6m} value={signedPct(t.return_6m)} tone={returnTone(t.return_6m)} />
            <Cell label="Return 1Y" help={HELP.return1y} value={signedPct(t.return_1y)} tone={returnTone(t.return_1y)} />
            <Cell label="vs 50-day SMA" help={HELP.sma50} value={signedPct(t.pct_vs_sma_50)} tone={returnTone(t.pct_vs_sma_50)} />
            <Cell label="vs 200-day SMA" help={HELP.sma200} value={signedPct(t.pct_vs_sma_200)} tone={returnTone(t.pct_vs_sma_200)} />
            <Cell label="From 52w high" help={HELP.high52} value={signedPct(t.pct_from_52w_high)} tone={returnTone(t.pct_from_52w_high)} />
            <Cell label="From 52w low" help={HELP.low52} value={signedPct(t.pct_from_52w_low)} tone={returnTone(t.pct_from_52w_low)} />
          </div>
        </div>
      )}

      {v && (
        <div className="insights__block">
          <h4 className="snapshot__h4">
            <Calculator size={14} /> Valuation ratios{v.as_of_fy ? ` · FY${v.as_of_fy}` : ''}
          </h4>
          <div className="kpi-grid">
            <Cell label="P/E" help={HELP.pe} value={num(v.pe_ratio)} />
            <Cell label="P/S" help={HELP.ps} value={num(v.ps_ratio)} />
            <Cell label="Net margin" help={HELP.netMargin} value={pct(v.net_margin)} />
            <Cell label="ROE" help={HELP.roe} value={pct(v.roe)} />
            <Cell label="Debt / equity" help={HELP.de} value={num(v.debt_to_equity)} />
            <Cell label="Market cap" help={HELP.marketCap} value={money(v.market_cap)} />
            <Cell label="EPS (diluted)" help={HELP.eps} value={v.eps_diluted != null ? `$${num(v.eps_diluted)}` : '—'} />
            <Cell label="Revenue" help={HELP.revenue} value={money(v.revenue)} />
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
