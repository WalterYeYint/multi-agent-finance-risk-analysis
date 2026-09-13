import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Download, FileText } from 'lucide-react';
import InsightsPanel from './InsightsPanel';
import KpiCell from './KpiCell';
import { pickRecommendation, recTone } from '../utils/recommendation';

function pct(v, digits = 1) {
  return typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : '—';
}
function num(v, digits = 2) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
}
function formatWhen(iso) {
  if (!iso) return '—';
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return iso;
  return t.toLocaleString();
}

function sentimentClass(label) {
  if (label === 'bullish') return 'tag tag--positive';
  if (label === 'bearish') return 'tag tag--negative';
  return 'tag tag--neutral';
}

function downloadPdf(ticker, horizon, base64) {
  if (!base64) return;
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${ticker}-${horizon}-report.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Hover/focus explanations for the "At a glance" grid. Wording mirrors how each
// number is actually computed (src/utils/tools.py compute_risk / valuation,
// src/agents.py risk + fundamental agents) — keep in sync if those change.
const HELP = {
  sentiment: 'Overall tone of recent news coverage as judged by the sentiment agent: bullish, bearish, or neutral.',
  confidence: 'How sure the sentiment agent is of its bullish/bearish call, 0–100%. It is the agent\'s self-reported certainty, not a probability that the stock moves that way.',
  priceTrend: 'Direction of price over this horizon\'s lookback window: "upward" if the close rose more than 5% start-to-end, "downward" if it fell more than 5%, otherwise "sideways".',
  volRegime: 'Annualized volatility bucketed: low below 15%, medium 15–30%, high above 30%. Higher regimes mean bigger typical price swings.',
  annReturn: 'The lookback period\'s total return compounded to a one-year rate (252 trading days). Short windows can produce extreme annualized figures.',
  annVol: 'Standard deviation of daily returns scaled to one year (× √252). A 40% figure means a typical year sees ±40% swings.',
  maxDrawdown: 'Largest peak-to-trough fall in the lookback window — what you would have lost buying at the worst high and selling at the following low. Beyond −25% raises a DEEP_DRAWDOWN flag.',
  var95: 'One-day Value at Risk at 95%: a single-day loss at least this large is expected on roughly 1 in 20 trading days, assuming normally distributed returns.',
  sharpe: 'Annualized average daily return divided by its volatility (no risk-free rate subtracted). Roughly: above 1 is good return per unit of risk, near 0 is poor, negative means losing money for the risk taken.',
  health: 'The fundamental agent\'s 0–10 rating of financial health from the latest 10-K/10-Q — profitability, balance sheet strength and cash flow. 10 is strongest.',
};

function SnapshotView({ snapshot }) {
  const sentiment = snapshot.sentiment || {};
  const valuation = snapshot.valuation || {};
  const metrics = snapshot.metrics || {};
  const fundamental = snapshot.fundamental || {};
  const debate = snapshot.debate || {};
  const report = snapshot.report || {};
  const flags = Array.isArray(metrics.risk_flags) ? metrics.risk_flags : [];
  const rec = pickRecommendation(debate.consensus_summary);

  return (
    <div className="snapshot">
      {/* Header strip */}
      <div className="snapshot__head">
        <div className="snapshot__head-left">
          <span className={`tag ${snapshot.cached ? 'tag--neutral' : 'tag--positive'}`}>
            {snapshot.cached ? 'Cached' : 'Just generated'}
          </span>
          <span className="snapshot__when">Generated {formatWhen(snapshot.generated_at)}</span>
          {typeof snapshot.latency_ms === 'number' && (
            <span className="snapshot__latency">{(snapshot.latency_ms / 1000).toFixed(1)}s pipeline</span>
          )}
        </div>
        {snapshot.report_pdf_base64 && (
          <button
            type="button"
            className="ghost-btn"
            onClick={() => downloadPdf(snapshot.ticker, snapshot.horizon, snapshot.report_pdf_base64)}
          >
            <Download size={14} /> Download PDF
          </button>
        )}
      </div>

      {/* Recommendation / debate consensus — prominent color-coded verdict (N1) */}
      {(rec || debate.consensus_summary) && (
        <section className="snapshot__section rec-hero">
          <div className={`rec-hero__badge rec-hero__badge--${recTone(rec) || 'neutral'}`}>
            <span className="rec-hero__verdict">{rec || 'N/A'}</span>
            <span className="rec-hero__caption">recommendation</span>
          </div>
          <div className="rec-hero__body">
            <h3 className="snapshot__h3">Investment recommendation</h3>
            {debate.consensus_summary ? (
              <p className="snapshot__consensus">{debate.consensus_summary}</p>
            ) : (
              <p className="snapshot__consensus snapshot__muted">
                No consensus summary was produced for this run.
              </p>
            )}
          </div>
        </section>
      )}

      {/* KPI grid: sentiment + valuation + risk */}
      <section className="snapshot__section">
        <h3 className="snapshot__h3">At a glance</h3>
        <div className="kpi-grid">
          <KpiCell
            label="Sentiment"
            help={HELP.sentiment}
            value={
              <span className={sentimentClass(sentiment.overall_sentiment)}>
                {sentiment.overall_sentiment || '—'}
              </span>
            }
          />
          <KpiCell label="Confidence" help={HELP.confidence} value={pct(sentiment.confidence_score, 0)} />
          <KpiCell label="Price trend" help={HELP.priceTrend} value={valuation.price_trend || '—'} />
          <KpiCell label="Volatility regime" help={HELP.volRegime} value={valuation.volatility_regime || '—'} />
          <KpiCell label="Annualized return" help={HELP.annReturn} value={pct(valuation.annualized_return)} />
          <KpiCell label="Annualized vol" help={HELP.annVol} value={pct(valuation.annualized_volatility)} />
          <KpiCell label="Max drawdown" help={HELP.maxDrawdown} value={pct(metrics.max_drawdown)} tone="warning" />
          <KpiCell label="Daily VaR 95" help={HELP.var95} value={pct(metrics.daily_var_95)} tone="warning" />
          <KpiCell label="Sharpe-like" help={HELP.sharpe} value={num(metrics.sharpe_like)} />
          <KpiCell label="Health score" help={HELP.health} value={num(fundamental.financial_health_score, 1)} />
        </div>
        {flags.length > 0 && (
          <div className="snapshot__flags">
            {flags.map((f) => (
              <span key={f} className="tag tag--negative">{f}</span>
            ))}
          </div>
        )}
      </section>

      {/* N2: public-analyzer insights (technical/momentum + valuation ratios) */}
      <InsightsPanel insights={snapshot.insights} />

      {/* Investment recommendation from sentiment + fundamental */}
      {(sentiment.investment_recommendation || sentiment.summary) && (
        <section className="snapshot__section">
          <h3 className="snapshot__h3">Sentiment summary</h3>
          {sentiment.investment_recommendation && (
            <p className="snapshot__rec">{sentiment.investment_recommendation}</p>
          )}
          {sentiment.summary && <p className="snapshot__copy">{sentiment.summary}</p>}
          {Array.isArray(sentiment.key_insights) && sentiment.key_insights.length > 0 && (
            <ul className="snapshot__list">
              {sentiment.key_insights.map((k, i) => (
                <li key={i}>{k}</li>
              ))}
            </ul>
          )}
        </section>
      )}

      {fundamental.executive_summary && (
        <section className="snapshot__section">
          <h3 className="snapshot__h3">Fundamental — {fundamental.filing_type || 'filing'}</h3>
          <p className="snapshot__copy">{fundamental.executive_summary}</p>
          <div className="snapshot__two-col">
            <div>
              <h4 className="snapshot__h4">Business highlights</h4>
              <ul className="snapshot__list">
                {(fundamental.business_highlights || []).map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="snapshot__h4">Risk factors</h4>
              <ul className="snapshot__list">
                {(fundamental.risk_factors || []).map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      {report.markdown_report && (
        <details className="snapshot__report">
          <summary>
            <FileText size={14} /> Full markdown report
          </summary>
          <div className="snapshot__markdown">
            <ReactMarkdown>{report.markdown_report}</ReactMarkdown>
          </div>
        </details>
      )}
    </div>
  );
}

export default SnapshotView;
