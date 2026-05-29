import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Download, FileText } from 'lucide-react';

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

function KpiCell({ label, value, tone }) {
  return (
    <div className={`kpi-cell${tone ? ` kpi-cell--${tone}` : ''}`}>
      <div className="kpi-cell__label">{label}</div>
      <div className="kpi-cell__value">{value}</div>
    </div>
  );
}

function SnapshotView({ snapshot }) {
  const sentiment = snapshot.sentiment || {};
  const valuation = snapshot.valuation || {};
  const metrics = snapshot.metrics || {};
  const fundamental = snapshot.fundamental || {};
  const debate = snapshot.debate || {};
  const report = snapshot.report || {};
  const flags = Array.isArray(metrics.risk_flags) ? metrics.risk_flags : [];

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

      {/* Recommendation / debate consensus */}
      {debate.consensus_summary && (
        <section className="snapshot__section">
          <h3 className="snapshot__h3">Investment recommendation</h3>
          <p className="snapshot__consensus">{debate.consensus_summary}</p>
        </section>
      )}

      {/* KPI grid: sentiment + valuation + risk */}
      <section className="snapshot__section">
        <h3 className="snapshot__h3">At a glance</h3>
        <div className="kpi-grid">
          <KpiCell
            label="Sentiment"
            value={
              <span className={sentimentClass(sentiment.overall_sentiment)}>
                {sentiment.overall_sentiment || '—'}
              </span>
            }
          />
          <KpiCell label="Confidence" value={pct(sentiment.confidence_score, 0)} />
          <KpiCell label="Price trend" value={valuation.price_trend || '—'} />
          <KpiCell label="Volatility regime" value={valuation.volatility_regime || '—'} />
          <KpiCell label="Annualized return" value={pct(valuation.annualized_return)} />
          <KpiCell label="Annualized vol" value={pct(valuation.annualized_volatility)} />
          <KpiCell label="Max drawdown" value={pct(metrics.max_drawdown)} tone="warning" />
          <KpiCell label="Daily VaR 95" value={pct(metrics.daily_var_95)} tone="warning" />
          <KpiCell label="Sharpe-like" value={num(metrics.sharpe_like)} />
          <KpiCell label="Health score" value={num(fundamental.financial_health_score, 1)} />
        </div>
        {flags.length > 0 && (
          <div className="snapshot__flags">
            {flags.map((f) => (
              <span key={f} className="tag tag--negative">{f}</span>
            ))}
          </div>
        )}
      </section>

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
