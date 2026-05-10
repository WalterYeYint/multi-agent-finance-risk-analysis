import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import jsPDF from 'jspdf';
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  BarChart3,
  Database,
  Download,
  FileText,
  GitBranch,
  Newspaper,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import './App.css';

const initialFormState = {
    ticker: 'AAPL',
    period: '1wk',
    interval: '1d',
    horizon_days: 30,
    mode: 'chain' // change this to chain or debate mode
};

const RESULT_PAGES = [
  { id: 'overview', label: 'Executive Overview', icon: BarChart3 },
  { id: 'market', label: 'Market Intelligence', icon: TrendingUp },
  { id: 'report', label: 'Full Report', icon: FileText },
  { id: 'debate', label: 'Agent Debate', icon: GitBranch },
  { id: 'raw', label: 'Raw Data', icon: Database },
];

const toNumber = (value) => {
  if (typeof value === 'number') return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const toPercent = (value, digits = 2) => {
  const numeric = toNumber(value);
  if (numeric === null) return 'N/A';
  return `${(numeric * 100).toFixed(digits)}%`;
};

const toFixedNumber = (value, digits = 2) => {
  const numeric = toNumber(value);
  if (numeric === null) return 'N/A';
  return numeric.toFixed(digits);
};

const titleize = (value) => {
  if (!value) return 'N/A';
  return String(value)
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (match) => match.toUpperCase());
};

const recommendationTone = (text = '') => {
  const normalized = text.toLowerCase();
  if (normalized.includes('buy') || normalized.includes('bullish')) return 'positive';
  if (normalized.includes('sell') || normalized.includes('bearish')) return 'negative';
  return 'neutral';
};

const normalizeDebateEntries = (argumentBucket) => {
  if (!argumentBucket) return [];

  if (Array.isArray(argumentBucket)) {
    return argumentBucket.map((entry, index) => ({ label: `Round ${index + 1}`, content: entry }));
  }

  if (typeof argumentBucket === 'object') {
    return Object.entries(argumentBucket).map(([round, entry], index) => {
      const numericRound = Number(round);
      const hasNumericRound = Number.isFinite(numericRound);
      return {
        label: hasNumericRound ? `Round ${numericRound + 1}` : `Round ${index + 1}`,
        content: entry,
      };
    });
  }

  return [];
};

function App() {
  const [formData, setFormData] = useState(initialFormState);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [promptStatus, setPromptStatus] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [providerInfo, setProviderInfo] = useState(null);
  const [activePage, setActivePage] = useState('overview');

  useEffect(() => {
    // Detect active model/provider from backend
    const fetchModels = async () => {
      try {
        const res = await axios.get('/api/models');
        setProviderInfo(res.data);
      } catch (_) {
        // ignore – UI still works without this
      }
    };
    fetchModels();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'horizon_days' ? Number(value) : value
    }));
  };

  useEffect(() => {
    if (result) {
      setActivePage('overview');
    }
  }, [result]);

  const parsePrompt = () => {
    if (!prompt.trim()) {
      setPromptStatus({ type: 'error', message: 'Enter a natural language request to parse.' });
      return;
    }

    const normalized = prompt.trim();
    const updates = {};

    const stopwords = new Set([
      'give', 'me', 'for', 'the', 'last', 'past', 'stock', 'stocks', 'shares', 'please',
      'show', 'get', 'pull', 'interval', 'intervals', 'of', 'at', 'a', 'an', 'analysis',
      'report', 'risk', 'horizon', 'days', 'day', 'over', 'next', 'with', 'and', 'to',
      'year', 'years', 'month', 'months', 'week', 'weeks', 'data'
    ]);

    const tickerMatch =
      normalized.match(/\bstock\s+(?:for|of)\s+([A-Za-z]{1,5})\b/i) ||
      normalized.match(/\b([A-Za-z]{1,5})\b\s*(?:stock|shares|equity)/i);

    if (tickerMatch) {
      updates.ticker = tickerMatch[1].toUpperCase();
    } else {
      const candidateTickers = normalized.match(/\b[A-Za-z]{1,5}\b/g) || [];
      const tickerCandidate = candidateTickers.find(
        token => !stopwords.has(token.toLowerCase())
      );
      if (tickerCandidate) {
        updates.ticker = tickerCandidate.toUpperCase();
      }
    }

    const periodMatch = normalized.match(/(?:last|past)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)/i);
    if (periodMatch) {
      const count = parseInt(periodMatch[1], 10);
      const unit = periodMatch[2].toLowerCase();
      if (!Number.isNaN(count) && count > 0) {
        if (unit.startsWith('day')) {
          updates.period = count <= 1 ? '1d' : count <= 5 ? '5d' : '1mo';
        } else if (unit.startsWith('week')) {
          updates.period = count <= 1 ? '1wk' : '1mo';
        } else if (unit.startsWith('month')) {
          if (count === 1) updates.period = '1mo';
          else if (count === 3) updates.period = '3mo';
          else if (count === 6) updates.period = '6mo';
          else if (count >= 12) updates.period = '1y';
        } else if (unit.startsWith('year')) {
          updates.period = count >= 2 ? '2y' : '1y';
        }
      }
    }

    const intervalMatch = normalized.match(/intervals?\s*(?:of|at)?\s*(?:every\s*)?(\d+)?\s*(minute|minutes|min|hour|hours|day|days|daily|week|weekly)/i);
    if (intervalMatch) {
      const count = intervalMatch[1] ? parseInt(intervalMatch[1], 10) : 1;
      const unit = intervalMatch[2].toLowerCase();
      if (unit.startsWith('min')) {
        if (count === 1) updates.interval = '1m';
        else if (count === 5) updates.interval = '5m';
        else if (count === 15) updates.interval = '15m';
        else if (count === 30) updates.interval = '30m';
      } else if (unit.startsWith('hour')) {
        updates.interval = '1h';
      } else if (unit.startsWith('day') || unit === 'daily') {
        updates.interval = '1d';
      } else if (unit.startsWith('week') || unit === 'weekly') {
        updates.interval = '1wk';
      }
    }

    const horizonMatch =
      normalized.match(/(?:risk\s+horizon|horizon|next)\s*(?:of|for|around)?\s*(\d+)\s*day/i) ||
      normalized.match(/(?:over|for)\s+the\s+next\s+(\d+)\s*day/i);
    if (horizonMatch) {
      const days = parseInt(horizonMatch[1], 10);
      if (!Number.isNaN(days) && days > 0) {
        updates.horizon_days = Math.min(Math.max(days, 1), 365);
      }
    }

    if (Object.keys(updates).length === 0) {
      setPromptStatus({
        type: 'error',
        message: 'Could not parse the request. Update the form manually or try a different phrasing.'
      });
      return;
    }

    setFormData(prev => ({
      ...prev,
      ...updates
    }));

    const summary = Object.entries(updates)
      .map(([key, value]) => `${key.replace('_', ' ')} → ${value}`)
      .join(', ');

    setPromptStatus({
      type: 'success',
      message: `Applied prompt: ${summary}`
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/api/analyze', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while analyzing the stock');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = () => {
    if (downloading || (!result?.report_pdf_base64 && !result?.report?.markdown_report)) {
      return;
    }

    setDownloading(true);
    try {
      if (result?.report_pdf_base64) {
        const link = document.createElement('a');
        link.href = `data:application/pdf;base64,${result.report_pdf_base64}`;
        link.download = `AnalysisReport_${result.ticker || 'stock'}.pdf`;
        link.click();
        return;
      }

      const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'pt',
        format: 'a4',
      });

      const margin = 40;
      const pageWidth = doc.internal.pageSize.getWidth();
      const usableWidth = pageWidth - margin * 2;
      const lineHeight = 16;
      let cursorY = margin;

      const title = `Finance Risk Analysis — ${result.ticker || ''}`.trim();
      doc.setFont('Helvetica', 'bold');
      doc.setFontSize(18);
      doc.text(title, margin, cursorY);
      cursorY += lineHeight * 2;

      const sections = result.report.markdown_report
        .replace(/#+\s?/g, '')
        .replace(/\*\*/g, '')
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean);

      doc.setFont('Helvetica', 'normal');
      doc.setFontSize(11);

      sections.forEach(line => {
        const wrapped = doc.splitTextToSize(line, usableWidth);
        wrapped.forEach(textLine => {
          if (cursorY + lineHeight > doc.internal.pageSize.getHeight() - margin) {
            doc.addPage();
            cursorY = margin;
          }
          doc.text(textLine, margin, cursorY);
          cursorY += lineHeight;
        });
        cursorY += lineHeight * 0.5;
      });

      const filename = `${result.ticker || 'analysis'}_risk_report.pdf`;
      doc.save(filename);
    } catch (err) {
      console.error('Failed to export PDF:', err);
      setPromptStatus({
        type: 'error',
        message: 'Unable to download PDF. Please try again.',
      });
    } finally {
      setDownloading(false);
    }
  };

  const riskSummary = useMemo(() => {
    const annualVol = toNumber(result?.metrics?.annual_vol);
    const dailyVar = toNumber(result?.metrics?.daily_var_95);
    const flagsCount = result?.metrics?.risk_flags?.length || 0;

    let posture = 'Moderate';
    if ((annualVol !== null && annualVol >= 0.3) || (dailyVar !== null && dailyVar <= -0.03) || flagsCount >= 3) {
      posture = 'Elevated';
    } else if ((annualVol !== null && annualVol < 0.16) && flagsCount === 0) {
      posture = 'Contained';
    }

    return posture;
  }, [result]);

  const recommendation =
    result?.debate?.consensus_summary ||
    result?.sentiment?.investment_recommendation ||
    'No recommendation available for this run.';

  const resultTicker = result?.ticker || formData.ticker;
  const reportFindings = result?.report?.key_findings || [];
  const riskFlags = result?.metrics?.risk_flags || result?.report?.risk_flags || [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Multi-Agent Finance Risk Studio</h1>
          <p>Institution-style risk intelligence powered by collaborative AI agents.</p>
        </div>
        <div className="provider-badge">
          <Sparkles size={16} />
          {providerInfo?.provider === 'openai'
            ? `OpenAI • ${providerInfo.current_model}`
            : providerInfo?.provider === 'ollama'
            ? `Ollama • ${providerInfo.current_model}`
            : 'Provider unknown'}
        </div>
      </header>

      <div className="workspace-layout">
        <aside className="control-panel">
          <section className="panel-card">
            <h2>Run Analysis</h2>
            {/* <p>Type naturally or configure fields manually.</p>

            <div className="prompt-helper">
              <label htmlFor="prompt">Natural language prompt</label>
              <textarea
                id="prompt"
                placeholder='e.g. "Analyze GOOGL for last 3 months at daily interval with 30 day horizon"'
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <button type="button" className="secondary-btn" onClick={parsePrompt}>
                Apply Prompt
              </button>
              {promptStatus && (
                <div className={`status-banner ${promptStatus.type}`}>
                  {promptStatus.message}
                </div>
              )}
            </div> */}

            <form onSubmit={handleSubmit} className="input-form">
              <div className="form-group">
                <label>Ticker</label>
                <input
                  type="text"
                  name="ticker"
                  value={formData.ticker}
                  onChange={handleInputChange}
                  placeholder="AAPL, MSFT, GOOGL"
                  required
                />
              </div>

              <div className="form-group">
                <label>Period</label>
                <select name="period" value={formData.period} onChange={handleInputChange}>
                  <option value="1d">1 Day</option>
                  <option value="5d">5 Days</option>
                  <option value="1mo">1 Month</option>
                  <option value="3mo">3 Months</option>
                  <option value="6mo">6 Months</option>
                  <option value="1y">1 Year</option>
                  <option value="2y">2 Years</option>
                </select>
              </div>

              <div className="form-group">
                <label>Interval</label>
                <select name="interval" value={formData.interval} onChange={handleInputChange}>
                  <option value="1m">1 Minute</option>
                  <option value="5m">5 Minutes</option>
                  <option value="15m">15 Minutes</option>
                  <option value="30m">30 Minutes</option>
                  <option value="1h">1 Hour</option>
                  <option value="1d">1 Day</option>
                  <option value="1wk">1 Week</option>
                </select>
              </div>

              <div className="form-group">
                <label>Risk Horizon (days)</label>
                <input
                  type="number"
                  name="horizon_days"
                  value={formData.horizon_days}
                  onChange={handleInputChange}
                  min="1"
                  max="365"
                  required
                />
              </div>

              <button type="submit" disabled={loading} className="primary-btn">
                {loading ? 'Analyzing…' : 'Analyze Risk'}
              </button>

              {(result?.report?.markdown_report || result?.report_pdf_base64) && (
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={handleDownloadPdf}
                  disabled={downloading}
                >
                  <Download size={16} />
                  {downloading ? 'Preparing PDF…' : 'Download Report PDF'}
                </button>
              )}
            </form>
          </section>

          <section className="panel-card quick-facts">
            <h3>Session Snapshot</h3>
            <div className="mini-metric">
              <span>Ticker</span>
              <strong>{resultTicker || '—'}</strong>
            </div>
            <div className="mini-metric">
              <span>Period</span>
              <strong>{formData.period}</strong>
            </div>
            <div className="mini-metric">
              <span>Horizon</span>
              <strong>{formData.horizon_days} days</strong>
            </div>
            <div className="mini-metric">
              <span>Risk Posture</span>
              <strong>{result ? riskSummary : '—'}</strong>
            </div>
          </section>
        </aside>

        <main className="results-panel">
          {error && (
            <div className="error-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}

          {!result && !loading && (
            <section className="empty-state">
              <h2>Ready for analysis</h2>
              <p>Run a ticker analysis to see executive dashboards, market intelligence, debate logs, and export-ready reports.</p>
              <div className="workflow-grid">
                <article>
                  <h4>Collaboration Workflow</h4>
                  <img src="/visualizations/langgraph_collaboration.png" alt="Collaboration workflow" />
                </article>
                <article>
                  <h4>Debate Workflow</h4>
                  <img src="/visualizations/langgraph_debate.png" alt="Debate workflow" />
                </article>
              </div>
            </section>
          )}

          {result && (
            <>
              <section className="result-header">
                <div>
                  <h2>{resultTicker} Risk Intelligence</h2>
                  <p>
                    {result?.report?.as_of ? `Last refreshed ${result.report.as_of}` : 'Latest run completed'}
                  </p>
                </div>
                <div className={`recommendation-chip ${recommendationTone(recommendation)}`}>
                  <ShieldCheck size={16} />
                  <span>{titleize(result?.sentiment?.investment_recommendation || 'monitor')}</span>
                </div>
              </section>

              <nav className="result-nav" aria-label="Result pages">
                {RESULT_PAGES.map((page) => {
                  const Icon = page.icon;
                  return (
                    <button
                      key={page.id}
                      type="button"
                      className={`result-nav-item ${activePage === page.id ? 'active' : ''}`}
                      onClick={() => setActivePage(page.id)}
                    >
                      <Icon size={16} />
                      {page.label}
                    </button>
                  );
                })}
              </nav>

              {activePage === 'overview' && (
                <section className="page-grid">
                  <div className="kpi-grid">
                    <article className="kpi-card">
                      <span>Cumulative Return</span>
                      <strong className={toNumber(result?.valuation?.cumulative_return) >= 0 ? 'positive' : 'negative'}>
                        {toPercent(result?.valuation?.cumulative_return)}
                      </strong>
                      <small>{toNumber(result?.valuation?.cumulative_return) >= 0 ? 'Positive momentum' : 'Recent contraction'}</small>
                    </article>

                    <article className="kpi-card">
                      <span>Annualized Volatility</span>
                      <strong>{toPercent(result?.valuation?.annualized_volatility, 1)}</strong>
                      <small>{titleize(result?.valuation?.volatility_regime)} regime</small>
                    </article>

                    <article className="kpi-card">
                      <span>95% Daily VaR</span>
                      <strong>{toPercent(result?.metrics?.daily_var_95, 2)}</strong>
                      <small>Downside estimate</small>
                    </article>

                    <article className="kpi-card">
                      <span>Sharpe-like</span>
                      <strong>{toFixedNumber(result?.metrics?.sharpe_like, 3)}</strong>
                      <small>Risk-adjusted return signal</small>
                    </article>

                    <article className="kpi-card">
                      <span>Sentiment Confidence</span>
                      <strong>{toPercent(result?.sentiment?.confidence_score)}</strong>
                      <small>{titleize(result?.sentiment?.overall_sentiment)}</small>
                    </article>

                    <article className="kpi-card">
                      <span>Risk Flags</span>
                      <strong>{riskFlags.length}</strong>
                      <small>{riskFlags.length ? 'Items require attention' : 'No critical flags detected'}</small>
                    </article>
                  </div>

                  <article className="content-card">
                    <h3>Executive Recommendation</h3>
                    <p>{recommendation}</p>
                  </article>

                  <article className="content-card">
                    <h3>Key Findings</h3>
                    {reportFindings.length > 0 ? (
                      <ul className="list-clean">
                        {reportFindings.map((finding, index) => (
                          <li key={`finding-${index}`}>
                            <Activity size={14} />
                            <span>{finding}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>No key findings were generated for this run.</p>
                    )}
                  </article>

                  <article className="content-card">
                    <h3>Risk Flags</h3>
                    {riskFlags.length > 0 ? (
                      <div className="tag-list">
                        {riskFlags.map((flag, index) => (
                          <span key={`flag-${index}`} className="tag warning">
                            <AlertTriangle size={12} />
                            {flag}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <div className="tag-list">
                        <span className="tag success">
                          <ShieldCheck size={12} />
                          No elevated risk flags
                        </span>
                      </div>
                    )}
                  </article>
                </section>
              )}

              {activePage === 'market' && (
                <section className="page-grid">
                  <article className="content-card">
                    <h3>Valuation Signals</h3>
                    <div className="metric-rows">
                      <div>
                        <span>Trend</span>
                        <strong>{titleize(result?.valuation?.price_trend)}</strong>
                      </div>
                      <div>
                        <span>Analysis Period</span>
                        <strong>{result?.valuation?.analysis_period || formData.period}</strong>
                      </div>
                      <div>
                        <span>Trading Days</span>
                        <strong>{result?.valuation?.trading_days ?? 'N/A'}</strong>
                      </div>
                      <div>
                        <span>Annual Return</span>
                        <strong>{toPercent(result?.valuation?.annualized_return)}</strong>
                      </div>
                    </div>
                    <div className="insight-row">
                      <h4>Valuation Insights</h4>
                      <ul className="list-clean">
                        {(result?.valuation?.valuation_insights || []).map((insight, index) => (
                          <li key={`valuation-${index}`}>
                            {toNumber(result?.valuation?.cumulative_return) >= 0 ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </article>

                  <article className="content-card">
                    <h3>Sentiment Summary</h3>
                    <div className="metric-rows">
                      <div>
                        <span>Overall Sentiment</span>
                        <strong>{titleize(result?.sentiment?.overall_sentiment)}</strong>
                      </div>
                      <div>
                        <span>News Items</span>
                        <strong>{result?.sentiment?.news_items_analyzed ?? result?.news?.items?.length ?? 0}</strong>
                      </div>
                      <div>
                        <span>Recommendation</span>
                        <strong>{titleize(result?.sentiment?.investment_recommendation)}</strong>
                      </div>
                    </div>
                    <ul className="list-clean">
                      {(result?.sentiment?.key_insights || []).map((insight, index) => (
                        <li key={`sentiment-${index}`}>
                          <Sparkles size={14} />
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </article>

                  <article className="content-card full-width">
                    <h3>
                      <Newspaper size={16} />
                      Recent Headlines
                    </h3>
                    {(result?.news?.items || []).length > 0 ? (
                      <div className="headline-list">
                        {result.news.items.slice(0, 12).map((item, index) => (
                          <div key={`news-${index}`} className="headline-item">
                            <p className="headline-title">{item.headline}</p>
                            <div className="headline-meta">
                              <span>{item.date || 'Unknown date'}</span>
                              <span className={`tag ${recommendationTone(item.sentiment)}`}>
                                {titleize(item.sentiment)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p>No market headlines were available for this run.</p>
                    )}
                  </article>
                </section>
              )}

              {activePage === 'report' && (
                <section className="page-grid">
                  <article className="content-card full-width">
                    <h3>Detailed Risk Analysis Report</h3>
                    <div className="markdown-content">
                      <ReactMarkdown>{result?.report?.markdown_report || 'No markdown report available.'}</ReactMarkdown>
                    </div>
                  </article>
                </section>
              )}

              {activePage === 'debate' && (
                <section className="page-grid">
                  <article className="content-card full-width">
                    <h3>Debate Consensus</h3>
                    <p>{result?.debate?.consensus_summary || 'Debate mode was not enabled for this run.'}</p>
                  </article>

                  {result?.debate?.agent_arguments ? (
                    Object.entries(result.debate.agent_arguments).map(([agentName, bucket]) => {
                      const entries = normalizeDebateEntries(bucket);
                      return (
                        <article key={agentName} className="content-card full-width">
                          <h3>{titleize(agentName)}</h3>
                          {entries.length > 0 ? (
                            <ul className="list-clean debate-list">
                              {entries.map((entry, index) => (
                                <li key={`${agentName}-${index}`}>
                                  <strong>{entry.label}:</strong>
                                  <span>{entry.content}</span>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p>No arguments available.</p>
                          )}
                        </article>
                      );
                    })
                  ) : (
                    <article className="content-card full-width">
                      <p>No debate transcript was generated for this run.</p>
                    </article>
                  )}
                </section>
              )}

              {activePage === 'raw' && (
                <section className="page-grid">
                  <article className="content-card full-width">
                    <h3>Raw Analysis Payload</h3>
                    <pre className="raw-json">{JSON.stringify(result, null, 2)}</pre>
                  </article>
                </section>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
