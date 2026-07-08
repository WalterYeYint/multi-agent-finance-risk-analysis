import React, { useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, CartesianGrid,
} from 'recharts';
import { usePriceSeries } from '../hooks/usePriceSeries';

const PERIOD_BY_HORIZON = { SHORT: '1mo', MID: '6mo', LONG: '2y' };
const HORIZON_LABEL = { SHORT: 'Short — 1 month', MID: 'Mid — 6 months', LONG: 'Long — 2 years' };

// Cap the number of points actually handed to Recharts. The LONG horizon is a
// ~2y *daily* series (~500 points); reconciling that many SVG vertices on every
// parent re-render (the TickerView polls all three horizons every 3s) is what
// made the chart lag. Uniformly striding down to ~MAX_POINTS keeps the line
// visually identical at chart resolution while cutting reconciliation cost.
const MAX_POINTS = 180;

function downsample(series) {
  if (series.length <= MAX_POINTS) return series;
  const stride = Math.ceil(series.length / MAX_POINTS);
  const out = [];
  for (let i = 0; i < series.length; i += stride) out.push(series[i]);
  // Always keep the last point so the headline "last price" lines up with the
  // right edge of the line.
  const last = series[series.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

function formatDate(d, period) {
  if (!d) return '';
  // YYYY-MM-DD → DD MMM (1mo / 6mo) or MMM yyyy (2y) for axis compactness.
  const [y, m, day] = d.split('-');
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const mi = parseInt(m, 10) - 1;
  if (period === '2y') return `${months[mi]} ${y.slice(2)}`;
  return `${parseInt(day, 10)} ${months[mi]}`;
}

function ChartTooltip({ active, payload, period }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="price-chart__tooltip">
      <div className="price-chart__tooltip-date">{formatDate(p.date, period)}</div>
      <div className="price-chart__tooltip-price">
        ${p.close.toFixed(2)}
      </div>
    </div>
  );
}

function PriceChart({ ticker, horizon }) {
  const period = PERIOD_BY_HORIZON[horizon] || '1mo';
  const { series, loading, error } = usePriceSeries(ticker, period);

  // Compute return over the period for the header — single source of truth
  // for the chart's headline metric. Derived from the full series (not the
  // downsampled one) so first/last/min/max stay exact.
  const stats = useMemo(() => {
    if (!series.length) return null;
    const first = series[0].close;
    const last = series[series.length - 1].close;
    const change = (last - first) / first;
    let min = Infinity;
    let max = -Infinity;
    for (const p of series) {
      if (p.close < min) min = p.close;
      if (p.close > max) max = p.close;
    }
    return { first, last, change, min, max };
  }, [series]);

  // The series actually rendered by Recharts, memoized so the (potentially
  // ~500-point) transform runs only when the fetched series changes, not on
  // every parent re-render.
  const chartSeries = useMemo(() => downsample(series), [series]);

  return (
    <div className="price-chart">
      <div className="price-chart__head">
        <div>
          <div className="price-chart__title">{ticker} price</div>
          <div className="price-chart__subtitle">{HORIZON_LABEL[horizon]}</div>
        </div>
        {stats && (
          <div className="price-chart__head-right">
            <div className="price-chart__last">${stats.last.toFixed(2)}</div>
            <div
              className={`price-chart__change ${
                stats.change >= 0 ? 'price-chart__change--up' : 'price-chart__change--down'
              }`}
            >
              {stats.change >= 0 ? '▲' : '▼'} {(stats.change * 100).toFixed(2)}%
            </div>
          </div>
        )}
      </div>

      <div className="price-chart__body">
        {loading && <div className="price-chart__muted">Loading price history…</div>}
        {!loading && error && (
          <div className="price-chart__muted">Could not load prices: {error}</div>
        )}
        {!loading && !error && series.length === 0 && (
          <div className="price-chart__muted">No price data available for this period.</div>
        )}
        {!loading && !error && series.length > 0 && (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartSeries} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={(d) => formatDate(d, period)}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: 'var(--border)' }}
                minTickGap={40}
              />
              <YAxis
                domain={['auto', 'auto']}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={50}
              />
              {stats && <ReferenceLine y={stats.first} stroke="var(--border)" strokeDasharray="2 4" />}
              <Tooltip content={<ChartTooltip period={period} />} cursor={{ stroke: 'var(--brand)', strokeDasharray: '2 4' }} />
              <Line
                type="monotone"
                dataKey="close"
                stroke={stats && stats.change >= 0 ? 'var(--success)' : 'var(--danger)'}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

// Memoized: the parent TickerView re-renders every ~3s while its snapshot
// poll loops run, but PriceChart's props (ticker, horizon) are stable strings,
// so there is no reason to re-render (and re-reconcile the Recharts SVG) then.
export default React.memo(PriceChart);
