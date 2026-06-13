import React from 'react';
import { useNavigate } from 'react-router-dom';
import Sparkline from './Sparkline';
import { usePriceSeries } from '../hooks/usePriceSeries';

const HORIZONS = ['SHORT', 'MID', 'LONG'];
const HORIZON_PERIOD = { SHORT: '1mo', MID: '6mo', LONG: '2y' };
const HORIZON_LABEL = { SHORT: 'Short', MID: 'Mid', LONG: 'Long' };

function tagClass(rec) {
  if (rec === 'BUY') return 'tag tag--positive';
  if (rec === 'SELL') return 'tag tag--negative';
  if (rec === 'HOLD') return 'tag tag--neutral';
  return 'tag';
}

function RecPill({ recommendation }) {
  if (!recommendation) return <span className="overview-table__empty">—</span>;
  return <span className={tagClass(recommendation)}>{recommendation}</span>;
}

/**
 * Per-cell price sparkline. Fetches lazily — only once this cell is actually
 * mounted — against the 15-min server-cached /api/price endpoint. Note the
 * fanout: N tickers × 3 horizons = 3N requests on first render; fine for the
 * current tracked-ticker counts, but if the table grows large, gate mounting
 * behind an IntersectionObserver.
 */
function SparkCell({ ticker, period }) {
  const { series, loading } = usePriceSeries(ticker, period);
  if (loading) return <span className="overview-table__empty">…</span>;

  const closes = (series || []).map((p) => p.close);
  const nums = closes.filter((v) => typeof v === 'number' && Number.isFinite(v));
  const up = nums.length < 2 || nums[nums.length - 1] >= nums[0];
  return (
    <Sparkline
      values={closes}
      width={120}
      height={36}
      stroke={up ? 'var(--success)' : 'var(--danger)'}
      fill={up ? 'rgba(15, 157, 88, 0.12)' : 'rgba(220, 38, 38, 0.12)'}
    />
  );
}

function OverviewRow({ entry }) {
  const navigate = useNavigate();
  const go = (horizon) =>
    navigate(`/t/${encodeURIComponent(entry.ticker)}?horizon=${horizon}`);

  return (
    <tr
      className="overview-table__row"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter') go('SHORT');
      }}
    >
      <td className="overview-table__sym" onClick={() => go('SHORT')}>
        {entry.ticker}
      </td>
      {HORIZONS.map((h) => (
        <React.Fragment key={h}>
          <td className="overview-table__rec" onClick={() => go(h)}>
            <RecPill recommendation={entry.horizons?.[h]?.recommendation} />
          </td>
          <td className="overview-table__spark" onClick={() => go(h)}>
            <SparkCell ticker={entry.ticker} period={HORIZON_PERIOD[h]} />
          </td>
        </React.Fragment>
      ))}
    </tr>
  );
}

/** Landing-page overview: one row per tracked ticker, a recommendation pill +
 *  price sparkline per horizon. Clicking any cell opens that horizon's view. */
function OverviewTable({ tickers }) {
  return (
    <div className="overview-table">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            {HORIZONS.map((h) => (
              <React.Fragment key={h}>
                <th>{HORIZON_LABEL[h]}</th>
                <th className="overview-table__spark">{HORIZON_LABEL[h]} graph</th>
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {tickers.map((t) => (
            <OverviewRow key={t.ticker} entry={t} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default OverviewTable;
