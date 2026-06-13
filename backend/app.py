from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import re
import traceback
import requests
import io
import base64
import openai

# Ensure we can import project code from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.horizons import get_horizon, HORIZONS
from utils.snapshots import (
    get_latest_snapshot, is_fresh, list_snapshot_history, list_tracked_tickers,
    list_latest_snapshots_overview, get_or_create_pending_job, get_job,
)

from markdown_pdf import MarkdownPdf, Section

from threading import Lock
from cachetools import TTLCache, cached

app = Flask(__name__)
CORS(app)


def _detect_model_provider():
    raw = (os.getenv('MODEL_PROVIDER') or 'auto').strip().lower()
    if not raw:
        raw = 'auto'
    openai_key = os.getenv('OPENAI_API_KEY')
    if raw == 'auto':
        provider = 'openai' if openai_key else 'ollama'
    else:
        provider = raw
    return provider, openai_key


def _is_ollama_available():
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    try:
        requests.get(f"{base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


@app.route('/', methods=['GET'])
def root():
    """Root endpoint for load-balancer health checks. ECS Express Mode's
    auto-provisioned ALB defaults its health-check path to '/' (the deploy
    action doesn't expose a way to override it), so this must return 200 and
    must NOT touch the DB — otherwise the target is marked unhealthy and the
    ALB serves 503 to everything."""
    return jsonify({'status': 'ok', 'service': 'finance-risk-analysis-api'})


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Finance Risk Analysis API is running'})


@app.route('/favicon.ico')
def favicon_placeholder():
    """Return an empty response so browsers stop logging 404s for favicons."""
    return ('', 204)


@app.route('/api/models', methods=['GET'])
def get_available_models():
    raw_provider = (os.getenv('MODEL_PROVIDER') or 'auto').strip().lower() or 'auto'
    provider, openai_key = _detect_model_provider()

    if provider == 'openai':
        if not openai_key:
            return jsonify({
                'error': (
                    "OpenAI GPT-4o not configured. Set OPENAI_API_KEY or set MODEL_PROVIDER=ollama."
                )
            }), 503
        current_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        description = 'OpenAI GPT model'
        models = [current_model]
    elif provider == 'ollama':
        if not _is_ollama_available():
            return jsonify({
                'error': 'Ollama is not reachable. Start it with: ollama serve'
            }), 503
        current_model = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
        description = 'Local Ollama model'
        models = [current_model]
    else:
        return jsonify({
            'error': f"Unknown MODEL_PROVIDER: {raw_provider}. Must be 'openai' or 'ollama'."
        }), 400

    return jsonify({
        'provider': provider,
        'models': models,
        'current_model': current_model,
        'description': description
    })

def _attach_pdf(resp: dict) -> dict:
    """Render the snapshot's markdown report to a base64 PDF (best-effort)."""
    md = (resp.get('report') or {}).get('markdown_report')
    if md:
        try:
            pdf = MarkdownPdf(toc_level=2, optimize=True)
            pdf.add_section(Section(md))
            out = io.BytesIO()
            pdf.save_bytes(out)
            out.seek(0)
            resp['report_pdf_base64'] = base64.b64encode(out.read()).decode('utf-8')
        except Exception as e:
            print(f"⚠️  PDF render failed: {e}")
    return resp


def _serialize_snapshot(snap: dict, *, cached: bool) -> dict:
    """Map a `snapshots` row into the API response. The agent fields are already
    stored as JSONB (model dumps), so this is a thin reshape — no more hand-rolled
    safe_get walking."""
    debate = snap.get('debate') or {}
    return {
        'ticker': snap['ticker'],
        'horizon': snap['horizon'],
        'generated_at': snap['generated_at'].isoformat() if snap.get('generated_at') else None,
        'cached': cached,
        'sentiment': snap.get('sentiment'),
        'fundamental': snap.get('fundamental'),
        'valuation': snap.get('valuation'),
        'metrics': snap.get('metrics'),
        'debate': {'consensus_summary': debate.get('consensus_summary')} if debate else None,
        'agent_arguments': debate.get('agent_arguments'),
        'report': {
            'markdown_report': snap.get('report_markdown'),
            'consensus_summary': debate.get('consensus_summary'),
        },
        'cost_usd': float(snap['cost_usd']) if snap.get('cost_usd') is not None else None,
        'latency_ms': snap.get('latency_ms'),
    }


def _job_response(job: dict, http_status: int = 202):
    """Shape a `jobs` row for the polling client."""
    return jsonify({
        'status': job['status'],
        'job_id': job['id'],
        'ticker': job['ticker'],
        'horizon': job['horizon'],
        'progress': job.get('progress'),
        'error': job.get('error'),
        'message': ('Snapshot is being generated by the worker. Poll this URL '
                    'or GET /api/jobs/{} until status is "ready".'.format(job['id'])),
    }), http_status


# In-process TTL cache for the snapshot read path. Collapses thundering-herd
# duplicates of the same (ticker, horizon) read down to one DB roundtrip per TTL
# window per backend instance — important because the worker writes asynchronously
# and the read path is many-to-one.
#
# TTL <= 60s ensures worker writes propagate within a minute, which is well inside
# every horizon's freshness window (Short 24h / Mid 72h / Long 168h). Cache keys
# are normalized at every call site (ticker uppercased) so 'aapl' and 'AAPL' share
# an entry.
_SNAPSHOT_TTL_S = 60
_TICKERS_TTL_S = 300
_OVERVIEW_TTL_S = 60
# Price data updates intraday but we don't need second-by-second freshness for a
# research chart. 15 min is a fair compromise between yfinance load and UI freshness.
_PRICE_TTL_S = 900
_cache_lock = Lock()
_snapshot_cache: TTLCache = TTLCache(maxsize=1024, ttl=_SNAPSHOT_TTL_S)
_history_cache: TTLCache = TTLCache(maxsize=512, ttl=_SNAPSHOT_TTL_S)
_tickers_cache: TTLCache = TTLCache(maxsize=4, ttl=_TICKERS_TTL_S)
_price_cache: TTLCache = TTLCache(maxsize=256, ttl=_PRICE_TTL_S)
_overview_cache: TTLCache = TTLCache(maxsize=1, ttl=_OVERVIEW_TTL_S)


@cached(cache=_snapshot_cache, lock=_cache_lock)
def _cached_latest_snapshot(ticker: str, horizon_name: str):
    return get_latest_snapshot(ticker, horizon_name)


@cached(cache=_history_cache, lock=_cache_lock)
def _cached_snapshot_history(ticker: str, horizon_name: str, days: int):
    return list_snapshot_history(ticker, horizon_name, days=days)


@cached(cache=_tickers_cache, lock=_cache_lock)
def _cached_tracked_tickers():
    return list_tracked_tickers()


@cached(cache=_overview_cache, lock=_cache_lock)
def _cached_overview():
    return list_latest_snapshots_overview()


# yfinance is NOT thread-safe: concurrent yf.download() calls clobber each other
# through shared global state, so three simultaneous (ticker, period) fetches all
# return whichever period won the race — which made every horizon's sparkline
# identical on the overview table. We serialize every download behind this lock
# AND fetch each ticker's full 2y series exactly once, slicing it for the shorter
# windows. That removes the race entirely, guarantees the three horizons are
# consistent slices of the same data, and cuts the N×3 fanout to N.
_yf_lock = Lock()

# Trading-day windows roughly matching yfinance's 1mo / 6mo / 2y row counts.
_PERIOD_TAIL = {"1mo": 22, "6mo": 124, "2y": None}  # None = full series


@cached(cache=_price_cache, lock=_cache_lock)
def _cached_price_full(ticker: str):
    """Full 2y daily close series for `ticker`, fetched once and cached. Returns
    list of {date: 'YYYY-MM-DD', close: float}; empty list on data failure.
    Serialized behind _yf_lock because yfinance is not thread-safe."""
    import math
    import yfinance as yf  # lazy import — keeps cold-start of the backend fast
    try:
        with _yf_lock:
            df = yf.download(ticker, period="2y", interval="1d",
                             auto_adjust=False, progress=False)
        if df is None or df.empty:
            return []
        df = df.reset_index()
        # yfinance may return a MultiIndex (single ticker still); flatten if so.
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = [c[0] if c[1] in ("", ticker) else c[0] for c in df.columns]
        rows = []
        for _, r in df.iterrows():
            close = r.get("Close")
            if close is None:
                continue
            try:
                close = float(close)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(close):
                continue
            date_val = r.get("Date")
            strftime = getattr(date_val, "strftime", None)
            date_str = strftime("%Y-%m-%d") if strftime else str(date_val)[:10]
            rows.append({"date": date_str, "close": close})
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  /api/price fetch failed for {ticker}: {e}")
        return []


def _cached_price_series(ticker: str, period: str):
    """Daily close prices over `period` (1mo / 6mo / 2y), sliced from the cached
    2y series so the three horizons are always distinct, consistent windows."""
    full = _cached_price_full(ticker)
    tail = _PERIOD_TAIL.get(period)
    if tail is None or len(full) <= tail:
        return full
    return full[-tail:]


def _resolve_snapshot(ticker: str, horizon):
    """Cache-or-enqueue. Fresh cache → 200 + snapshot. Otherwise enqueue a job
    (deduped per ticker+horizon) for the worker and return 202 + job info; the
    client polls this same endpoint until the snapshot is fresh."""
    cached = _cached_latest_snapshot(ticker, horizon.name)
    if cached and is_fresh(cached, horizon):
        return jsonify(_attach_pdf(_serialize_snapshot(cached, cached=True)))

    job = get_or_create_pending_job(ticker, horizon.name)
    return _job_response(job)


def _parse_horizon(name: str):
    """Return (Horizon, None) or (None, error_response)."""
    try:
        return get_horizon(name), None
    except ValueError as e:
        return None, (jsonify({'error': str(e)}), 400)


@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """Cache-or-compute a snapshot for {ticker, horizon}. `horizon` defaults to
    MID; the legacy period/interval/horizon_days fields are ignored."""
    try:
        data = request.get_json() or {}
        ticker = str(data.get('ticker', '')).strip().upper()
        if not ticker:
            return jsonify({'error': 'Missing required field: ticker'}), 400
        horizon, err = _parse_horizon(str(data.get('horizon', 'MID')))
        if err:
            return err
        return _resolve_snapshot(ticker, horizon)
    except openai.AuthenticationError:
        return jsonify({'error': 'OpenAI authentication failed. Set a valid OPENAI_API_KEY.'}), 502
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return jsonify({'error': f'Analysis failed due to OpenAI API error: {str(e)}'}), 502
    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/snapshot/<ticker>/<horizon>', methods=['GET'])
def get_snapshot(ticker, horizon):
    """Read (or compute on miss) the snapshot for one ticker + horizon."""
    try:
        h, err = _parse_horizon(horizon)
        if err:
            return err
        return _resolve_snapshot(ticker.strip().upper(), h)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'Snapshot failed: {str(e)}'}), 500


@app.route('/api/snapshot/<ticker>/<horizon>/history', methods=['GET'])
def snapshot_history(ticker, horizon):
    """Timeseries of past snapshots for the run/risk-history view."""
    h, err = _parse_horizon(horizon)
    if err:
        return err
    try:
        days = int(request.args.get('days', 90))
    except ValueError:
        days = 90
    rows = _cached_snapshot_history(ticker.strip().upper(), h.name, days)
    history = [
        {
            'generated_at': r['generated_at'].isoformat() if r.get('generated_at') else None,
            'overall_sentiment': (r.get('sentiment') or {}).get('overall_sentiment'),
            'consensus_summary': (r.get('debate') or {}).get('consensus_summary'),
            'risk_flags': (r.get('metrics') or {}).get('risk_flags'),
            'annual_vol': (r.get('metrics') or {}).get('annual_vol'),
            'max_drawdown': (r.get('metrics') or {}).get('max_drawdown'),
        }
        for r in rows
    ]
    return jsonify({'ticker': ticker.strip().upper(), 'horizon': h.name, 'history': history})


_VALID_PRICE_PERIODS = {"1mo", "6mo", "2y"}


@app.route('/api/price/<ticker>', methods=['GET'])
def price_series(ticker):
    """Daily close prices for one of the three horizon lookback windows.
    Used by the price chart on the TickerView. Server-cached for 15 min."""
    period = (request.args.get('period') or '1mo').strip().lower()
    if period not in _VALID_PRICE_PERIODS:
        return jsonify({'error': f"period must be one of {sorted(_VALID_PRICE_PERIODS)}"}), 400
    series = _cached_price_series(ticker.strip().upper(), period)
    return jsonify({
        'ticker': ticker.strip().upper(),
        'period': period,
        'series': series,
    })


def _pick_recommendation(text):
    """Extract the BUY/HOLD/SELL token from the debate's consensus_summary.
    Mirrors the client-side logic in HorizonSummaryStrip.jsx: word-boundary
    scan in priority order — explicit "SELL" wins over "BUY" if both appear,
    because recommendations are negated ("avoid a BUY") more than promoted."""
    if not text or not isinstance(text, str):
        return None
    upper = text.upper()
    for token in ('SELL', 'BUY', 'HOLD'):
        if re.search(rf'\b{token}\b', upper):
            return token
    return None


@app.route('/api/overview', methods=['GET'])
def overview():
    """Landing-page table: latest recommendation + headline return for every
    tracked (ticker, horizon). Strictly read-only — unlike /api/snapshot, this
    NEVER enqueues pipeline jobs, so the landing page can call it freely."""
    by_ticker = {}
    for row in _cached_overview():
        horizon = row['horizon']
        if horizon not in HORIZONS:
            continue
        entry = by_ticker.setdefault(row['ticker'], {
            h: {'recommendation': None, 'cumulative_return': None, 'generated_at': None}
            for h in HORIZONS
        })
        entry[horizon] = {
            'recommendation': _pick_recommendation(row.get('consensus_summary')),
            'cumulative_return': row.get('cumulative_return'),
            'generated_at': row['generated_at'].isoformat() if row.get('generated_at') else None,
        }
    return jsonify({'tickers': [
        {'ticker': ticker, 'horizons': horizons}
        for ticker, horizons in sorted(by_ticker.items())
    ]})


@app.route('/api/tickers', methods=['GET'])
def list_tickers():
    """Tickers that have at least one snapshot, for the landing page."""
    tickers = [
        {
            'ticker': t['ticker'],
            'last_updated': t['last_updated'].isoformat() if t.get('last_updated') else None,
            'snapshot_count': t['snapshot_count'],
        }
        for t in _cached_tracked_tickers()
    ]
    return jsonify({'horizons': list(HORIZONS), 'tickers': tickers})


@app.route('/api/jobs/<int:job_id>', methods=['GET'])
def job_status(job_id):
    """Poll a pipeline job. When status == 'ready', re-fetch the snapshot
    endpoint (it'll be a fresh cache hit)."""
    job = get_job(job_id)
    if job is None:
        return jsonify({'error': f'No job with id {job_id}'}), 404
    return jsonify({
        'id': job['id'],
        'ticker': job['ticker'],
        'horizon': job['horizon'],
        'status': job['status'],
        'progress': job.get('progress'),
        'snapshot_id': job.get('snapshot_id'),
        'error': job.get('error'),
        'requested_at': job['requested_at'].isoformat() if job.get('requested_at') else None,
        'started_at': job['started_at'].isoformat() if job.get('started_at') else None,
        'finished_at': job['finished_at'].isoformat() if job.get('finished_at') else None,
    })


if __name__ == '__main__':
    print("🚀 Starting Finance Risk Analysis API...")
    print("📊 Multi-Agent System Ready")
    provider, _openai_key = _detect_model_provider()
    if provider == 'openai':
        print(f"🤖 Using OpenAI model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    else:
        print(f"🤖 Using Ollama model: {os.getenv('OLLAMA_MODEL', 'qwen3:4b')}")
    app.run(
        debug=True,
        host=os.getenv('HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', '8000')),
    )
