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
    get_latest_job, get_latest_prices, _sanitize_for_json,
)
from utils.prices import fetch_price_series_polygon, slice_period, PERIOD_TAIL

from markdown_pdf import MarkdownPdf, Section

from threading import Lock
from cachetools import TTLCache, cached

app = Flask(__name__)
CORS(app)


# --- Robustness: every response is valid JSON, every error is JSON ----------

from flask.json.provider import DefaultJSONProvider  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402


class _SafeJSONProvider(DefaultJSONProvider):
    """Sanitize NaN / Infinity to null on EVERY response.

    Python's json emits those as bare `NaN`/`Infinity` tokens (invalid per
    RFC 8259), which makes the frontend's JSON.parse throw. Snapshots are already
    sanitized at write time, but a computed value (cumulative_return, a price,
    a metric) can still go non-finite on the serve path — this guarantees no
    endpoint can ever emit invalid JSON, regardless of what produced the value.
    """
    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize_for_json(obj), **kwargs)


app.json = _SafeJSONProvider(app)


@app.errorhandler(HTTPException)
def _handle_http_exception(e):
    """Return a JSON body for 404/405/400/etc. instead of Werkzeug's HTML page,
    so the SPA always gets parseable JSON."""
    return jsonify({'error': e.description, 'status': e.code}), e.code


@app.errorhandler(Exception)
def _handle_unexpected(e):
    """Last-resort net: any uncaught exception becomes a JSON 500 (no stack
    trace leaked to the client; the full traceback is logged server-side)."""
    app.logger.error("Unhandled error on %s: %s\n%s",
                     request.path, e, traceback.format_exc())
    return jsonify({'error': 'Internal server error'}), 500


# Tickers: 1–10 chars, start with a letter, then letters/digits/./- (covers
# BRK.B, BRK-B, RDS.A, etc.). Anything else (spaces, quotes, ';', unicode,
# over-long strings) is rejected before it can reach the DB or pipeline.
_TICKER_RE = re.compile(r'^[A-Z][A-Z0-9.\-]{0,9}$')


def _clean_ticker(raw):
    """Return (ticker, None) on success, else (None, (json_response, 400)).

    Uppercases + strips, then enforces the symbol shape so garbage never gets
    enqueued as a doomed job and injection-y strings are rejected outright.
    """
    t = str(raw or '').strip().upper()
    if not t:
        return None, (jsonify({'error': 'Missing required field: ticker'}), 400)
    if not _TICKER_RE.match(t):
        return None, (jsonify({
            'error': f'Invalid ticker {t!r}: expected 1–10 characters '
                     '(letters, digits, . or -) starting with a letter.'
        }), 400)
    return t, None


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
# Separate cache so the snapshot-read and the live-Polygon-fallback paths (both
# keyed on `ticker`) don't collide in one cachetools cache.
_price_fallback_cache: TTLCache = TTLCache(maxsize=256, ttl=_PRICE_TTL_S)
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


# Price history is served from the worker-persisted snapshot (`snapshots.prices`,
# a full ~2y Polygon daily-close series captured at pipeline run time), so the
# request path normally never calls Polygon. Polygon is a live fallback only,
# for tickers whose latest snapshot predates this feature or was generated
# without POLYGON_API_KEY on the worker. The Polygon fetch + slicing logic lives
# in utils.prices so the backend and the pipeline share one implementation.
#
# yfinance is deliberately NOT used: it pulls in curl_cffi (a native extension
# that crashes the backend container on python:slim — the ALB then serves 502)
# and Yahoo IP-blocks AWS datacenter ranges regardless.
_PERIOD_TAIL = PERIOD_TAIL  # {"1mo": 22, "6mo": 124, "2y": None}; None = full series


@cached(cache=_price_cache, lock=_cache_lock)
def _cached_snapshot_prices(ticker: str):
    """Full ~2y daily close series for `ticker`, read from the latest snapshot
    that carries one. Returns None when no snapshot has prices, so the caller
    can fall back to a live Polygon fetch. Cached 15 min per ticker."""
    return get_latest_prices(ticker)


@cached(cache=_price_fallback_cache, lock=_cache_lock)
def _cached_price_full(ticker: str):
    """Live Polygon fallback: full ~2y daily close series, list of
    {date: 'YYYY-MM-DD', close: float}; empty list on any failure or missing
    key. Only used when the snapshot has no persisted prices."""
    return fetch_price_series_polygon(ticker, years=2.0)


def _cached_price_series(ticker: str, period: str):
    """Daily close prices over `period` (1mo / 6mo / 2y). Reads the full series
    from the latest snapshot (worker-persisted Polygon series) and slices it per
    period; falls back to a live Polygon fetch only when no snapshot carries
    prices. The three horizons are always distinct slices of one ~2y series."""
    full = _cached_snapshot_prices(ticker)
    if not full:
        full = _cached_price_full(ticker)
    return slice_period(full or [], period)


def _resolve_snapshot(ticker: str, horizon, *, force: bool = False):
    """Cache-or-enqueue. Fresh cache → 200 + snapshot. Otherwise enqueue a job
    (deduped per ticker+horizon) for the worker and return 202 + job info; the
    client polls this same endpoint until the snapshot is fresh. `force=True`
    (client retry) re-enqueues even past a recent failure."""
    cached = _cached_latest_snapshot(ticker, horizon.name)
    if cached and is_fresh(cached, horizon):
        return jsonify(_attach_pdf(_serialize_snapshot(cached, cached=True)))

    # The TTL cache above can lag the worker by up to _SNAPSHOT_TTL_S. The moment
    # a job finishes it stops blocking create_job's (queued|running) dedup, so a
    # poll that lands in that lag window would see "no fresh snapshot, no in-flight
    # job" and enqueue a DUPLICATE job (→ a second, redundant snapshot). Confirm
    # against a fresh read before enqueuing — this only costs a DB roundtrip on
    # the miss path, which is exactly where we were about to enqueue anyway.
    fresh = get_latest_snapshot(ticker, horizon.name)
    if fresh and is_fresh(fresh, horizon):
        return jsonify(_attach_pdf(_serialize_snapshot(fresh, cached=True)))

    # Surface a recent failure instead of re-enqueuing it forever. A failed job
    # is terminal (it no longer blocks create_job's dedup), so without this the
    # next poll would silently start a fresh job and the client would spin
    # indefinitely, never seeing the failure. The client retries explicitly
    # (?retry=1 → force=True), which bypasses this and enqueues a new job.
    if not force:
        latest = get_latest_job(ticker, horizon.name)
        if latest and latest['status'] == 'failed':
            return _job_response(latest)

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
        data = request.get_json(silent=True) or {}
        ticker, err = _clean_ticker(data.get('ticker'))
        if err:
            return err
        horizon, err = _parse_horizon(str(data.get('horizon', 'MID')))
        if err:
            return err
        force = str(data.get('retry') or request.args.get('retry') or '').lower() \
            in ('1', 'true', 'yes')
        return _resolve_snapshot(ticker, horizon, force=force)
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
        tkr, err = _clean_ticker(ticker)
        if err:
            return err
        h, err = _parse_horizon(horizon)
        if err:
            return err
        force = (request.args.get('retry') or '').strip().lower() in ('1', 'true', 'yes')
        return _resolve_snapshot(tkr, h, force=force)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'Snapshot failed: {str(e)}'}), 500


@app.route('/api/snapshot/<ticker>/<horizon>/history', methods=['GET'])
def snapshot_history(ticker, horizon):
    """Timeseries of past snapshots for the run/risk-history view."""
    tkr, err = _clean_ticker(ticker)
    if err:
        return err
    h, err = _parse_horizon(horizon)
    if err:
        return err
    try:
        days = max(1, min(int(request.args.get('days', 90)), 365))
    except (ValueError, TypeError):
        days = 90
    rows = _cached_snapshot_history(tkr, h.name, days)
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
    return jsonify({'ticker': tkr, 'horizon': h.name, 'history': history})


_VALID_PRICE_PERIODS = {"1mo", "6mo", "2y"}


@app.route('/api/price/<ticker>', methods=['GET'])
def price_series(ticker):
    """Daily close prices for one of the three horizon lookback windows.
    Used by the price chart on the TickerView. Server-cached for 15 min."""
    tkr, err = _clean_ticker(ticker)
    if err:
        return err
    period = (request.args.get('period') or '1mo').strip().lower()
    if period not in _VALID_PRICE_PERIODS:
        return jsonify({'error': f"period must be one of {sorted(_VALID_PRICE_PERIODS)}"}), 400
    series = _cached_price_series(tkr, period)
    return jsonify({
        'ticker': tkr,
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
