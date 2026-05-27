from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
import threading
import queue
import requests
import io
import base64
import openai

# Ensure we can import project code from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import main  # src/main.py — run_pipeline_for_horizon
from utils.horizons import get_horizon, HORIZONS
from utils.snapshots import (
    get_latest_snapshot, is_fresh, list_snapshot_history, list_tracked_tickers,
)

from markdown_pdf import MarkdownPdf, Section

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


def _provider_unavailable_response():
    """Return an error response if the configured LLM provider isn't reachable,
    else None. Only checked on the compute path (cache hits skip it)."""
    provider, openai_key = _detect_model_provider()
    if provider == 'openai' and not openai_key:
        return jsonify({'error': "OpenAI not configured. Set OPENAI_API_KEY or MODEL_PROVIDER=ollama."}), 503
    if provider == 'ollama' and not _is_ollama_available():
        return jsonify({'error': 'Ollama is not reachable. Start it with: ollama serve'}), 503
    if provider not in ('openai', 'ollama'):
        return jsonify({'error': f"Unknown MODEL_PROVIDER: {provider}."}), 400
    return None


def _run_pipeline_with_timeout(ticker: str, horizon_name: str):
    """Run the full chain + debate pipeline for (ticker, horizon) under the
    watchdog timeout; it persists a snapshot as a side effect.
    Returns (ok: bool, error: Exception | None)."""
    result_queue: "queue.Queue" = queue.Queue(maxsize=1)

    def _run():
        try:
            main.run_pipeline_for_horizon(ticker, horizon_name)  # persists snapshot
            result_queue.put((True, None))
        except Exception as e:  # noqa: BLE001
            result_queue.put((False, e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    timeout_seconds = int(os.getenv('ANALYZE_TIMEOUT_SECS', '900'))
    thread.join(timeout_seconds)
    if thread.is_alive():
        return False, TimeoutError(f"Analysis timed out after {timeout_seconds}s")
    return result_queue.get()


def _resolve_snapshot(ticker: str, horizon):
    """Cache-or-compute: return a fresh cached snapshot, otherwise run the
    pipeline synchronously, persist, and return the new snapshot."""
    cached = get_latest_snapshot(ticker, horizon.name)
    if cached and is_fresh(cached, horizon):
        return jsonify(_attach_pdf(_serialize_snapshot(cached, cached=True)))

    unavailable = _provider_unavailable_response()
    if unavailable is not None:
        return unavailable

    print(f"🧠 Computing snapshot for {ticker} / {horizon.name} (cache miss/stale)")
    ok, err = _run_pipeline_with_timeout(ticker, horizon.name)
    if not ok:
        if isinstance(err, TimeoutError):
            return jsonify({'error': str(err)}), 504
        raise err

    fresh = get_latest_snapshot(ticker, horizon.name)
    if fresh is None:
        return jsonify({'error': 'Pipeline finished but no snapshot was stored.'}), 500
    return jsonify(_attach_pdf(_serialize_snapshot(fresh, cached=False)))


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
    rows = list_snapshot_history(ticker.strip().upper(), h.name, days=days)
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


@app.route('/api/tickers', methods=['GET'])
def list_tickers():
    """Tickers that have at least one snapshot, for the landing page."""
    tickers = [
        {
            'ticker': t['ticker'],
            'last_updated': t['last_updated'].isoformat() if t.get('last_updated') else None,
            'snapshot_count': t['snapshot_count'],
        }
        for t in list_tracked_tickers()
    ]
    return jsonify({'horizons': list(HORIZONS), 'tickers': tickers})


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
