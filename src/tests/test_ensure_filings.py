"""EDGAR ingest, committed: ensure_filings is a no-op (no network) when the
ticker already has chunks under the ACTIVE embedding namespace, re-ingests on a
miss (including a provider-switched ticker that has filings rows but no chunks
for this namespace), skips when the embedder itself is broken, and swallows
EDGAR errors."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.edgar_ingest as edgar  # noqa: E402


def _rag(has_chunks: bool, embedding_dim: int = 1024):
    """Stub FundamentalRAG for the two things ensure_filings inspects: whether
    the ticker has chunks under the active namespace, and whether the embedder
    probe succeeded (embedding_dim > 0)."""
    rag = MagicMock()
    rag.has_chunks.return_value = has_chunks
    rag.embedding_dim = embedding_dim
    rag.embedding_model = f"test-{embedding_dim}d"
    return rag


def test_noop_when_chunks_present():
    """Hot path: chunks already stored under this namespace → 0, NO EDGAR call."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag(True)), \
         patch.object(edgar, "ingest_tickers") as ingest:
        assert edgar.ensure_filings("AAPL") == 0
        ingest.assert_not_called()


def test_ingests_on_miss():
    """No chunks under the active namespace → ingest (covers both a brand-new
    ticker and one ingested under a different embedding provider)."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag(False)), \
         patch.object(edgar, "ingest_tickers", return_value=2) as ingest:
        assert edgar.ensure_filings("AAPL") == 2
        ingest.assert_called_once()


def test_skips_ingest_when_embedder_broken():
    """Dimension probe failed (e.g. Bedrock IAM denial) → ingest would only die
    at the embed step, so don't re-download from EDGAR on every run."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag(False, embedding_dim=0)), \
         patch.object(edgar, "ingest_tickers") as ingest:
        assert edgar.ensure_filings("AAPL") == 0
        ingest.assert_not_called()


def test_swallows_edgar_errors():
    """A flaky SEC API must never raise out of ensure_filings."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag(False)), \
         patch.object(edgar, "ingest_tickers", side_effect=RuntimeError("EDGAR down")) as ingest:
        assert edgar.ensure_filings("AAPL") == 0  # degraded, no exception
        ingest.assert_called_once()  # it actually reached EDGAR, then swallowed


def test_db_check_failure_degrades_to_noop():
    """If the stored-chunks check itself blows up (DB hiccup), don't fail the run
    and don't attempt a network ingest either."""
    with patch.object(edgar, "FundamentalRAG", side_effect=RuntimeError("db down")), \
         patch.object(edgar, "ingest_tickers") as ingest:
        assert edgar.ensure_filings("AAPL") == 0
        ingest.assert_not_called()
