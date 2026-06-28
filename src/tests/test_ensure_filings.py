"""EDGAR ingest, committed: ensure_filings is a no-op (no network) when filings
already exist, ingests on a miss, and swallows EDGAR errors."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.edgar_ingest as edgar  # noqa: E402


def _rag_with(filings):
    rag = MagicMock()
    rag.get_available_filings.return_value = filings
    return rag


def test_noop_when_filings_present():
    """Hot path: filings already stored → return 0, NO EDGAR/network call."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag_with([{"filing_type": "10-K"}])), \
         patch.object(edgar, "ingest_tickers") as ingest:
        assert edgar.ensure_filings("AAPL") == 0
        ingest.assert_not_called()


def test_ingests_on_miss():
    with patch.object(edgar, "FundamentalRAG", return_value=_rag_with([])), \
         patch.object(edgar, "ingest_tickers", return_value=2) as ingest:
        assert edgar.ensure_filings("AAPL") == 2
        ingest.assert_called_once()


def test_swallows_edgar_errors():
    """A flaky SEC API must never raise out of ensure_filings."""
    with patch.object(edgar, "FundamentalRAG", return_value=_rag_with([])), \
         patch.object(edgar, "ingest_tickers", side_effect=RuntimeError("EDGAR down")):
        assert edgar.ensure_filings("AAPL") == 0  # degraded, no exception
