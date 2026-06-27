"""
Day 3 (Sprint 1) — RAG / ingestion correctness, unit level (no live DB).

Covers:
  - number-grounding logic (the hallucination signal wired into the agent),
  - the multi-row chunk-INSERT batching (pipeline-free, correct sizes/params),
  - the retrieval date-range overlap filter SQL.
"""

import os
import sys
from types import SimpleNamespace

# Put src/ on the path so `utils.*` imports resolve when run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.grounding import (  # noqa: E402
    extract_numbers, grounding_report, GroundingResult,
)
from utils.rag_utils import _build_chunk_insert_batches, FundamentalRAG  # noqa: E402


def _fund(**kw):
    """Minimal stand-in for a FundamentalAnalysis (only the fields grounding reads)."""
    base = dict(
        executive_summary="", competitive_position="", growth_prospects="",
        investment_thesis="", key_financial_metrics=[], business_highlights=[],
        risk_factors=[], concerns_and_risks=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------- grounding
def test_extract_numbers_strips_commas():
    assert extract_numbers("revenue 1,234.5 and 42") == {"1234.5", "42"}


def test_grounding_report_splits_grounded_and_ungrounded():
    f = _fund(
        executive_summary="Revenue 391,035 and margin 46.2",
        key_financial_metrics=["EPS 6.08", "made-up 99999"],
    )
    source = "total net sales 391035 ... gross margin 46.2 ... diluted eps 6.08"
    r = grounding_report(f, source)
    assert isinstance(r, GroundingResult)
    assert r.total == 4 and r.grounded == 3
    assert r.ungrounded == ["99999"]
    assert abs(r.ratio - 0.75) < 1e-9


def test_grounding_ignores_trivial_small_ints():
    # 7, 10, 3 are all in the 0–10 trivial set → excluded → nothing to ground.
    r = grounding_report(_fund(executive_summary="score 7 of 10, 3 segments"), "")
    assert r == GroundingResult(0, 0, []) and r.ratio is None


def test_grounding_no_substantive_numbers():
    assert grounding_report(_fund(executive_summary="no digits"), "src").total == 0


# --------------------------------------------- multi-row INSERT batching
def _rows(n):
    return [(1, "AAPL", i, "content", [0.0], "mock:16d", {}) for i in range(n)]


def test_insert_batches_sizes_and_count():
    batches = list(_build_chunk_insert_batches(_rows(1200), batch_size=500))
    assert len(batches) == 3
    # number of "(%s, ...)" groups per batch == row count in that batch
    sizes = [values_sql.count("(%s") for values_sql, _ in batches]
    assert sizes == [500, 500, 200]


def test_insert_batch_param_flattening_preserves_order():
    [(values_sql, flat)] = list(_build_chunk_insert_batches(_rows(3), batch_size=500))
    assert values_sql == ", ".join(["(%s, %s, %s, %s, %s, %s, %s)"] * 3)
    assert len(flat) == 7 * 3
    # row 0: filing_id=1, chunk_index=0 ; row 1: chunk_index=1
    assert flat[0] == 1 and flat[2] == 0 and flat[7 + 2] == 1
    # embedding_model tag carried in each row (index 5 within the 7-tuple)
    assert flat[5] == "mock:16d"


def test_insert_batches_empty():
    assert list(_build_chunk_insert_batches([], 500)) == []


# ------------------------------------------- retrieval date-range filter
def test_search_sql_has_period_overlap_filter():
    sql = FundamentalRAG._build_search_sql(None)
    assert "f.period_end >= %(from_date)s" in sql
    assert "f.period_start <= %(to_date)s" in sql
    assert "c.embedding_model = %(model)s" in sql
    assert "filing_type" not in sql  # omitted when not requested


def test_search_sql_filing_type_optional():
    sql = FundamentalRAG._build_search_sql("10-K")
    assert "f.filing_type = %(filing_type)s" in sql
