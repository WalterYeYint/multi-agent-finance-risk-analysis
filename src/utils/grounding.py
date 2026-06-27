"""
Number grounding — an approximate hallucination signal for the fundamental
analysis: does a number the model quoted (revenue, a margin, a count) actually
appear in the retrieved filing text?

Pure number logic lives here so it has exactly one implementation, shared by:
  - the fundamental agent (guard/log: flag ungrounded numbers per run), and
  - the eval suite's metric (b).

`grounding_report` is dependency-free (no DB) and is the unit-tested core;
`ground_against_filings` adds the DB-backed source fetch and is failure-tolerant
so it can never break a pipeline run.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

# A numeric token: a digit, then digits/commas, optional decimal part.
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")

# Small integers (0–10) are everywhere — years' digits, list counts, the 0–10
# health score — and almost always "ground" trivially, inflating the rate
# without signal. Excluded so the metric reflects substantive figures.
_TRIVIAL = {str(n) for n in range(0, 11)}


class GroundingResult(NamedTuple):
    grounded: int
    total: int
    ungrounded: list  # the substantive numbers NOT found in the source text

    @property
    def ratio(self) -> Optional[float]:
        return self.grounded / self.total if self.total else None


def extract_numbers(text: str) -> set:
    """Numeric tokens, comma-stripped ('1,234.5' -> '1234.5')."""
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def fundamental_numbers(fundamental) -> set:
    """Every numeric token across the model-authored fields of a
    FundamentalAnalysis (the fields the LLM fills — not system metadata)."""
    if fundamental is None:
        return set()
    fields = [
        fundamental.executive_summary, fundamental.competitive_position,
        fundamental.growth_prospects, fundamental.investment_thesis,
        *(fundamental.key_financial_metrics or []),
        *(fundamental.business_highlights or []),
        *(fundamental.risk_factors or []),
        *(fundamental.concerns_and_risks or []),
    ]
    return extract_numbers(" ".join(str(x) for x in fields))


def grounding_report(fundamental, source_text: str) -> GroundingResult:
    """Compare the substantive numbers in `fundamental` against `source_text`
    (the filing chunks). Pure — no DB, no network. Trivial 0–10 ints ignored."""
    nums = {n for n in fundamental_numbers(fundamental) if n not in _TRIVIAL}
    if not nums:
        return GroundingResult(0, 0, [])
    source = (source_text or "").replace(",", "")
    grounded = sorted(n for n in nums if n in source)
    ungrounded = sorted(n for n in nums if n not in source)
    return GroundingResult(len(grounded), len(nums), ungrounded)


def ground_against_filings(fundamental, ticker: str, *, rag=None) -> GroundingResult:
    """Ground `fundamental`'s numbers against all stored filing text for `ticker`.

    Failure-tolerant by contract: any error (no DB, no filings, etc.) returns an
    empty result rather than raising, so this can be called as a non-fatal guard
    inside the pipeline.
    """
    try:
        if rag is None:
            from utils.rag_utils import FundamentalRAG
            rag = FundamentalRAG()
        return grounding_report(fundamental, rag.all_chunk_text(ticker))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  grounding check skipped for {ticker}: {e}")
        return GroundingResult(0, 0, [])
