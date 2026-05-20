"""
Eval suite v0 — quality measurements for the multi-agent workflow.

Unlike the pass/fail unit tests in src/tests/, these are slow, non-deterministic
*measurements*, so this is a standalone script that runs the full pipeline a few
times for one ticker and prints three metrics:

  (a) Structured-output success rate — did the sentiment / fundamental agents
      return populated schemas (vs. the empty-fallback)?
  (b) Number grounding — do the numbers in the fundamental analysis actually
      appear in the retrieved filing chunks? (approximate hallucination signal)
  (c) Recommendation stability — does the BUY/HOLD/SELL recommendation agree
      across repeated runs of the same ticker?

Usage:
    python -m src.evals.eval_suite --ticker AAPL --runs 3

Each run executes the chain graph + the debate graph, so this is minutes-per-run
on a local model. It needs Postgres (RAG) and an LLM provider configured.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.runnables import RunnableConfig  # noqa: E402
from agents import State  # noqa: E402
from main import build_chain_graph, build_final_recommendation_graph  # noqa: E402
from utils.config import get_llm  # noqa: E402
from utils.rag_utils import FundamentalRAG  # noqa: E402
from utils.schemas import DebateReport  # noqa: E402

# The queries the fundamental agent feeds query_10k_documents — reused here so
# eval (b) compares against a representative slice of the ticker's filings.
STANDARD_QUERIES = [
    "financial metrics", "business segments", "risk factors",
    "competitive position", "growth prospects", "investment thesis",
    "concerns and risks",
]
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


# --------------------------------------------------------------------- run it
def run_pipeline(ticker: str, period: str, interval: str, horizon_days: int) -> dict:
    """Run the chain graph then the debate graph once; return collected outputs."""
    chain = build_chain_graph()
    state = State(ticker=ticker, period=period, interval=interval,
                  horizon_days=horizon_days)
    res = State(**chain.invoke(state, config=RunnableConfig()))

    debate_graph = build_final_recommendation_graph()
    res.debate = DebateReport(agent_list=["fundamental", "sentiment", "valuation"])
    res.debate.agent_max_turn = 5
    res = State(**debate_graph.invoke(res, config=RunnableConfig(recursion_limit=100)))

    return {
        "sentiment": res.sentiment,
        "fundamental": res.fundamental,
        "consensus": res.debate.consensus_summary if res.debate else "",
    }


# ----------------------------------------------------------------- metric (a)
def _sentiment_populated(s) -> bool:
    return bool(s and s.overall_sentiment and s.key_insights)


def _fundamental_populated(f) -> bool:
    return bool(f and f.executive_summary and f.business_highlights
                and f.filing_type != "N/A")


# ----------------------------------------------------------------- metric (b)
def _numbers(text: str) -> set[str]:
    """Numeric tokens, comma-stripped (e.g. '1,234.5' -> '1234.5')."""
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def number_grounding(fundamental, ticker: str) -> tuple[int, int]:
    """Return (grounded, total): how many numbers in the fundamental analysis
    also appear in the ticker's retrieved filing chunks."""
    if fundamental is None:
        return 0, 0
    fields = [
        fundamental.executive_summary, fundamental.competitive_position,
        fundamental.growth_prospects, fundamental.investment_thesis,
        *fundamental.key_financial_metrics, *fundamental.business_highlights,
        *fundamental.risk_factors, *fundamental.concerns_and_risks,
    ]
    output_numbers = _numbers(" ".join(str(x) for x in fields))
    if not output_numbers:
        return 0, 0

    rag = FundamentalRAG()
    chunk_lists = rag.retrieve_relevant_chunks_batch(
        ticker, STANDARD_QUERIES, from_date=date(2000, 1, 1), to_date=date.today())
    source = " ".join(
        doc.page_content for chunks in chunk_lists for doc in chunks
    ).replace(",", "")

    grounded = sum(1 for n in output_numbers if n in source)
    return grounded, len(output_numbers)


# ----------------------------------------------------------------- metric (c)
def extract_recommendation(llm, consensus: str) -> str:
    """Reduce a debate consensus summary to BUY / SELL / HOLD."""
    if not consensus:
        return "N/A"
    resp = llm.invoke(
        f"Output as a single word - BUY, SELL, or HOLD - from the following "
        f"text: {consensus}")
    text = (resp.content if hasattr(resp, "content") else str(resp)).upper()
    for word in ("BUY", "SELL", "HOLD"):
        if word in text:
            return word
    return "N/A"


# ----------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Eval suite v0 for the multi-agent workflow.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--runs", type=int, default=3, help="pipeline repetitions (default 3)")
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print(f"EVAL SUITE v0 — {args.ticker}, {args.runs} run(s)")
    print("=" * 60)

    runs: list[dict] = []
    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs} ...", flush=True)
        try:
            runs.append(run_pipeline(args.ticker, args.period, args.interval,
                                     args.horizon_days))
        except Exception as e:
            print(f"  run failed: {e}")
            runs.append({"sentiment": None, "fundamental": None, "consensus": ""})

    llm = get_llm()
    recos = [extract_recommendation(llm, r["consensus"]) for r in runs]

    # (a) structured-output success rate
    sent_ok = sum(_sentiment_populated(r["sentiment"]) for r in runs)
    fund_ok = sum(_fundamental_populated(r["fundamental"]) for r in runs)

    # (b) number grounding — first run with a populated fundamental analysis
    grounded = total = 0
    for r in runs:
        if _fundamental_populated(r["fundamental"]):
            grounded, total = number_grounding(r["fundamental"], args.ticker)
            break

    # (c) recommendation stability
    decided = [r for r in recos if r != "N/A"]
    stable = len(set(decided)) <= 1 and len(decided) == len(recos) and bool(decided)

    n = len(runs)
    print()
    print("RESULTS")
    print("-" * 60)
    a_flag = "PASS" if sent_ok == n and fund_ok == n else "WARN"
    print(f"(a) Structured-output success           [{a_flag}]")
    print(f"      sentiment:   {sent_ok}/{n} populated")
    print(f"      fundamental: {fund_ok}/{n} populated")

    pct = f"{grounded / total:.0%}" if total else "n/a"
    print("(b) Number grounding (fundamental vs retrieved chunks)  [INFO]")
    print(f"      {grounded}/{total} numbers found in source ({pct})")
    print("      approximate — years / small integers match trivially")

    c_flag = "PASS" if stable else "WARN"
    dist = ", ".join(f"{k}×{v}" for k, v in Counter(recos).items())
    print(f"(c) Recommendation stability            [{c_flag}]")
    print(f"      {dist}  ->  {'STABLE' if stable else 'UNSTABLE'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
