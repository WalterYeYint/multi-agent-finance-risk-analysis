"""
SEC EDGAR auto-ingest.

Downloads recent 10-K / 10-Q filings straight from the free SEC EDGAR API and
ingests them into the Postgres + pgvector RAG store — replacing the manual
"drop a PDF into data/filings/" workflow. Idempotent (skips filings already
stored by accession number) and safe to run on a cron / EventBridge schedule.

Usage:
    python -m src.utils.edgar_ingest --tickers AAPL,MSFT,GOOGL --forms 10-Q --limit 2
    python -m src.utils.edgar_ingest --tickers TSLA --forms 10-Q --limit 4
    python -m src.utils.edgar_ingest --tickers AAPL --force

Environment:
    SEC_USER_AGENT   Contact string SEC requires, e.g. "Name name@example.com".
                     SEC throttles / blocks requests without a real contact.
    FILINGS_RAW_DIR  Where extracted filing text is archived (default ./data/filings_raw).
    DATABASE_URL     Postgres connection (see utils/db.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, date
from typing import Dict, List, Optional

import requests
from dateutil.relativedelta import relativedelta

# Allow `python src/utils/edgar_ingest.py` and `python -m src.utils.edgar_ingest`
# to both resolve the `utils.*` package imports the rest of the codebase uses.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from utils.rag_utils import FundamentalRAG  # noqa: E402

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:  # pragma: no cover
    _BS4 = False

# SEC fair-access policy: <= 10 requests/second, descriptive User-Agent required.
SEC_RATE_LIMIT_SECONDS = 0.25
_DEFAULT_UA = "multi-agent-finance-risk-analysis research (set SEC_USER_AGENT)"

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def _headers() -> Dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip() or _DEFAULT_UA
    if ua == _DEFAULT_UA:
        print("⚠️  SEC_USER_AGENT not set. SEC asks for a real contact "
              "(e.g. 'Your Name you@example.com'); requests may be throttled.")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _get(url: str, *, as_json: bool = False, timeout: int = 30):
    """GET with the SEC headers and a polite delay; returns json or text."""
    time.sleep(SEC_RATE_LIMIT_SECONDS)
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.json() if as_json else resp.text


# --------------------------------------------------------------------- lookup
def load_cik_map() -> Dict[str, int]:
    """Fetch SEC's ticker→CIK map (uppercased ticker -> integer CIK)."""
    data = _get(COMPANY_TICKERS_URL, as_json=True)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def select_filings(submissions: dict, forms: List[str], limit: int) -> List[dict]:
    """Pick the most recent `limit` filings per requested form from a submissions doc."""
    recent = submissions.get("filings", {}).get("recent", {})
    accession = recent.get("accessionNumber", [])
    form = recent.get("form", [])
    filing_date = recent.get("filingDate", [])
    report_date = recent.get("reportDate", [])
    primary_doc = recent.get("primaryDocument", [])

    wanted = {f.upper() for f in forms}
    per_form: Dict[str, int] = {f: 0 for f in wanted}
    picked: List[dict] = []

    # Arrays are newest-first.
    for i in range(len(accession)):
        f = (form[i] or "").upper()
        if f not in wanted or per_form[f] >= limit:
            continue
        if not report_date[i] or not primary_doc[i]:
            continue
        per_form[f] += 1
        picked.append({
            "accession": accession[i],
            "form": f,
            "filing_date": filing_date[i],
            "report_date": report_date[i],
            "primary_doc": primary_doc[i],
        })
        if all(c >= limit for c in per_form.values()):
            break
    return picked


def compute_period(form: str, report_date: str
                   ) -> tuple[int, int, int, date, date]:
    """
    Map a filing's period-of-report end date to
    (year, start_month, end_month, period_start, period_end).

    `period_end` is the exact report date; `period_start` is one year back for a
    10-K / three months back for a 10-Q — these drive RAG date-range filtering
    and, unlike the year+month ints, correctly span a calendar-year boundary.
    The year/start_month/end_month ints are retained as the natural dedup key.
    """
    period_end = datetime.strptime(report_date, "%Y-%m-%d").date()
    if "10-K" in form.upper():
        period_start = period_end - relativedelta(years=1)
        return period_end.year, 1, 12, period_start, period_end
    period_start = period_end - relativedelta(months=3)
    return (period_end.year, max(1, period_end.month - 2), period_end.month,
            period_start, period_end)


# -------------------------------------------------------------------- extract
def extract_text(raw: str, max_chars: int) -> str:
    """Strip a filing HTML document down to plain text, capped at max_chars."""
    if _BS4:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    else:  # pragma: no cover - bs4 is in requirements
        import re
        text = re.sub(r"<[^>]+>", " ", raw)

    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def save_raw(raw_dir: str, ticker: str, form: str, accession: str, text: str) -> str:
    """Archive extracted filing text locally; return the path."""
    os.makedirs(raw_dir, exist_ok=True)
    safe_form = form.replace("/", "")
    path = os.path.join(raw_dir, f"{ticker}-{safe_form}-{accession}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------- ingest
def ingest_ticker(rag: FundamentalRAG, ticker: str, cik: int, forms: List[str],
                  limit: int, raw_dir: str, max_chars: int, force: bool) -> int:
    """Ingest up to `limit` filings per form for one ticker. Returns count ingested."""
    ticker = ticker.upper()
    print(f"\n{'=' * 60}\n📥 {ticker} (CIK {cik})")

    submissions = _get(SUBMISSIONS_URL.format(cik=cik), as_json=True)
    company = submissions.get("name", ticker)
    filings = select_filings(submissions, forms, limit)
    if not filings:
        print(f"   No {', '.join(forms)} filings found.")
        return 0

    ingested = 0
    for f in filings:
        accession = f["accession"]
        if not force and rag.has_filing(accession):
            print(f"   ⏭️  {f['form']} {accession} already ingested — skipping.")
            continue

        acc_nodash = accession.replace("-", "")
        doc_url = ARCHIVE_URL.format(cik=cik, acc=acc_nodash, doc=f["primary_doc"])
        print(f"   ⬇️  {f['form']} report-date {f['report_date']} — {doc_url}")

        try:
            raw = _get(doc_url)
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            continue

        text = extract_text(raw, max_chars)
        if len(text) < 500:
            print(f"   ⚠️  Extracted only {len(text)} chars — skipping.")
            continue

        year, start_month, end_month, period_start, period_end = compute_period(
            f["form"], f["report_date"])
        save_raw(raw_dir, ticker, f["form"], accession, text)

        if rag.ingest_text(
            text,
            ticker=ticker,
            filing_type=f["form"],
            filing_year=year,
            filing_start_month=start_month,
            filing_end_month=end_month,
            period_start=period_start,
            period_end=period_end,
            source=doc_url,
            accession_no=accession,
            company=company,
            force=force,
        ):
            ingested += 1

    return ingested


def ingest_tickers(tickers: List[str], forms: List[str], limit: int,
                   raw_dir: str, max_chars: int, force: bool) -> int:
    """Resolve CIKs and ingest filings for each ticker. Returns total ingested."""
    rag = FundamentalRAG()
    print("🔎 Fetching SEC ticker→CIK map...")
    cik_map = load_cik_map()

    total = 0
    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        if cik is None:
            print(f"\n❌ {ticker}: not found in SEC ticker map — skipping.")
            continue
        try:
            total += ingest_ticker(rag, ticker, cik, forms, limit,
                                   raw_dir, max_chars, force)
        except Exception as e:
            print(f"\n❌ {ticker}: ingest failed — {e}")
    return total


# ----------------------------------------------------- on-demand / sweep hooks
def _env_forms() -> List[str]:
    raw = os.getenv("EDGAR_FORMS", "10-K,10-Q")
    forms = [f.strip().upper() for f in raw.split(",") if f.strip()]
    return forms or ["10-K", "10-Q"]


def _env_limit() -> int:
    try:
        return max(1, int(os.getenv("EDGAR_LIMIT", "4")))
    except (TypeError, ValueError):
        return 4


def ensure_filings(ticker: str, *, progress_cb=None) -> int:
    """Pipeline pre-flight: make sure `ticker` has SEC filings in the RAG store.

    Fast path (the common case): if the ticker already has stored filings we
    return immediately and make NO network call. Only on a true cache miss do we
    hit EDGAR. Any EDGAR / network failure is logged and swallowed — the pipeline
    must still proceed (falling back to local PDFs or running degraded); a flaky
    SEC API must NEVER fail the whole job. Returns the number of filings ingested
    (0 if already present or on error). Forms / limit come from EDGAR_FORMS /
    EDGAR_LIMIT (defaults 10-K,10-Q / 4)."""
    ticker = ticker.upper()
    try:
        rag = FundamentalRAG()
        if rag.get_available_filings(ticker):
            return 0  # already have filings — no network call on the hot path
    except Exception as e:  # noqa: BLE001 - DB hiccup shouldn't fail the run
        print(f"⚠️  ensure_filings: could not check stored filings for {ticker}: {e}")
        return 0

    if progress_cb is not None:
        try:
            progress_cb("ingesting filings")
        except Exception:  # noqa: BLE001
            pass

    forms = _env_forms()
    limit = _env_limit()
    raw_dir = os.getenv("FILINGS_RAW_DIR", "./data/filings_raw")
    try:
        n = ingest_tickers([ticker], forms=forms, limit=limit,
                           raw_dir=raw_dir, max_chars=400_000, force=False)
        print(f"📥 ensure_filings({ticker}): ingested {n} filing(s).")
        return n
    except Exception as e:  # noqa: BLE001 - EDGAR down must not crash the pipeline
        print(f"⚠️  ensure_filings({ticker}): EDGAR ingest failed (continuing): {e}")
        return 0


def refresh_tracked_filings(tickers: List[str]) -> int:
    """Weekly sweep: re-check each tracked ticker for NEW filings and ingest them.

    Idempotency by accession number means only genuinely-new filings cost
    embedding work; already-stored accessions are skipped cheaply. Per-ticker
    failures are isolated by ingest_tickers' own try/except; this wrapper also
    swallows any top-level error so a sweep can never kill the worker loop.
    Returns the total number of new filings ingested."""
    forms = _env_forms()
    limit = _env_limit()
    raw_dir = os.getenv("FILINGS_RAW_DIR", "./data/filings_raw")
    clean = [t.strip().upper() for t in tickers if t and t.strip()]
    if not clean:
        return 0
    try:
        return ingest_tickers(clean, forms=forms, limit=limit,
                              raw_dir=raw_dir, max_chars=400_000, force=False)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  refresh_tracked_filings: sweep failed: {e}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-ingest 10-K/10-Q filings from SEC EDGAR into the pgvector RAG store.")
    parser.add_argument("--tickers", required=True,
                        help="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL")
    parser.add_argument("--forms", default="10-K,10-Q",
                        help="Comma-separated filing forms (default: 10-K,10-Q)")
    parser.add_argument("--limit", type=int, default=2,
                        help="Max filings per form per ticker (default: 2)")
    parser.add_argument("--max-chars", type=int, default=400_000,
                        help="Cap on extracted text per filing (default: 400000)")
    parser.add_argument("--raw-dir", default=os.getenv("FILINGS_RAW_DIR", "./data/filings_raw"),
                        help="Directory to archive extracted filing text")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest filings even if already stored")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    forms = [f.strip().upper() for f in args.forms.split(",") if f.strip()]
    if not tickers:
        print("❌ No tickers provided.")
        return 1

    print(f"🚀 EDGAR ingest — tickers={tickers} forms={forms} "
          f"limit={args.limit} force={args.force}")
    total = ingest_tickers(tickers, forms, args.limit, args.raw_dir,
                           args.max_chars, args.force)
    print(f"\n✅ Done. Ingested {total} filing(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
