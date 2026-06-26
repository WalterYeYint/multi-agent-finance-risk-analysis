"""
RAG utilities for storing and retrieving 10-K/10-Q financial documents.

Backed by Postgres + pgvector (see utils/db.py). Filing metadata lives in the
`filings` table; chunk text and embeddings live in `filing_chunks`. Documents
can be ingested from local PDFs (batch_ingest_documents) or directly as text
(ingest_text — used by the SEC EDGAR auto-ingest script, utils/edgar_ingest.py).
"""

import os
import glob
import calendar
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import date

import numpy as np
from psycopg.types.json import Jsonb
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_embeddings
from utils.db import connect, ensure_schema


def _normalize_filing_type(filing_type: str) -> str:
    """Normalise '10-K'/'10k'/'10-Q' to a dash-free upper form ('10K', '10Q')."""
    return (filing_type or "").replace("-", "").replace("_", "").upper()


def _period_dates_from_months(year: int, start_month: int,
                              end_month: int) -> tuple[date, date]:
    """Build a (period_start, period_end) date pair from a year + month range."""
    last_day = calendar.monthrange(year, end_month)[1]
    return date(year, start_month, 1), date(year, end_month, last_day)


class FundamentalRAG:
    """RAG system for fundamental analysis using 10-K/10-Q documents."""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # The pgvector `embedding` column is dimension-agnostic, so chunks from
        # different embedding providers can coexist. We namespace every chunk by
        # an `embedding_model` tag that includes the vector dimension, and
        # retrieval filters on it — distance ops only ever see same-dim vectors.
        try:
            self.embedding_dim = len(self.embeddings.embed_query("test"))
        except Exception as e:
            print(f"⚠️  Could not probe embedding dimension: {e}")
            self.embedding_dim = 0
        self.embedding_model = f"{self._embedding_provider()}-{self.embedding_dim}d"
        print(f"🔍 RAG embedding namespace: {self.embedding_model}")

        ensure_schema()

    @staticmethod
    def _embedding_provider() -> str:
        """Best-effort label for the active embeddings provider (namespacing only)."""
        if os.getenv("OPENAI_API_KEY"):
            return "openai:" + os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
        provider = os.getenv("MODEL_PROVIDER", "").lower()
        if provider in ("", "auto", "ollama"):
            return "ollama:" + os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return provider

    # ----------------------------------------------------------------- ingest
    def _find_filing_id(self, cur, accession_no, ticker, filing_type,
                        filing_year, filing_start_month, filing_end_month):
        """Return the id of an existing filing row, or None."""
        if accession_no:
            cur.execute("SELECT id FROM filings WHERE accession_no = %s", (accession_no,))
            row = cur.fetchone()
            if row:
                return row[0]
        cur.execute(
            """SELECT id FROM filings
               WHERE ticker = %s AND filing_type = %s AND filing_year = %s
                 AND filing_start_month = %s AND filing_end_month = %s""",
            (ticker, filing_type, filing_year, filing_start_month, filing_end_month),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _ingest_chunks(self, chunks: List[Document], *, ticker: str, filing_type: str,
                       filing_year: int, filing_start_month: int, filing_end_month: int,
                       period_start: date, period_end: date, source: str,
                       accession_no: Optional[str] = None,
                       company: Optional[str] = None, force: bool = False) -> bool:
        """Embed and store chunks for one filing. Idempotent unless force=True."""
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        if not chunks:
            print(f"⚠️  No non-empty chunks to ingest for {ticker} {filing_type}.")
            return False

        ticker = ticker.upper()
        filing_type = _normalize_filing_type(filing_type)
        ensure_schema()

        try:
            with connect() as conn:
                with conn.cursor() as cur:
                    # Dedup is per (filing, embedding_model): one filing row can
                    # hold chunks from several embedding providers side by side,
                    # so switching MODEL_PROVIDER later needs no re-ingest.
                    filing_id = self._find_filing_id(
                        cur, accession_no, ticker, filing_type,
                        filing_year, filing_start_month, filing_end_month)
                    if filing_id is not None:
                        cur.execute(
                            "SELECT count(*) FROM filing_chunks "
                            "WHERE filing_id = %s AND embedding_model = %s",
                            (filing_id, self.embedding_model))
                        model_chunks = cur.fetchone()[0]
                        if model_chunks > 0:
                            if not force:
                                print(f"⏭️  {ticker} {filing_type} {filing_year} "
                                      f"already ingested for embedding model "
                                      f"'{self.embedding_model}' — skipping.")
                                return True
                            # force: replace only this model's chunks, leaving
                            # the filing row and other providers' chunks intact.
                            cur.execute(
                                "DELETE FROM filing_chunks "
                                "WHERE filing_id = %s AND embedding_model = %s",
                                (filing_id, self.embedding_model))
                    else:
                        cur.execute(
                            """INSERT INTO filings
                               (ticker, filing_type, filing_year, filing_start_month,
                                filing_end_month, period_start, period_end,
                                accession_no, company, source)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               RETURNING id""",
                            (ticker, filing_type, filing_year, filing_start_month,
                             filing_end_month, period_start, period_end,
                             accession_no, company, source),
                        )
                        filing_id = cur.fetchone()[0]

                    texts = [c.page_content for c in chunks]
                    vectors = self.embeddings.embed_documents(texts)

                    rows = []
                    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                        meta = dict(chunk.metadata or {})
                        meta.update({
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "filing_year": filing_year,
                            "filing_start_month": filing_start_month,
                            "filing_end_month": filing_end_month,
                            "source": source,
                        })
                        rows.append((
                            filing_id, ticker, idx, chunk.page_content,
                            np.asarray(vec, dtype=np.float32),
                            self.embedding_model, Jsonb(meta),
                        ))

                    # NB: do NOT use cur.executemany() here. psycopg3's
                    # executemany() sends rows via libpq *pipeline mode*, which
                    # the Supabase transaction-mode pooler (pgbouncer) does not
                    # support — it drops the connection mid-batch ("Pipeline
                    # [BAD]" / "SSL SYSCALL error: EOF detected"). Instead, insert
                    # with single multi-row INSERT statements (one round-trip per
                    # batch, no pipeline). Together with prepare_threshold=None on
                    # the connection (see utils/db.py) this is fully pooler-safe.
                    # Batch so the bound-parameter count (7 per row) stays well
                    # under Postgres's 65535-params-per-statement limit.
                    _cols = ("(filing_id, ticker, chunk_index, content, "
                             "embedding, embedding_model, metadata)")
                    _BATCH = 500
                    for _i in range(0, len(rows), _BATCH):
                        _batch = rows[_i:_i + _BATCH]
                        _values = ", ".join(
                            ["(%s, %s, %s, %s, %s, %s, %s)"] * len(_batch))
                        _flat = [_v for _row in _batch for _v in _row]
                        cur.execute(
                            f"INSERT INTO filing_chunks {_cols} VALUES {_values}",
                            _flat,
                        )
                    # num_chunks is the total across all embedding models.
                    cur.execute(
                        "UPDATE filings SET num_chunks = "
                        "(SELECT count(*) FROM filing_chunks WHERE filing_id = %s) "
                        "WHERE id = %s",
                        (filing_id, filing_id))
                conn.commit()

            print(f"✅ Ingested {len(chunks)} chunks for {ticker} {filing_type} "
                  f"{filing_year} under embedding model '{self.embedding_model}'")
            return True

        except Exception as e:
            print(f"❌ Error ingesting {ticker} {filing_type}: {e}")
            return False

    def ingest_document(self, ticker: str, filing_type: str, filing_year: int,
                        filing_start_month: int, filing_end_month: int,
                        document_path: str) -> bool:
        """Ingest a 10-K/10-Q PDF document into the vector store."""
        try:
            loader = PyPDFLoader(document_path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
        except Exception as e:
            print(f"Error loading PDF for {ticker}: {e}")
            return False

        # PDF filenames only carry a single year, so the period stays within it.
        period_start, period_end = _period_dates_from_months(
            filing_year, filing_start_month, filing_end_month)
        return self._ingest_chunks(
            chunks, ticker=ticker, filing_type=filing_type, filing_year=filing_year,
            filing_start_month=filing_start_month, filing_end_month=filing_end_month,
            period_start=period_start, period_end=period_end, source=document_path,
        )

    def ingest_text(self, text: str, *, ticker: str, filing_type: str, filing_year: int,
                    filing_start_month: int, filing_end_month: int,
                    period_start: date, period_end: date, source: str,
                    accession_no: Optional[str] = None, company: Optional[str] = None,
                    force: bool = False) -> bool:
        """Ingest a filing supplied as raw text (e.g. SEC EDGAR HTML extracted to text)."""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.text_splitter.split_documents([doc])
        return self._ingest_chunks(
            chunks, ticker=ticker, filing_type=filing_type, filing_year=filing_year,
            filing_start_month=filing_start_month, filing_end_month=filing_end_month,
            period_start=period_start, period_end=period_end, source=source,
            accession_no=accession_no, company=company, force=force,
        )

    def has_filing(self, accession_no: str) -> bool:
        """
        Return True if this filing's chunks are already stored *for the active
        embedding model*. Another provider's chunks for the same filing don't
        count — so re-running ingest under a new provider adds its vectors
        rather than skipping.
        """
        if not accession_no:
            return False
        ensure_schema()
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT 1 FROM filing_chunks c
                   JOIN filings f ON f.id = c.filing_id
                   WHERE f.accession_no = %s AND c.embedding_model = %s
                   LIMIT 1""",
                (accession_no, self.embedding_model))
            return cur.fetchone() is not None

    # --------------------------------------------------------------- retrieve
    @staticmethod
    def _build_search_sql(filing_type: Optional[str]) -> str:
        """Cosine-distance search SQL. A filing matches if its coverage period
        overlaps [from_date, to_date]: period_end >= from AND period_start <= to."""
        sql = [
            "SELECT c.content, c.metadata",
            "FROM filing_chunks c",
            "JOIN filings f ON f.id = c.filing_id",
            "WHERE c.ticker = %(ticker)s",
            "  AND c.embedding_model = %(model)s",
            "  AND f.period_end >= %(from_date)s",
            "  AND f.period_start <= %(to_date)s",
        ]
        if filing_type:
            sql.append("  AND f.filing_type = %(filing_type)s")
        sql.append("ORDER BY c.embedding <=> %(qvec)s")
        sql.append("LIMIT %(k)s")
        return "\n".join(sql)

    @staticmethod
    def _search(cur, sql: str, base_params: Dict[str, Any],
                qvec: np.ndarray) -> List[Document]:
        """Run one similarity search on an open cursor and return Documents."""
        cur.execute(sql, {**base_params, "qvec": qvec})
        return [Document(page_content=row[0], metadata=row[1] or {})
                for row in cur.fetchall()]

    def _diagnose_empty(self, cur, ticker: str) -> None:
        """Turn a silent zero-result into a clear log line when the cause is an
        embedding-model namespace mismatch (filings ingested by another provider)."""
        cur.execute(
            "SELECT DISTINCT embedding_model FROM filing_chunks WHERE ticker = %s",
            (ticker.upper(),),
        )
        models = [r[0] for r in cur.fetchall()]
        if models and self.embedding_model not in models:
            print(f"⚠️  Retrieval for {ticker} found nothing: filings are stored "
                  f"under embedding model(s) {models}, but the active model is "
                  f"'{self.embedding_model}'. Re-ingest with the current provider, "
                  f"or switch the embeddings provider back.")

    def retrieve_relevant_chunks_batch(
        self,
        ticker: str,
        queries: List[str],
        filing_type: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        k: int = 5,
    ) -> List[List[Document]]:
        """
        Retrieve the top-k chunks for each query in `queries`.

        All queries are embedded in a single `embed_documents` call and all
        searches run on one reused connection. A filing is in range if its
        coverage period overlaps [from_date, to_date].
        """
        if not queries:
            return []

        from_date = from_date or date.min
        to_date = to_date or date.today()

        vectors = self.embeddings.embed_documents(list(queries))
        sql = self._build_search_sql(filing_type)
        base_params: Dict[str, Any] = {
            "ticker": ticker.upper(),
            "model": self.embedding_model,
            "from_date": from_date,
            "to_date": to_date,
            "k": k,
        }
        if filing_type:
            base_params["filing_type"] = _normalize_filing_type(filing_type)

        ensure_schema()
        with connect() as conn, conn.cursor() as cur:
            results = [
                self._search(cur, sql, base_params,
                             np.asarray(vec, dtype=np.float32))
                for vec in vectors
            ]
            if not any(results):
                self._diagnose_empty(cur, ticker)
        return results

    def retrieve_relevant_chunks(
        self,
        ticker: str,
        query: str,
        filing_type: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        k: int = 5,
    ) -> List[Document]:
        """Retrieve the top-k chunks for a single query (see the batch variant)."""
        batch = self.retrieve_relevant_chunks_batch(
            ticker, [query], filing_type=filing_type,
            from_date=from_date, to_date=to_date, k=k)
        return batch[0] if batch else []

    def get_available_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """Return a list of stored filings for a ticker."""
        try:
            ensure_schema()
            with connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT ticker, filing_type, source, company,
                              ingested_at, num_chunks
                       FROM filings WHERE ticker = %s
                       ORDER BY ingested_at DESC""",
                    (ticker.upper(),),
                )
                rows = cur.fetchall()
            return [
                {
                    "ticker": r[0],
                    "filing_type": r[1],
                    "source": r[2],
                    "company": r[3],
                    "ingestion_date": r[4].isoformat() if r[4] else None,
                    "num_chunks": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            print(f"Error getting available filings for {ticker}: {e}")
            return []


# Sample 10-K content for demo/initialisation (used by initialize_sample_data).
SAMPLE_DOCUMENTS = {
    "AAPL": {"company": "Apple Inc."},
    "MSFT": {"company": "Microsoft Corporation"},
    "GOOGL": {"company": "Alphabet Inc."},
}

_SAMPLE_10K_CONTENT = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION

FORM 10-K — ANNUAL REPORT

For the fiscal year ended September 30, 2023

Apple Inc.

BUSINESS
The Company designs, manufactures and markets smartphones, personal computers,
tablets, wearables and accessories, and sells a variety of related services.

FINANCIAL PERFORMANCE
Total net sales were $383.3 billion for 2023, compared to $394.3 billion for 2022.
Gross margin percentage was 44.1% for 2023, compared to 43.3% for 2022.
Operating income was $114.3 billion for 2023, compared to $119.4 billion for 2022.

RISK FACTORS
Global and regional economic conditions could materially adversely affect the
Company. Adverse economic conditions can reduce demand for the Company's products
and services.
"""


def initialize_sample_data(rag_system: FundamentalRAG) -> None:
    """Seed the RAG store with small sample 10-K text (for demos without filings)."""
    for ticker, info in SAMPLE_DOCUMENTS.items():
        content = _SAMPLE_10K_CONTENT.replace("Apple Inc.", info["company"])
        rag_system.ingest_text(
            content,
            ticker=ticker,
            filing_type="10-K",
            filing_year=2023,
            filing_start_month=1,
            filing_end_month=12,
            period_start=date(2023, 1, 1),
            period_end=date(2023, 12, 31),
            source=f"sample::{ticker}",
            company=info["company"],
        )
        print(f"Initialized sample data for {ticker} ({info['company']})")


def validate_file_path(file_path: str) -> bool:
    """Validate that the file exists and is a PDF."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print(f"⚠️  Warning: File is not a PDF: {file_path}")
        print("   The system is designed for PDF documents but will attempt to process.")
    return True


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    if not ticker or len(ticker) < 1 or len(ticker) > 5:
        print(f"❌ Error: Invalid ticker symbol: {ticker}")
        print("   Ticker should be 1-5 characters long (e.g., AAPL, MSFT)")
        return False
    return True


def validate_filing_type(filing_type: str) -> bool:
    """Validate filing type."""
    valid_types = ["10K", "10Q", "8K", "20F"]
    if _normalize_filing_type(filing_type) not in valid_types:
        print(f"❌ Error: Invalid filing type: {filing_type}")
        print(f"   Valid types are: {', '.join(valid_types)}")
        return False
    return True


def ingest_single_document(
    rag_system: FundamentalRAG,
    ticker: str,
    filing_type: str,
    filing_year: int,
    filing_start_month: int,
    filing_end_month: int,
    document_path: str,
) -> bool:
    """Ingest a single PDF document, with input validation."""
    print("\n📄 Ingesting document:")
    print(f"   Ticker: {ticker}")
    print(f"   Filing Type: {filing_type}")
    print(f"   Path: {document_path}")

    if not validate_ticker(ticker):
        return False
    if not validate_filing_type(filing_type):
        return False
    if not validate_file_path(document_path):
        return False

    try:
        success = rag_system.ingest_document(
            ticker.upper(), filing_type, filing_year,
            filing_start_month, filing_end_month, document_path)
        if success:
            print(f"✅ Successfully ingested {ticker} {filing_type}")
        else:
            print(f"❌ Failed to ingest {ticker} {filing_type}")
        return success
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


def parse_filename_metadata(filename: str) -> dict[str, Any] | None:
    """
    Parse metadata from a filing PDF filename.

    Expected format: TICKER-FILINGTYPE-Q#-STARTMONTH-ENDMONTH-YEAR.pdf
    e.g. AAPL-10Q-Q3-4-6-2025.pdf
    """
    basename = os.path.basename(filename).replace('.pdf', '').replace('.PDF', '')
    parts = basename.replace('-', '_').split('_')

    if len(parts) >= 6:
        try:
            return {
                'ticker': parts[0].upper(),
                'filing_type': parts[1].upper(),
                'filing_year': int(parts[5]),
                'filing_start_month': int(parts[3]),
                'filing_end_month': int(parts[4]),
            }
        except (ValueError, IndexError):
            return None
    return None


def batch_ingest_documents(rag_system: FundamentalRAG, directory: str) -> int:
    """Ingest all PDF documents from a directory."""
    if not os.path.exists(directory):
        print(f"❌ Error: Directory not found: {directory}")
        return 0

    pdf_files: List[str] = []
    for pattern in ("*.pdf", "*.PDF"):
        pdf_files.extend(glob.glob(os.path.join(directory, pattern)))

    if not pdf_files:
        print(f"❌ No PDF files found in directory: {directory}")
        return 0

    print(f"\n📁 Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")

    successful_ingestions = 0
    for pdf_file in pdf_files:
        print("\n" + "=" * 60)
        metadata = parse_filename_metadata(pdf_file)
        if metadata:
            if ingest_single_document(
                rag_system,
                metadata['ticker'],
                metadata['filing_type'],
                metadata['filing_year'],
                metadata['filing_start_month'],
                metadata['filing_end_month'],
                pdf_file,
            ):
                successful_ingestions += 1
        else:
            print(f"⚠️  Could not parse metadata from filename: {os.path.basename(pdf_file)}")
            print("   Expected: TICKER-FILINGTYPE-Q#-STARTMONTH-ENDMONTH-YEAR.pdf")
            print("   Skipping this file.")

    return successful_ingestions
