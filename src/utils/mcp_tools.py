"""Load a self-hosted SEC EDGAR MCP server's tools as LangChain tools.

This is the *client* half of N4: the fundamental agent already has
``query_10k_documents`` (semantic RAG over filing *text* — good at narrative,
bad at exact numbers because get_text() flattens the financial tables). A SEC
EDGAR MCP server exposes the **structured XBRL** financials (revenue, EPS,
margins, tagged by concept/period) that the RAG cannot reliably return, so the
two are complementary.

Design / safety:
  * OFF by default — set ``USE_SEC_MCP=1`` to enable. When off, this returns [],
    so the fundamental agent behaves exactly as before (RAG-only).
  * The server is *self-hosted* and fully env-configurable (we don't hard-wire a
    single upstream — that bit us with the Polygon→Massive rebrand):
      - ``SEC_MCP_URL``      → connect to an already-running HTTP endpoint, else
      - ``SEC_MCP_COMMAND``  → spawn a stdio subprocess (default: ``uvx``)
      - ``SEC_MCP_ARGS``     → args for it (default: ``sec-edgar-mcp``)
  * Graceful degradation: any failure (adapters missing, server won't launch,
    network down) logs a warning and returns [] — a fresh/unconfigured box, or a
    flaky SEC API, can never break a pipeline run.
  * Tools are loaded once and cached (loading spawns the server to read its tool
    schema; each subsequent *tool call* opens its own short-lived session, per
    langchain-mcp-adapters' auto-session model).

get_tools() is async; the pipeline is sync (worker / eval / main all run without
an outer event loop), so we bridge with asyncio.run — see run_agent_async in
agents.py for the matching call-side.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

_CACHE: list | None = None  # loaded tools, cached across calls in a process

# The stefanoamorelli/sec-edgar-mcp server exposes ~21 tools (financials, XBRL,
# filings, insider trading, form-4...). Handing all of them to a small local
# model (llama3.2:3b) overwhelms its tool selection, so by default we expose only
# the exact-financials tools that complement the RAG. Override with
# SEC_MCP_TOOLS="all" or a custom comma-separated allowlist.
_DEFAULT_TOOLS = "get_financials,get_company_facts,get_key_metrics,get_segment_data,compare_periods"


def sec_mcp_enabled() -> bool:
    return os.getenv("USE_SEC_MCP", "").strip().lower() in ("1", "true", "yes", "on")


def _select(tools: list) -> list:
    """Narrow the server's tools to the configured allowlist (default: financials)."""
    raw = os.getenv("SEC_MCP_TOOLS", "").strip()
    if raw.lower() == "all":
        return tools
    wanted = {n.strip() for n in (raw or _DEFAULT_TOOLS).split(",") if n.strip()}
    picked = [t for t in tools if t.name in wanted]
    # If none matched (e.g. the server renamed its tools), fall back to all rather
    # than silently handing the agent zero financial tools.
    return picked or tools


def _connection() -> dict[str, Any]:
    """Build the langchain-mcp-adapters connection dict for the SEC EDGAR server."""
    url = os.getenv("SEC_MCP_URL", "").strip()
    if url:
        return {"transport": "streamable_http", "url": url}

    command = os.getenv("SEC_MCP_COMMAND", "uvx").strip()
    args = (os.getenv("SEC_MCP_ARGS", "").strip() or "sec-edgar-mcp").split()

    # The server needs a real SEC contact string. The project stores it as
    # SEC_USER_AGENT; sec-edgar-mcp reads SEC_EDGAR_USER_AGENT — map it across so
    # a single .env var configures both the in-process EDGAR ingest and the MCP
    # server. Inherit the rest of the environment (PATH, etc.).
    env = {**os.environ}
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if ua and not env.get("SEC_EDGAR_USER_AGENT"):
        env["SEC_EDGAR_USER_AGENT"] = ua

    return {"transport": "stdio", "command": command, "args": args, "env": env}


def load_sec_mcp_tools() -> list:
    """Return the SEC EDGAR MCP server's tools as LangChain tools (or [])."""
    global _CACHE
    if not sec_mcp_enabled():
        return []
    if _CACHE is not None:
        return _CACHE

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"sec": _connection()})
        tools = _select(asyncio.run(client.get_tools()))
        names = ", ".join(t.name for t in tools) or "(none)"
        print(f"✅ SEC EDGAR MCP: using {len(tools)} tools — {names}")
        _CACHE = tools
    except Exception as e:  # noqa: BLE001 — never let MCP setup break a run
        print(f"⚠️  SEC EDGAR MCP unavailable ({type(e).__name__}: {e}); "
              f"fundamental agent falls back to RAG-only.")
        _CACHE = []
    return _CACHE


_XBRL_CACHE: dict[str, str] = {}  # ticker -> concatenated financial-tool text


def _ticker_arg(tool, ticker: str) -> dict:
    """Map the ticker onto whatever the tool names its identifier argument."""
    args = getattr(tool, "args", {}) or {}
    for cand in ("identifier", "ticker", "symbol", "cik", "company"):
        if cand in args:
            return {cand: ticker}
    return {next(iter(args), "identifier"): ticker}


def fetch_sec_xbrl_text(ticker: str,
                        tool_names: tuple = ("get_company_facts", "get_financials",
                                             "get_key_metrics", "get_segment_data")) -> str:
    """Raw text of the SEC MCP financial tools for ``ticker`` (or "").

    Used as an EXTRA grounding source: the exact XBRL numbers the fundamental
    agent pulls via MCP never appear in the flattened RAG text chunks, so without
    this the number-grounding check flags correct figures as hallucinations.
    Best-effort and cached per ticker; returns "" if MCP is disabled / the server
    is unavailable / a call errors — grounding then just uses the RAG source.
    """
    tools = load_sec_mcp_tools()
    if not tools:
        return ""
    key = ticker.upper()
    if key in _XBRL_CACHE:
        return _XBRL_CACHE[key]
    by_name = {t.name: t for t in tools}
    parts: list[str] = []
    for name in tool_names:
        tool = by_name.get(name)
        if tool is None:
            continue
        try:
            res = asyncio.run(tool.ainvoke(_ticker_arg(tool, key)))
            parts.append(res if isinstance(res, str) else str(res))
        except Exception as e:  # noqa: BLE001 — a bonus source, never required
            print(f"⚠️  SEC MCP {name} failed for {ticker}: {e}")
    text = "\n".join(parts)
    _XBRL_CACHE[key] = text
    return text
