"""In-program LLM cost metering + budget ceilings.

Every chat model constructed by config.get_llm() gets a UsageCallback attached,
which accumulates exact per-call token counts (from langchain's usage_metadata)
into a process-global RUN_METER, priced per the model's $/Mtok rates. The
pipeline (run_pipeline_for_horizon) resets the meter at run start and persists
the total into the snapshot's `cost_usd` column at the end — so cost is visible
per snapshot immediately, with no AWS billing lag.

Two ceilings, both opt-in (0 = disabled):
  • RUN_TOKEN_BUDGET       — max total tokens for ONE pipeline run. Checked
    before each LLM call; exceeding raises, which fails the job with a clear
    reason (surfaced by the existing failed-job UX). Catches runaway loops
    mid-run — something AWS-side budgets (8–24h billing lag) cannot do.
  • DAILY_COST_BUDGET_USD  — checked by the worker before claiming a job (see
    worker.py); sums today's persisted snapshot costs. NB: failed runs' spend
    isn't persisted, so this gate under-counts by whatever failures burned.

The worker is a single serial process, so one global meter is safe; the Lock
covers LangGraph's parallel fan-out (sentiment ‖ valuation ‖ fundamental run
concurrently inside a superstep).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler


# $ per 1M tokens (input, output), matched by substring against the model id.
# First hit wins. Override/extend via LLM_PRICE_INPUT_PER_MTOK /
# LLM_PRICE_OUTPUT_PER_MTOK for models not listed here.
_PRICES = [
    ("opus", 15.00, 75.00),
    ("sonnet", 3.00, 15.00),
    ("haiku-4", 1.00, 5.00),
    ("haiku", 0.25, 1.25),        # claude-3-haiku (legacy)
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.00),
    ("gemini", 1.25, 5.00),
    ("llama", 0.0, 0.0),          # local Ollama — free
    ("nomic", 0.0, 0.0),
]


def resolve_pricing(model_id: str) -> tuple[float, float]:
    """($/Mtok input, $/Mtok output) for a model id; env override wins; (0,0)
    for unknown models (tokens still counted, cost just reads $0)."""
    env_in, env_out = os.getenv("LLM_PRICE_INPUT_PER_MTOK"), os.getenv("LLM_PRICE_OUTPUT_PER_MTOK")
    if env_in and env_out:
        return float(env_in), float(env_out)
    mid = (model_id or "").lower()
    for needle, pin, pout in _PRICES:
        if needle in mid:
            return pin, pout
    return 0.0, 0.0


class RunMeter:
    """Thread-safe accumulator for one pipeline run's token usage + cost."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.input_tokens = 0
            self.output_tokens = 0
            self.cost_usd = 0.0
            self.calls = 0

    def add(self, input_tokens: int, output_tokens: int,
            price_in: float, price_out: float) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += (input_tokens * price_in + output_tokens * price_out) / 1e6
            self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "calls": self.calls,
            }


RUN_METER = RunMeter()


class RunBudgetExceeded(RuntimeError):
    """One pipeline run blew past RUN_TOKEN_BUDGET — fail the job, don't keep paying."""


def _run_token_budget() -> int:
    try:
        return int(os.getenv("RUN_TOKEN_BUDGET", "0"))
    except ValueError:
        return 0


class UsageCallback(BaseCallbackHandler):
    """Per-model callback: counts tokens into RUN_METER and enforces the
    per-run token ceiling. Attached at model construction in get_llm(), so it
    fires on every call regardless of how config propagates through langgraph."""

    raise_error = True  # let RunBudgetExceeded propagate instead of being logged

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.price_in, self.price_out = resolve_pricing(model_id)

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._check_budget()

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._check_budget()

    def _check_budget(self) -> None:
        budget = _run_token_budget()
        if budget > 0 and RUN_METER.total_tokens >= budget:
            raise RunBudgetExceeded(
                f"Run token budget exceeded: {RUN_METER.total_tokens:,} tokens used "
                f"(RUN_TOKEN_BUDGET={budget:,}, ~${RUN_METER.cost_usd:.2f} so far). "
                f"Refusing further LLM calls for this run.")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        tin, tout = _extract_usage(response)
        if tin or tout:
            RUN_METER.add(tin, tout, self.price_in, self.price_out)


def _extract_usage(response: Any) -> tuple[int, int]:
    """(input_tokens, output_tokens) from an LLMResult, tolerating provider
    shape differences. usage_metadata (langchain-core standard, populated by
    ChatBedrockConverse / ChatOpenAI / ChatAnthropic) is preferred; falls back
    to llm_output['token_usage'|'usage']. Returns (0, 0) when absent."""
    try:
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        out: Optional[dict] = getattr(response, "llm_output", None) or {}
        usage = out.get("token_usage") or out.get("usage") or {}
        tin = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        tout = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        return int(tin), int(tout)
    except Exception:  # noqa: BLE001 — metering must never break an LLM call
        return 0, 0
