"""Cost metering + budget ceilings (utils/cost_meter.py) — pure logic, no LLM/DB."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # src/

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

from utils.cost_meter import (  # noqa: E402
    RUN_METER, RunBudgetExceeded, UsageCallback, resolve_pricing,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("RUN_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("LLM_PRICE_INPUT_PER_MTOK", raising=False)
    monkeypatch.delenv("LLM_PRICE_OUTPUT_PER_MTOK", raising=False)
    RUN_METER.reset()
    yield
    RUN_METER.reset()


def _llm_result(tin, tout):
    msg = AIMessage(content="x",
                    usage_metadata={"input_tokens": tin, "output_tokens": tout,
                                    "total_tokens": tin + tout})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_pricing_table_resolution():
    assert resolve_pricing("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == (3.00, 15.00)
    assert resolve_pricing("us.anthropic.claude-haiku-4-5-20251001-v1:0") == (1.00, 5.00)
    assert resolve_pricing("gpt-4o") == (2.50, 10.00)
    assert resolve_pricing("gpt-4o-mini") == (0.15, 0.60)
    assert resolve_pricing("llama3.2:3b") == (0.0, 0.0)       # local = free
    assert resolve_pricing("totally-unknown-model") == (0.0, 0.0)


def test_pricing_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "1.5")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "6.0")
    assert resolve_pricing("anything") == (1.5, 6.0)


def test_meter_accumulates_cost():
    cb = UsageCallback("us.anthropic.claude-sonnet-4-5-20250929-v1:0")  # $3/$15 per Mtok
    cb.on_llm_end(_llm_result(1_000_000, 100_000))
    cb.on_llm_end(_llm_result(500_000, 50_000))
    s = RUN_METER.snapshot()
    assert s["input_tokens"] == 1_500_000
    assert s["output_tokens"] == 150_000
    assert s["calls"] == 2
    # 1.5 Mtok * $3 + 0.15 Mtok * $15 = 4.5 + 2.25 = 6.75
    assert abs(s["cost_usd"] - 6.75) < 1e-6


def test_meter_reset_between_runs():
    cb = UsageCallback("gpt-4o")
    cb.on_llm_end(_llm_result(1000, 100))
    assert RUN_METER.total_tokens == 1100
    RUN_METER.reset()
    assert RUN_METER.total_tokens == 0
    assert RUN_METER.snapshot()["cost_usd"] == 0.0


def test_missing_usage_metadata_is_zero_not_crash():
    cb = UsageCallback("gpt-4o")
    msg = AIMessage(content="no usage attached")
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert RUN_METER.total_tokens == 0
    assert RUN_METER.snapshot()["calls"] == 0


def test_run_budget_disabled_by_default():
    cb = UsageCallback("gpt-4o")
    cb.on_llm_end(_llm_result(10_000_000, 1_000_000))
    cb.on_chat_model_start({}, [])          # no budget set → never raises
    cb.on_llm_start({}, [])


def test_run_budget_raises_when_exceeded(monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "1000")
    cb = UsageCallback("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    cb.on_chat_model_start({}, [])          # under budget → fine
    cb.on_llm_end(_llm_result(900, 200))    # now at 1100 > 1000
    with pytest.raises(RunBudgetExceeded):
        cb.on_chat_model_start({}, [])
    with pytest.raises(RunBudgetExceeded):
        cb.on_llm_start({}, [])
