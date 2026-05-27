from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple, Type

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from markdown_pdf import MarkdownPdf, Section

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agents import (
    State,
    data_agent,
    risk_agent,
    sentiment_agent,
    valuation_agent,
    fundamental_agent,
    writer_agent,
)
# from .agents import State, data_agent, risk_agent, writer_agent
from agents import debate_sentiment_agent, debate_valuation_agent, debate_fundamental_agent, debate_manager, route_debate
from utils.schemas import DebateReport
from utils.horizons import get_horizon
from utils.snapshots import save_snapshot


def build_chain_graph():
    """Default linear pipeline graph identical to the original implementation."""
    g = StateGraph(State)
    g.add_node("data", data_agent)
    g.add_node("sentiment", sentiment_agent)
    g.add_node("valuation", valuation_agent)
    g.add_node("fundamental", fundamental_agent)
    g.add_node("risk", risk_agent)
    g.add_node("writer", writer_agent)

    g.set_entry_point("data")
    # data fans out to the three independent analysts, which run in parallel;
    # risk has three incoming edges so it runs once, after all three finish.
    g.add_edge("data", "sentiment")
    g.add_edge("data", "valuation")
    g.add_edge("data", "fundamental")
    g.add_edge("sentiment", "risk")
    g.add_edge("valuation", "risk")
    g.add_edge("fundamental", "risk")
    g.add_edge("risk", "writer")
    g.add_edge("writer", END)
    return g.compile()

def build_final_recommendation_graph():
    g = StateGraph(State)
    # Multi-agent Debate
    g.add_node("debate_manager", debate_manager)
    g.add_node("debate_fundamental", debate_fundamental_agent)
    g.add_node("debate_sentiment", debate_sentiment_agent)
    g.add_node("debate_valuation", debate_valuation_agent)
    g.add_node("writer", writer_agent)
    g.set_entry_point("debate_manager")

    g.add_edge("debate_fundamental", "debate_manager")
    g.add_edge("debate_sentiment", "debate_manager")
    g.add_edge("debate_valuation", "debate_manager")

    g.add_conditional_edges(
        "debate_manager",
        route_debate,
        { 
            "END": "writer",
            "Fundamental":"debate_fundamental",
            "Sentiment":"debate_sentiment",
            "Valuation":"debate_valuation"
        },
    )
    g.add_edge("writer", END)
    
    return g.compile()


def run_pipeline_for_horizon(
    ticker: str,
    horizon_name: str,
    end_date: Optional[str] = None,
    *,
    persist: bool = True,
) -> Tuple[State, Optional[int]]:
    """
    Programmatic entry point for the worker / API. Runs the chain graph then
    the debate graph for one (ticker, horizon) pair — *no file IO* — and (if
    persist=True) writes a row to the `snapshots` table.

    The horizon preset (Short / Mid / Long → period + horizon_days) comes from
    utils.horizons; agents keep reading state.period / state.horizon_days as
    today. Returns (final_state, snapshot_id_or_None).
    """
    h = get_horizon(horizon_name)
    chain = build_chain_graph()
    state = State(
        ticker=ticker,
        period=h.period,
        interval="1d",
        horizon_days=h.horizon_days,
        end_date=end_date,
    )

    t0 = time.time()
    final = State(**chain.invoke(state, config=RunnableConfig()))

    debate_graph = build_final_recommendation_graph()
    final.debate = DebateReport(
        agent_list=["fundamental", "sentiment", "valuation"])
    final.debate.agent_max_turn = 5
    final = State(**debate_graph.invoke(
        final, config=RunnableConfig(recursion_limit=100)))
    latency_ms = int((time.time() - t0) * 1000)

    snapshot_id: Optional[int] = None
    if persist:
        snapshot_id = save_snapshot(
            ticker=ticker, horizon=h.name, state=final,
            latency_ms=latency_ms, cost_usd=None)
    return final, snapshot_id


def resolve_mode(mode: Optional[str] = None) -> str:
    env_value = os.getenv("ANALYSIS_MODE")
    value = mode or env_value or "chain"
    value = value.strip().lower()
    if value not in {"chain", "debate"}:
        return "chain"
    return value


def get_workflow(mode: Optional[str] = None) -> Tuple[object, Type[State]]:
    """
    Returns a compiled LangGraph graph and the associated state class based
    on the chosen analysis mode.
    """
    resolved = resolve_mode(mode)
    if resolved == "debate":
        from debate_agents import DebateState, build_debate_graph

        return build_debate_graph(), DebateState
    return build_chain_graph(), State


def build_graph():
    """Backward-compatible helper returning the chain graph."""
    return build_chain_graph()


def run_all_graphs(ticker="GOOGL", period="1wk", interval="1d", horizon_days=30, end_date=None):
    final_state_dict = {}
    # if not os.path.exists("final_state.json"):
    graph, state_cls = get_workflow()
    with open("frontend/public/visualizations/langgraph_collaboration.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    # state = state_cls(ticker="AAPL", period="1wk", interval="1d", horizon_days=30)
    state = state_cls(ticker=ticker, period=period, interval=interval, horizon_days=horizon_days, end_date=end_date)
    final_state = graph.invoke(state, config=RunnableConfig())
    print(final_state['report'].markdown_report)

    # Save to JSON file
    final_state_dict = final_state 
    state_obj = State(**final_state_dict)
    with open("final_state.json", "w") as f:
            f.write(state_obj.model_dump_json(indent=2))

    with open("final_state.json", "r") as f:
        final_state = State.model_validate_json(f.read())

        debateGraph = build_final_recommendation_graph()
        debateReport = DebateReport(agent_list=["fundamental", "sentiment", "valuation"])
        debateReport.agent_max_turn = 5
        final_state.debate = debateReport
        with open("frontend/public/visualizations/langgraph_debate.png", "wb") as f:
            f.write(debateGraph.get_graph().draw_mermaid_png())
        final_state = debateGraph.invoke(final_state, config=RunnableConfig(recursion_limit=100), verbose=True)
        print('Debate Terminated: ',final_state['debate'].terminated)

        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(final_state['report'].markdown_report))
        pdf.save(f"AnalysisReport_{final_state['ticker']}.pdf")

        # Save to JSON file
        final_state_dict = final_state 
        state_obj = State(**final_state_dict)
        with open("final_state_with_debate.json", "w") as f:
                f.write(state_obj.model_dump_json(indent=2))
    return state_obj

    # debateGraph = build_debate_graph(final_state)
    # final_state = debateGraph.invoke(final_state, config=RunnableConfig(), verbose=True)


if __name__ == "__main__":
    run_all_graphs()
