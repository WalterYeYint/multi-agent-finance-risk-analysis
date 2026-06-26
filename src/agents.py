import ast
from datetime import datetime
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from utils.config import get_llm
from utils.tools import get_price_history, get_recent_news, query_10k_documents, period_to_months_range, compute_risk, compute_valuation_metrics
from utils.constants import RISK_SYSTEM, SENTIMENT_SYSTEM, VALUATION_SYSTEM, FUNDAMENTAL_SYSTEM
from utils.schemas import (
    MarketData, NewsBundle, NewsItem, RiskMetrics, RiskReport,
    SentimentSummary, SentimentExtract, ValuationMetrics,
    FundamentalAnalysis, FundamentalExtract, DebateReport
)
from utils.rag_utils import FundamentalRAG
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

# LangSmith visibility
# import os
# os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")


class State(BaseModel):
    ticker: str
    period: str = "1y"
    interval: str = "1d"
    horizon_days: int = 30
    market: Optional[MarketData] = None
    news: Optional[NewsBundle] = None
    sentiment: Optional[SentimentSummary] = None
    valuation: Optional[ValuationMetrics] = None
    fundamental: Optional[FundamentalAnalysis] = None
    metrics: Optional[RiskMetrics] = None
    report: Optional[RiskReport] = None
    debate: Optional[DebateReport] = None
    end_date: Optional[str] = None


def data_agent(state: State, config: RunnableConfig):
    # llm = get_llm()
    # _ = llm.invoke([SystemMessage(content=DATA_SYSTEM), HumanMessage(content=f"ticker={state.ticker}")])  # no-op, just for tracing
    end_date = None
    if state.end_date is not None: 
        end_date = datetime.strptime(state.end_date, "%Y-%m-%d")
    price_csv = get_price_history.invoke({"ticker": state.ticker, "period": state.period, "interval": state.interval, "end_date": end_date})
    news_raw = get_recent_news.invoke({"ticker": state.ticker, "period": state.period, "end_date": end_date})
    items = []
    try:
        for r in ast.literal_eval(news_raw):
            # print("News is:", r["content"])
            items.append(NewsItem(date=str(r["date"]), headline=str(r["headline"]), sentiment=str(r["sentiment"]), content=str(r["content"])))
    except Exception as e:
        print(f"Exception occured:{e}")
    
    # Return only the keys this agent produces; LangGraph merges them into the
    # graph state, leaving every other field intact.
    return {
        "market": MarketData(
            ticker=state.ticker,
            period=state.period,
            interval=state.interval,
            price_csv=price_csv,
        ),
        "news": NewsBundle(
            ticker=state.ticker,
            window_days=min(14, state.horizon_days),
            items=items,
        ),
    }


def sentiment_agent(state: State, config: RunnableConfig):
    """
    Sentiment agent that analyzes news using reflection-enhanced prompting.
    Implements a multi-step process: summarize, critique, refine, conclude.
    """
    llm = get_llm()
    
    if not state.news or not state.news.items:
        # No news data available
        sentiment_summary = SentimentSummary(
            ticker=state.ticker,
            news_items_analyzed=0,
            overall_sentiment="neutral",
            confidence_score=0.0,
            summary="No news data available for analysis.",
            investment_recommendation="Cannot provide recommendation due to lack of news data.",
            key_insights=["No news items found for analysis"],
            methodology="LLM-based reflection-enhanced summarization"
        )
        return {"sentiment": sentiment_summary}
    
    # Prepare news data for analysis
    news_text = []
    for item in state.news.items:
        news_text.append(f"Date: {item.date}\nHeadline: {item.headline}\nContent: {item.content}")
    
    news_content = "\n\n".join(news_text)
    
    # Step 1: Initial Analysis with Reflection-Enhanced Prompting
    analysis_prompt = f"""
    Analyze the following news items for {state.ticker} using reflection-enhanced prompting:

    NEWS DATA:
    {news_content}

    PROCESS:
    1. SUMMARIZE: First, provide a concise summary of each news item and its potential market impact.
    2. CRITIQUE: Evaluate your summary - is it comprehensive? Are you missing key insights? What biases might be present?
    3. REFINE: Based on your critique, improve and refine your analysis.
    4. CONCLUDE: Provide your final assessment.

    Your response should also include:
    - Overall sentiment analysis (bullish/bearish/neutral)
    - Confidence level (0.0 to 1.0)
    - Key insights and reasoning
    - Investment recommendation with clear rationale

    Provide a concise summary along with an informed recommendation on whether to invest in this stock for the next {state.horizon_days} days.
    """

    # The schema is enforced by a constrained-decoding structuring step
    # (create_react_agent's response_format), not by a prompt marker — so it
    # works the same with GPT-4o and small local models like llama3.1.
    structuring_prompt = (
        "Using the analysis above, fill EVERY field of the structured sentiment "
        "summary. key_insights must be a list of plain-sentence strings (not "
        "objects). confidence_score must be between 0.0 and 1.0."
    )
    sentiment_agent = create_react_agent(
        llm, [], prompt=SENTIMENT_SYSTEM,
        response_format=(structuring_prompt, SentimentExtract),
    )

    # Execute the agent
    extract = None
    try:
        result = sentiment_agent.invoke({"messages": [("human", analysis_prompt)]})
        extract = result.get("structured_response")
    except Exception as e:
        print(f"Agent execution error: {e}")

    if extract is None:
        print("🚫 No structured sentiment produced; using empty summary.")
        sentiment_summary = SentimentSummary()
    else:
        # Map the model's extract into the storage schema, adding system-owned
        # fields rather than trusting the model for them.
        sentiment_summary = SentimentSummary(
            ticker=state.ticker,
            news_items_analyzed=len(state.news.items),
            overall_sentiment=extract.overall_sentiment,
            confidence_score=extract.confidence_score,
            summary=extract.summary,
            investment_recommendation=extract.investment_recommendation,
            key_insights=extract.key_insights,
            methodology="LLM-based reflection-enhanced summarization",
        )
    
    return {"sentiment": sentiment_summary}


def fundamental_agent(state: State, config: RunnableConfig):
    """
    Fundamental agent that analyzes 10-K/10-Q data using RAG as a tool.
    """
    end_year = None
    end_month = None
    if state.end_date is not None: 
        end_date = datetime.strptime(state.end_date, "%Y-%m-%d")
        end_year = end_date.year
        end_month = end_date.month

    # Filings come exclusively from SEC EDGAR (no local-PDF seed). The pipeline
    # normally pulls them up front via ensure_filings(); this is a self-sufficiency
    # net for when the agent is invoked directly. ensure_filings() no-ops (no
    # network call) when filings are already stored, and swallows EDGAR/network
    # errors so the agent still runs degraded rather than crashing.
    rag_system = FundamentalRAG()
    available_filings = rag_system.get_available_filings(state.ticker)
    if not available_filings:
        print(f"No filings stored for {state.ticker}; pulling from SEC EDGAR...")
        from utils.edgar_ingest import ensure_filings
        ensure_filings(state.ticker)
        available_filings = rag_system.get_available_filings(state.ticker)
    
    if not available_filings:
        # Create a basic fundamental analysis with no data
        fundamental_analysis = FundamentalAnalysis(
            ticker=state.ticker,
            filing_type="N/A",
            filing_date="N/A",
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            executive_summary=f"No 10-K/10-Q data available for {state.ticker}",
            key_financial_metrics=[],
            business_highlights=[],
            risk_factors=["No data available"],
            competitive_position="Unable to assess due to lack of data",
            growth_prospects="Unable to assess due to lack of data",
            financial_health_score=5.0,
            investment_thesis="Cannot provide investment thesis without fundamental data",
            concerns_and_risks=["No fundamental data available for analysis"],
            methodology="RAG-enhanced 10-K/10-Q document analysis"
        )
    else:
        from_year, from_month, to_year, to_month = period_to_months_range(state.period, end_year, end_month)

        # Create agent with tools. The schema is enforced by a constrained-
        # decoding structuring step (response_format), so it works the same
        # with GPT-4o and small local models like llama3.1.
        llm = get_llm()
        structuring_prompt = (
            "Using the analysis and retrieved filing excerpts above, fill EVERY "
            "field of the structured fundamental analysis. financial_health_score "
            "must be a number between 0 and 10. List fields must contain plain "
            "strings."
        )
        fundamental_agent = create_react_agent(
            llm, [query_10k_documents], prompt=FUNDAMENTAL_SYSTEM,
            response_format=(structuring_prompt, FundamentalExtract),
        )

        query_msg = f"""
    Please analyze {state.ticker} using the 10-K/10-Q documents from month:{from_month}, year:{from_year} to month:{to_month}, year:{to_year}.
    When calling the query_10k_documents tool, pass your queries as a comma-separated string like this: "financial metrics, business segments, risk factors, competitive position, growth prospects, investment thesis, concerns and risks"
    Cover the executive summary, key financial metrics, business highlights, risk factors, competitive position, growth prospects, a 0-10 financial health score, an investment thesis, and concerns/risks.
    """

        # Execute the agent
        extract = None
        try:
            result = fundamental_agent.invoke({"messages": [("human", query_msg)]})
            extract = result.get("structured_response")
        except Exception as e:
            print(f"query_10k_documents invocation failed: {e}")

        filing_info = available_filings[0]
        if extract is None:
            print("🚫 No structured fundamental analysis produced; using empty analysis.")
            fundamental_analysis = FundamentalAnalysis()
        else:
            # Map the model's extract into the storage schema, adding system-
            # owned fields rather than trusting the model for them.
            fundamental_analysis = FundamentalAnalysis(
                ticker=state.ticker,
                filing_type=filing_info.get("filing_type", "10-K"),
                filing_date=filing_info.get("ingestion_date", "Unknown"),
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                executive_summary=extract.executive_summary,
                key_financial_metrics=extract.key_financial_metrics,
                business_highlights=extract.business_highlights,
                risk_factors=extract.risk_factors,
                competitive_position=extract.competitive_position,
                growth_prospects=extract.growth_prospects,
                financial_health_score=extract.financial_health_score,
                investment_thesis=extract.investment_thesis,
                concerns_and_risks=extract.concerns_and_risks,
                methodology="RAG-enhanced 10-K/10-Q document analysis",
            )

    
    return {"fundamental": fundamental_analysis}


def valuation_agent(state: State, config: RunnableConfig):
    """
    Valuation agent that analyzes historical price data to compute valuation metrics
    and trends using computational tools for volatility and return calculations.
    """
    llm = get_llm()
    valuation_metrics = None
    
    # Check if we have market data
    if not state.market or not state.market.price_csv:
        print("⚠️  No market data available for valuation analysis")
        # Create default valuation metrics
        valuation_metrics = ValuationMetrics(
            ticker=state.ticker,
            analysis_period=state.period,
            trading_days=0,
            cumulative_return=0.0,
            annualized_return=0.0,
            daily_volatility=0.0,
            annualized_volatility=0.0,
            price_trend="sideways",
            volatility_regime="medium",
            valuation_insights=["No market data available for analysis"],
            trend_analysis="Cannot analyze trends without price data",
            risk_assessment="Unable to assess risk without market data"
        )
    else:
        # The metrics are pure deterministic math — compute them directly
        # instead of depending on the LLM to choose to call the tool. Small
        # local models often don't, which previously left valuation_metrics
        # unset and crashed the error handler.
        metrics_dict = compute_valuation_metrics(
            state.market.price_csv, state.ticker, state.period)
        valuation_metrics = ValuationMetrics(**metrics_dict)

        # Use the LLM only for narrative trend commentary (best-effort): the
        # computed metrics stand on their own if this call fails.
        analysis_prompt = f"""
        Based on these computed valuation metrics for {state.ticker} over {state.period}:
        {metrics_dict}

        Provide an enhanced trend analysis and investment implications for the
        next {state.horizon_days} days.
        """
        try:
            response = llm.invoke([
                SystemMessage(content=VALUATION_SYSTEM),
                HumanMessage(content=analysis_prompt),
            ])
            valuation_metrics.trend_analysis = (
                response.content if hasattr(response, "content") else str(response))
            print(f"trend_analysis is: {valuation_metrics.trend_analysis}")
        except Exception as e:
            print(f"⚠️  LLM trend commentary unavailable: {e}")
            valuation_metrics.trend_analysis += (
                "\n\nLLM commentary unavailable; using computed metrics only.")
    
    return {"valuation": valuation_metrics}


def risk_agent(state: State, config: RunnableConfig):
    stats = compute_risk.invoke({"price_csv": state.market.price_csv if state.market else ""})
    notes, flags = [], []
    if "error" in stats:
        notes.append(stats["error"])
        stats = {"annual_vol": 0.0, "max_drawdown": 0.0, "daily_var_95": 0.0, "sharpe_like": None}
        flags.append("DATA_QUALITY")
    if stats["annual_vol"] and stats["annual_vol"] > 0.45: flags.append("HIGH_VOLATILITY")
    if stats["max_drawdown"] and stats["max_drawdown"] < -0.25: flags.append("DEEP_DRAWDOWN")

    metrics = RiskMetrics(
        ticker=state.ticker,
        horizon_days=state.horizon_days,
        annual_vol=round(float(stats["annual_vol"]),4),
        max_drawdown=round(float(stats["max_drawdown"]),4),
        daily_var_95=round(float(stats["daily_var_95"]),4),
        sharpe_like=(None if stats.get("sharpe_like") is None else round(float(stats["sharpe_like"]),3)),
        notes=notes,
        risk_flags=sorted(set(flags)),
    )
    
    return {"metrics": metrics}


def writer_agent(state: State, config: RunnableConfig):
    # llm = get_llm()
    
    # Build sentiment section
    sentiment_section = ""
    if state.sentiment:
        sentiment_section = f"""
## Sentiment Analysis
- **Overall Sentiment**: {state.sentiment.overall_sentiment.title()}
- **Confidence Score**: {state.sentiment.confidence_score:.1%}
- **News Items Analyzed**: {state.sentiment.news_items_analyzed}
- **Investment Recommendation**: {state.sentiment.investment_recommendation}

### Key Insights
{chr(10).join(f"- {insight}" for insight in state.sentiment.key_insights)}

### Summary
{state.sentiment.summary}
"""
    else:
        sentiment_section = """
## Sentiment Analysis
No sentiment analysis available - insufficient news data.
"""

    # Build valuation section
    valuation_section = ""
    if state.valuation:
        valuation_section = f"""
## Valuation Analysis
- **Analysis Period**: {state.valuation.analysis_period}
- **Trading Days Analyzed**: {state.valuation.trading_days}
- **Cumulative Return**: {state.valuation.cumulative_return:.4f} ({state.valuation.cumulative_return:.2%})
- **Annualized Return**: {state.valuation.annualized_return:.4f} ({state.valuation.annualized_return:.2%})
- **Daily Volatility**: {state.valuation.daily_volatility:.6f}
- **Annualized Volatility**: {state.valuation.annualized_volatility:.4f} ({state.valuation.annualized_volatility:.2%})
- **Price Trend**: {state.valuation.price_trend.title()}
- **Volatility Regime**: {state.valuation.volatility_regime.title()}

### Valuation Insights
{chr(10).join(f"- {insight}" for insight in state.valuation.valuation_insights)}

### Trend Analysis
{state.valuation.trend_analysis}

### Risk Assessment
{state.valuation.risk_assessment}
"""
    else:
        valuation_section = """
## Valuation Analysis
No valuation analysis available - insufficient market data.
"""

    # Build fundamental analysis section
    fundamental_section = ""
    if state.fundamental:
        fundamental_section = f"""
## Fundamental Analysis (10-K/10-Q Based)
- **Filing Type**: {state.fundamental.filing_type}
- **Filing Date**: {state.fundamental.filing_date}
- **Financial Health Score**: {state.fundamental.financial_health_score:.1f}/10.0

### Executive Summary
{state.fundamental.executive_summary}

### Business Highlights
{chr(10).join(f"- {highlight}" for highlight in state.fundamental.business_highlights)}

### Risk Factors
{chr(10).join(f"- {risk}" for risk in state.fundamental.risk_factors)}

### Competitive Position
{state.fundamental.competitive_position}

### Growth Prospects
{state.fundamental.growth_prospects}

### Investment Thesis
{state.fundamental.investment_thesis}

### Concerns and Risks
{chr(10).join(f"- {concern}" for concern in state.fundamental.concerns_and_risks)}
"""
    else:
        fundamental_section = """
## Fundamental Analysis
No fundamental analysis available - no 10-K/10-Q data found.
"""

    current_ts = datetime.utcnow().strftime('%b %d, %Y at %I:%M %p UTC').replace(' 0', ' ').lstrip('0')

    metrics_obj = state.metrics
    valuation_obj = state.valuation
    sentiment_obj = state.sentiment
    fundamental_obj = state.fundamental

    if metrics_obj and metrics_obj.risk_flags:
        risk_summary = "Risk watchlist flagged: " + ", ".join(metrics_obj.risk_flags)
    else:
        risk_summary = "Key risk indicators are within typical ranges."

    volatility_label = "unknown"
    if metrics_obj and metrics_obj.annual_vol is not None:
        volatility_label = "low" if metrics_obj.annual_vol < 0.15 else "elevated"

    if valuation_obj:
        valuation_trend = valuation_obj.price_trend
        valuation_tone = (
            "Trend shows constructive momentum."
            if valuation_obj.price_trend != "downward"
            else "Trend pressure leans negative; review positioning."
        )
    else:
        valuation_trend = "Price momentum unclear"
        valuation_tone = "Insufficient data to characterise trend."

    if sentiment_obj:
        if sentiment_obj.overall_sentiment == "bullish":
            sentiment_tone = "Sentiment skews bullish."
        elif sentiment_obj.overall_sentiment == "bearish":
            sentiment_tone = "Sentiment caution persists."
        else:
            sentiment_tone = "Sentiment mix appears balanced."
    else:
        sentiment_tone = "Sentiment data limited."

    if fundamental_obj:
        if fundamental_obj.financial_health_score >= 7:
            fundamental_tone = (
                f"Financial health score {fundamental_obj.financial_health_score:.1f}/10 reflects solid fundamentals."
            )
        else:
            fundamental_tone = (
                f"Financial health score {fundamental_obj.financial_health_score:.1f}/10 highlights areas to monitor."
            )
    else:
        fundamental_tone = "Fundamental details unavailable."

    if sentiment_obj and sentiment_obj.overall_sentiment == "bullish":
        bottom_line = "Position looks resilient but stay selective."
    else:
        bottom_line = "Maintain watchful posture and reassess catalysts."

    news_entries = []
    if state.news and state.news.items:
        for item in state.news.items:
            news_entries.append(f"- {item.date}: {item.headline} [{item.sentiment}]")
    news_section = "\n".join(news_entries) if news_entries else "No recent headlines captured."

    notes_text = (
        ", ".join(metrics_obj.notes) if metrics_obj and metrics_obj.notes else "No unusual observations logged."
    )

    md = f"""# {state.ticker} Investment Brief
**Last refreshed:** {current_ts}

---

## Snapshot
| Lens | Takeaway |
| --- | --- |
| Price action | {valuation_tone} |
| Risk posture | {risk_summary} |
| Sentiment pulse | {sentiment_tone} |
| Fundamentals | {fundamental_tone} |

---

## Executive Dashboard
- **What stands out:** {valuation_trend} trend paired with {volatility_label} volatility.
- **Primary question:** Is the current narrative supportive of further upside given risk levels?
- **Bottom line:** {bottom_line}

---

## Decision Lens
### 1. Market Structure
- Period assessed: {state.period}
- Horizon in focus: {state.horizon_days} days
- Storyline: {valuation_tone}

### 2. Risk Review
- Default read: {risk_summary}
- Notes: {notes_text}

### 3. Fundamental Pulse
- Filing types covered: {fundamental_obj.filing_type if fundamental_obj else 'N/A'}
- Executive summary: {fundamental_obj.executive_summary if fundamental_obj else 'Insufficient filing coverage.'}
- Thesis highlights: {fundamental_obj.investment_thesis if fundamental_obj else 'No official thesis compiled.'}

---

## Supporting Detail
### Sentiment & Narrative
{sentiment_section}

### Valuation Context
{valuation_section}

### Fundamental Highlights
{fundamental_section}

### Recent Headlines
{news_section}

---

## Methodology Notes
- Sequenced multi-agent workflow: market data → sentiment → valuation → fundamentals → risk → writer.
- Market data via yfinance; fallback narratives when data unavailable.
- News ingestion uses Polygon (if configured) otherwise synthetic briefs; sentiment generated via LLM feedback loop.
- Fundamental insights extracted through RAG over ingested 10-K/10-Q filings.
- Risk metrics computed from returns-based analytics (volatility, drawdown, VaR).

---

{"## Investment Final Recommendation\n" + state.debate.consensus_summary if state.debate else ""}
"""
    # _ = llm.invoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content="draft report")])  # tracing
    
    # Create key findings including valuation and fundamental analysis
    key_findings = [
        "Market action reviewed across price, risk, fundamental, and sentiment dimensions.",
        "Composite takeaway blends model-driven metrics with narrative context."
    ]
    if state.valuation:
        key_findings.append(
            f"Market tone: {state.valuation.price_trend} trend amid {state.valuation.volatility_regime} volatility regime."
        )
    if state.sentiment:
        key_findings.append(
            f"Sentiment skew: {state.sentiment.overall_sentiment} with confidence {state.sentiment.confidence_score:.0%}."
        )
    if state.fundamental:
        key_findings.append(
            f"Fundamental signal: Financial health score {state.fundamental.financial_health_score:.1f}/10 from latest filings."
        )

    report = RiskReport(
        ticker=state.ticker,
        as_of=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        summary=f"Comprehensive risk and valuation analysis for {state.ticker} with sentiment analysis.",
        key_findings=key_findings,
        metrics_table={
            "annual_vol": state.metrics.annual_vol,
            "max_drawdown": state.metrics.max_drawdown,
            "daily_var_95": state.metrics.daily_var_95,
            "sharpe_like": state.metrics.sharpe_like,
        },
        risk_flags=state.metrics.risk_flags,
        methodology=(
            "Gaussian VaR; log returns; daily OHLC from yfinance; "
            "Valuation with 252-day annualization; LLM sentiment analysis."
        ),
        markdown_report=md,
    )
    
    return {"report": report}

# Debate
def debate_manager(state: State):
    """Debate Manager control debates"""
    current_agent = 'debate_manager'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()

    DEBATE_MANAGER_SYSTEM = f"""
                            You are the Debate Manager coordinating three agents: Fundamental, Sentiment, and Valuation.
                            Your task:
                            - Carefully read the specialized agents arguments.
                            - Analyze agreements, disagreements, and the overall tone.
                            - Identify key evidence and logic from each.
                            - Synthesize these viewpoints into **one concise, reasoned conclusion**.
                            - The conclusion must be objective, actionable, and justified.

                            Your output is final conclusion and must follow this concern:
                            - Highlight points of agreement or conflict.
                            - Note which arguments are stronger or better supported.
                            - Provide a single, coherent conclusion that integrates all perspectives.
                            - If uncertainty remains, explain it clearly.
                            - Be balanced, analytical, and clear about judgement to invest in the stock within the next {state.horizon_days} days.

                            Output format:
                            - Output is consise summary
                            - Based on your summary, give a recommendation to 'buy', 'hold', or 'sell'.
                            """
    # - If the conclusion is not converged, output 'Output: Sentiment' or 'Output: Fundamental' to continue the debate.
    # - If the conclusion is converged, output with format 'Output: Accepted' to end the debate.
    
    # Initialize debate
    if new_state.debate.agent_turn_count is None:
        new_state.debate.agent_turn_count = {agent:0 for agent in new_state.debate.agent_list}

    counts = new_state.debate.agent_turn_count.values()
    counts_list = list(counts)

    if len(set(counts_list)) == 1 and all(c > 0 for c in counts_list):
        print(f'Debate Manager turn-{next(iter(counts))-1}')
        messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
        
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                messages.append(HumanMessage(content=f"This is based on {agent.title()} Agent arguments: \"\"{latest}\"\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        response = llm.invoke(messages)
    
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
        prev_consensus_summary = new_state.debate.consensus_summary
        new_state.debate.consensus_summary = response_text
        new_state.debate.agent_arguments[current_agent].append(response_text)
        # print(f"manager conclusions: {response_text}")

        if all(c > 1 for c in counts_list):
            messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
            messages.append(HumanMessage(content=prev_consensus_summary))
            messages.append(HumanMessage(content=response_text))
            messages.append(HumanMessage(content="""Based on both Consensus Summaries, do this action:
                                                - Compare both Consensus Summaries, and get the final recommendation
                                                - Say in this format 'First: {your first recommendation}, Second: {your second recommendation}, Action: {DEBATE or END}'
                                                - If both Consensus Summaries have similar recommendations, just fill Action with 'END', else than that say 'DEBATE'.
                                                    """))
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            new_state.debate.terminated = 'END' if response_text.__contains__("END") else ''
    
    #- If you only have one consensus summary, output 'Output: Sentiment' or 'Output: Fundamental' to clarify.
    # - If you have done two concusion before, and your conclusion for positions either 'buy','hold', or 'sell' is changed from your previous conclusion, output 'Output: Sentiment' or 'Output: Fundamental' to continue the debate.
    # - If you have done concusion before, and your conclusion for positions either 'buy','hold', or 'sell' is not changed from your previous conclusion, output with format 'Output: Accepted' to end the debate.
    # print(new_state.debate.agent_turn_count)

    if all(v == new_state.debate.agent_max_turn-1 for v in new_state.debate.agent_turn_count.values()):
        # print("ARGS:", new_state.debate.agent_arguments)
        new_state.debate.terminated = "ENDMAX"

    if new_state.debate.terminated == "END" or new_state.debate.terminated == "ENDMAX":
        messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
        messages.append(HumanMessage(content=f"""
                                        Refine the Final Consensus into a clear, concise summary suitable for a report.
                                        - Make it brief and precise.
                                        - Do not mention agents or the debate process.
                                        - Output only the final polished summary, with no headings or extra text.

                                        Final Consensus:
                                        "{new_state.debate.consensus_summary}"
                                     """))
        llm.temperature = 0.4
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        new_state.debate.consensus_summary = response_text
        new_state.debate.agent_arguments[current_agent].append(response_text)
        print(f"Final Consensus Summary:\n{new_state.debate.consensus_summary}")

    return new_state

def route_debate(state: State):
    if len(set(state.debate.agent_turn_count.values())) == 1:
        print(f"\n\nDebate routing turn:{min(state.debate.agent_turn_count.values())} ============================")
    if state.debate.terminated == "ENDMAX" or state.debate.terminated == "END":
        return "END"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "fundamental":
        return "Fundamental"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "sentiment":
        return "Sentiment"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "valuation":
        return "Valuation"

def debate_fundamental_agent(state: State):
    current_agent = 'fundamental'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    DEBATE_SYSTEM = f"""
                        You are the Fundamental Analysis Agent.

                        Specialization:
                        Focus on financial performance, balance sheet strength, profitability, debt, valuation, and long-term business potential.

                        Report:
                        \"\"\"{state.fundamental.executive_summary}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                        """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other Agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section.\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other Agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state

def debate_sentiment_agent(state: State):
    current_agent = 'sentiment'
    llm = get_llm(temperature=0.6)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    DEBATE_SYSTEM = f"""
                        You are the Sentiment Analysis Agent.

                        Specialization:
                        Analyze the tone, emotional language, and implied investor sentiment in a report.
                        Identify whether the sentiment is optimistic, neutral, or negative, and explain why.

                        Report:
                        \"\"\"{state.sentiment.summary}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        # initial_analysis_prompt = f"""
        #                             Summarize the overall sentiment:
        #                             - Describe the tone (positive, neutral, or negative)
        #                             - Mention emotional or linguistic indicators of this tone
        #                             - Explain how investors might feel after reading it
        #                                 """
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                  """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section..\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")
    
        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state

def debate_valuation_agent(state: State):
    current_agent = 'valuation'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    DEBATE_SYSTEM = f"""
                        You are the Valuation Analysis Agent.

                        Specialization:
                        Analyze the valuation trends of a given asset or portfolio over an extended time horizon. 
                        To complete the task, you must analyze the historical valuation data of the asset or portfolio provided, identify trends and patterns in valuation metrics over time, and interpret the implications of these trends for investors or stakeholders.

                        Focus your analysis on:
                        1. Price trend analysis (upward, downward, sideways movement)
                        2. Volatility regime assessment (low, medium, high volatility periods)
                        3. Risk-return profile evaluation
                        4. Investment implications and outlook
                        5. Key patterns and inflection points in the data

                        Report:
                        \"\"\"{state.valuation.trend_analysis}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                        """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other Agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section.\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other Agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state