#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama.llms import OllamaLLM

from langgraph.graph import StateGraph, START, END


# LangGraph State Schema
class AnalysisState(TypedDict):
    """State schema for the financial analysis workflow"""
    ticker: str
    primary_analysis: Optional[str]
    critique_report: Optional[str]
    risk_assessment: Optional[str]
    analysis_validation: Optional[str]
    final_recommendation: Optional[str]
    iteration_count: int
    needs_improvement: bool
    quality_grade: Optional[str]
    agent_messages: List[Dict[str, str]]  # Custom message tracking


class VerboseCallback(BaseCallbackHandler):
    """Callback to stream tokens and show tool I/O."""
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print("\n" + "=" * 50)
        print("🧠 LLM REASONING STREAM:")
        print("=" * 50)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        print("\n" + "=" * 50)
        print("📝 LLM FINISHED GENERATION")
        print("=" * 50)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "Unknown")
        print(f"\n🔧 Executing Tool: {tool_name}")
        print(f"📥 Input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        snippet = output if len(output) <= 200 else output[:200] + "..."
        print(f"📤 Tool Result: {snippet}")


def extract_ticker(query: str) -> Optional[str]:
    matches = re.findall(r'\b[A-Z]{1,5}(?:\d*)\b', query)
    return matches[0] if matches else None


def get_latest_news(stock: yf.Ticker, max_items: int = 10) -> List[Dict[str, str]]:
    try:
        raw = stock.news or []
        out = []
        for item in raw[:max_items]:
            title = item.get("title", "No title")
            publisher = item.get("provider", {}).get("displayName", "Unknown")
            pub = item.get("pubDate")
            try:
                date = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ") if pub else datetime.now()
            except ValueError:
                date = datetime.now()
            summary = item.get("summary", "No summary available")
            if title != "No title":
                out.append({
                    "title": title,
                    "publisher": publisher,
                    "published": date.strftime("%Y-%m-%d"),
                    "summary": summary,
                })
        return out
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def financial_data_analysis_tool(ticker: str) -> str:
    if ":" in ticker:
        ticker = ticker.split(":")[-1].strip()
    ticker = ticker.strip("'\"").upper()
    
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    if info.get("regularMarketPrice") is None:
        return f"Invalid ticker: {ticker}"
    hist = stock.history(period="3mo")
    if hist.empty:
        return f"No data available for {ticker}"
    current = hist["Close"].iloc[-1]
    past = hist["Close"].iloc[0]
    pct = (current - past) / past * 100
    mcap = info.get("marketCap", 0)
    mcap_str = f"${mcap/1e9:.1f}B" if mcap >= 1e9 else f"${mcap/1e6:.1f}M"
    return (
        f"Financial data for {ticker}:\n"
        f"• Current Price: ${current:.2f}\n"
        f"• 3-Month Change: {pct:.1f}%\n"
        f"• Market Cap: {mcap_str}\n"
        f"• Sector: {info.get('sector','N/A')}\n"
        f"• P/E Ratio: {info.get('trailingPE','N/A')}"
    )


def search_tool(query: str) -> str:
    ticker = extract_ticker(query)
    if not ticker:
        return "Could not find a valid ticker symbol."
    stock = yf.Ticker(ticker)
    news = get_latest_news(stock, max_items=3)
    if not news:
        return f"No recent news found for {ticker}."
    lines = [
        f"• {n['title']} ({n['published']}) – {n['summary'][:150]}..."
        for n in news
    ]
    return "Recent news for {}:\n{}".format(ticker, "\n".join(lines))


def sentiment_analysis_tool(text: str) -> str:
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text[:500])
    c = scores["compound"]
    if c >= 0.05:
        sentiment = "Positive (Bullish)"
    elif c <= -0.05:
        sentiment = "Negative (Bearish)"
    else:
        sentiment = "Neutral"
    return f"Sentiment: {sentiment} (Score: {c:.2f})"


def risk_assessment_tool(ticker: str) -> str:
    if ":" in ticker:
        ticker = ticker.split(":")[-1].strip()
    ticker = ticker.strip("'\"").upper()
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    hist = stock.history(period="1y")
    
    if hist.empty:
        return f"No historical data available for risk assessment of {ticker}"
    
    daily_returns = hist['Close'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100
    
    beta = info.get('beta', 'N/A')
    debt_to_equity = info.get('debtToEquity', 'N/A')
    current_ratio = info.get('currentRatio', 'N/A')
    
    risk_level = "Medium"
    if isinstance(beta, (int, float)) and beta > 1.5:
        risk_level = "High"
    elif isinstance(beta, (int, float)) and beta < 0.8:
        risk_level = "Low"
    
    return (
        f"Risk Assessment for {ticker}:\n"
        f"• Volatility: {volatility:.1f}% annually\n"
        f"• Beta: {beta} (market sensitivity)\n"
        f"• Debt-to-Equity: {debt_to_equity}\n"
        f"• Current Ratio: {current_ratio}\n"
        f"• Overall Risk Level: {risk_level}"
    )


def validate_analysis_tool(analysis_text: str) -> str:
    required_elements = [
        ("price", ["price", "trading at", "$"]),
        ("trend", ["change", "%", "up", "down", "growth"]),
        ("valuation", ["p/e", "pe", "ratio", "multiple"]),
        ("recommendation", ["buy", "sell", "hold"]),
        ("risk", ["risk", "volatile", "beta"])
    ]
    
    missing_elements = []
    analysis_lower = analysis_text.lower()
    
    for element, keywords in required_elements:
        if not any(keyword in analysis_lower for keyword in keywords):
            missing_elements.append(element)
    
    reasoning_indicators = ["because", "due to", "given", "considering", "based on"]
    has_reasoning = any(indicator in analysis_lower for indicator in reasoning_indicators)
    
    validation_result = "Analysis Validation:\n"
    if missing_elements:
        validation_result += f"⚠️  Missing elements: {', '.join(missing_elements)}\n"
    else:
        validation_result += "✅ All key elements present\n"
    
    if has_reasoning:
        validation_result += "✅ Reasoning provided\n"
    else:
        validation_result += "⚠️  Lacks clear reasoning\n"
    
    if "buy" in analysis_lower or "sell" in analysis_lower or "hold" in analysis_lower:
        validation_result += "✅ Clear recommendation given\n"
    else:
        validation_result += "❌ No clear recommendation\n"
    
    return validation_result


# Define tools
primary_tools = [
    Tool(name="financial_data_analysis", func=financial_data_analysis_tool,
         description="Get financial metrics for a stock ticker. Input: ticker symbol only (e.g., AAPL)"),
    Tool(name="search", func=search_tool,
         description="Search for recent news about a stock ticker. Input: ticker symbol only (e.g., AAPL)"),
    Tool(name="sentiment_analysis", func=sentiment_analysis_tool,
         description="Analyze sentiment of news text. Input: news text to analyze"),
]

critique_tools = [
    Tool(name="risk_assessment", func=risk_assessment_tool,
         description="Assess risk factors including volatility, beta, and financial ratios. Input: ticker symbol only (e.g., AAPL)"),
    Tool(name="validate_analysis", func=validate_analysis_tool,
         description="Validate completeness and quality of financial analysis. Input: analysis text to validate"),
]


def create_llm():
    """Create and configure the LLM"""
    return OllamaLLM(
        model="mistral-small3.1:latest",
        base_url="http://localhost:11434",
        temperature=0.1,
    )


# LangGraph Node Functions
def primary_analysis_node(state: AnalysisState) -> AnalysisState:
    """Primary financial analysis node"""
    print(f"\n🎯 PRIMARY ANALYSIS NODE - Analyzing {state['ticker']}")
    print("=" * 60)
    
    llm = create_llm()
    
    react_prompt = PromptTemplate.from_template("""
You are a financial analysis agent that MUST follow the ReAct format exactly.

Available tools:
{tools}

Tool names: {tool_names}

MANDATORY FORMAT:
Question: [the input question]
Thought: [your reasoning]
Action: [one of: {tool_names}]
Action Input: [just the ticker symbol like "AAPL" or text for sentiment_analysis]
Observation: [tool output]
… (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: [analysis + BUY/SELL/HOLD]

IMPORTANT: For Action Input, use ONLY the ticker symbol (e.g., "AAPL") for financial_data_analysis, search, and risk_assessment tools.

Question: Analyze {ticker} stock and provide an investment recommendation.
Thought:{agent_scratchpad}
""")

    agent = create_react_agent(llm=llm, tools=primary_tools, prompt=react_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=primary_tools,
        verbose=True,
        callbacks=[VerboseCallback()],
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=8,
    )
    
    result = executor.invoke({
        "input": f"Analyze {state['ticker']} stock and provide investment recommendation",
        "ticker": state['ticker']
    })
    
    state["primary_analysis"] = result["output"]
    state["agent_messages"].append({
        "agent": "primary_analyst", 
        "content": result["output"]
    })
    
    return state


def critique_analysis_node(state: AnalysisState) -> AnalysisState:
    """Critique and validation node"""
    print(f"\n🔍 CRITIQUE ANALYSIS NODE - Reviewing {state['ticker']}")
    print("=" * 60)
    
    llm = create_llm()
    
    critique_prompt = PromptTemplate.from_template("""
You are a senior financial analyst tasked with critiquing investment analysis.

Available tools:
{tools}

Tool names: {tool_names}

MANDATORY FORMAT:
Question: [the input question]
Thought: [your reasoning about what to critique]
Action: [one of: {tool_names}]
Action Input: [input for the tool]
Observation: [tool output]
… (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide my critique
Final Answer: [detailed critique with improvements and final grade A-F]

Your role is to:
1. Assess risk factors that may have been overlooked
2. Validate the analysis completeness and reasoning quality
3. Check if the recommendation aligns with the data presented
4. Identify any potential biases or missing considerations
5. Provide an overall quality grade (A-F) and specific improvements

Question: Critique this financial analysis for {ticker}: {analysis}
Thought:{agent_scratchpad}
""")

    agent = create_react_agent(llm=llm, tools=critique_tools, prompt=critique_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=critique_tools,
        verbose=True,
        callbacks=[VerboseCallback()],
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=6,
    )

    result = executor.invoke({
        "input": f"Critique this financial analysis for {state['ticker']}: {state['primary_analysis']}",
        "ticker": state['ticker'],
        "analysis": state['primary_analysis']
    })
    
    state["critique_report"] = result["output"]
    state["agent_messages"].append({
        "agent": "critique_analyst", 
        "content": result["output"]
    })
    
    # Extract quality grade for routing decisions
    critique_lower = result["output"].lower()
    if "grade a" in critique_lower or "grade: a" in critique_lower:
        state["quality_grade"] = "A"
        state["needs_improvement"] = False
    elif "grade b" in critique_lower or "grade: b" in critique_lower:
        state["quality_grade"] = "B"
        state["needs_improvement"] = False
    else:
        state["quality_grade"] = "C-F"
        state["needs_improvement"] = True
    
    return state


def improvement_node(state: AnalysisState) -> AnalysisState:
    """Node to improve analysis based on critique"""
    print(f"\n🔧 IMPROVEMENT NODE - Enhancing analysis for {state['ticker']}")
    print("=" * 60)
    
    llm = create_llm()
    
    improvement_prompt = PromptTemplate.from_template("""
You are a financial analyst tasked with improving your analysis based on expert critique.

Available tools:
{tools}

Tool names: {tool_names}

Previous Analysis: {previous_analysis}
Expert Critique: {critique}

Your task: Address the critique points and provide an improved analysis.

MANDATORY FORMAT:
Question: [the input question]
Thought: [your reasoning about improvements needed]
Action: [one of: {tool_names}]
Action Input: [input for the tool]
Observation: [tool output]
… (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information for an improved analysis
Final Answer: [improved analysis addressing critique points + BUY/SELL/HOLD]

Question: Improve the financial analysis for {ticker} based on the expert critique.
Thought:{agent_scratchpad}
""")

    agent = create_react_agent(llm=llm, tools=primary_tools, prompt=improvement_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=primary_tools,
        verbose=True,
        callbacks=[VerboseCallback()],
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=8,
    )

    result = executor.invoke({
        "input": f"Improve the financial analysis for {state['ticker']} based on expert critique",
        "ticker": state['ticker'],
        "previous_analysis": state['primary_analysis'],
        "critique": state['critique_report']
    })
    
    state["primary_analysis"] = result["output"]  # Update with improved analysis
    state["iteration_count"] += 1
    state["agent_messages"].append({
        "agent": "improvement_analyst", 
        "content": result["output"]
    })
    
    return state


def final_synthesis_node(state: AnalysisState) -> AnalysisState:
    """Final node to synthesize all findings"""
    print(f"\n📋 FINAL SYNTHESIS NODE - Consolidating results for {state['ticker']}")
    print("=" * 60)
    
    final_recommendation = f"""
COMPREHENSIVE FINANCIAL ANALYSIS REPORT
=====================================
Ticker: {state['ticker']}
Quality Grade: {state.get('quality_grade', 'N/A')}
Analysis Iterations: {state['iteration_count']}

PRIMARY ANALYSIS:
{state['primary_analysis']}

EXPERT CRITIQUE:
{state['critique_report']}

INVESTMENT DECISION SUMMARY:
The analysis has been validated through expert critique. 
Consider both the primary analysis and expert feedback when making investment decisions.
Quality Grade: {state.get('quality_grade', 'Pending')}
"""
    
    state["final_recommendation"] = final_recommendation
    state["agent_messages"].append({
        "agent": "synthesizer", 
        "content": final_recommendation
    })
    
    return state


def should_improve(state: AnalysisState) -> str:
    """Conditional routing: decide if analysis needs improvement"""
    max_iterations = 2
    
    if state["iteration_count"] >= max_iterations:
        return "synthesize"
    
    if state.get("needs_improvement", False):
        return "improve"
    else:
        return "synthesize"


def create_financial_analysis_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("primary_analysis", primary_analysis_node)
    workflow.add_node("critique_analysis", critique_analysis_node)
    workflow.add_node("improvement", improvement_node)
    workflow.add_node("final_synthesis", final_synthesis_node)
    
    # Define the flow
    workflow.add_edge(START, "primary_analysis")
    workflow.add_edge("primary_analysis", "critique_analysis")
    
    # Conditional routing based on critique
    workflow.add_conditional_edges(
        "critique_analysis",
        should_improve,
        {
            "improve": "improvement",
            "synthesize": "final_synthesis"
        }
    )
    
    # After improvement, go back to critique (creates the cycle)
    workflow.add_edge("improvement", "critique_analysis")
    workflow.add_edge("final_synthesis", END)
    
    return workflow.compile()


def run_langgraph_analysis(ticker: str) -> Dict[str, Any]:
    """Run the LangGraph multi-agent analysis"""
    print(f"\n🎭 LANGGRAPH MULTI-AGENT ANALYSIS for {ticker.upper()}")
    print("=" * 80)
    
    # Initialize state
    initial_state = AnalysisState(
        ticker=ticker.upper(),
        primary_analysis=None,
        critique_report=None,
        risk_assessment=None,
        analysis_validation=None,
        final_recommendation=None,
        iteration_count=0,
        needs_improvement=False,
        quality_grade=None,
        agent_messages=[]
    )
    
    # Create and run the graph
    graph = create_financial_analysis_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph financial analysis")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    
    try:
        result = run_langgraph_analysis(ticker)
        
        print("\n\n" + "="*80)
        print("🎯 LANGGRAPH ANALYSIS COMPLETE")
        print("="*80)
        print(result["final_recommendation"])
        
        if args.debug:
            print("\n\n" + "="*80)
            print("🔍 DEBUG: Agent Message History")
            print("="*80)
            for i, msg in enumerate(result["agent_messages"], 1):
                print(f"\n{i}. {msg['agent'].upper()}:")
                print("-" * 40)  
                print(msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content'])
                
    except Exception as e:
        print(f"\n❌ Error running LangGraph analysis: {e}")
        raise