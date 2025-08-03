#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama.llms import OllamaLLM


class VerboseCallback(BaseCallbackHandler):
    """Callback to stream tokens (including Thought:) and show tool I/O."""
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print("\n" + "=" * 50)
        print("🧠 LLM REASONING STREAM:")
        print("=" * 50)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        # Stream each token (captures Thought, Action, etc.)
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


def financial_data_analysis_tool(ticker: str) -> str:
    # Handle various input formats like "ticker: AAPL" or just "AAPL"
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
    """Assess risk factors for the given stock"""
    # Handle various input formats like "ticker: AAPL" or just "AAPL"
    if ":" in ticker:
        ticker = ticker.split(":")[-1].strip()
    ticker = ticker.strip("'\"").upper()
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    hist = stock.history(period="1y")
    
    if hist.empty:
        return f"No historical data available for risk assessment of {ticker}"
    
    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist['Close'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
    
    # Beta (systematic risk)
    beta = info.get('beta', 'N/A')
    
    # Debt-to-equity ratio
    debt_to_equity = info.get('debtToEquity', 'N/A')
    
    # Current ratio (liquidity)
    current_ratio = info.get('currentRatio', 'N/A')
    
    # Risk assessment
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
    """Validate the completeness and quality of financial analysis"""
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
    
    # Check for reasoning quality
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
    
    # Check recommendation strength
    if "buy" in analysis_lower or "sell" in analysis_lower or "hold" in analysis_lower:
        validation_result += "✅ Clear recommendation given\n"
    else:
        validation_result += "❌ No clear recommendation\n"
    
    return validation_result


tools = [
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


def create_proper_react_agent(ticker: str) -> Dict[str, Any]:
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

    llm = OllamaLLM(
        model="mistral-small3.1:latest",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[VerboseCallback()],
        handle_parsing_errors=True,         # retry on parse failures
        return_intermediate_steps=True,
        max_iterations=8,
    )

    print(f"\n🚀 Running ReAct agent on {ticker}\n" + ("=" * 60))
    return executor.invoke({
        "input": f"Analyze {ticker} stock and provide investment recommendation",
        "ticker": ticker
    })


def create_critique_agent(ticker: str) -> Dict[str, Any]:
    """Create a critique agent to review and validate analysis"""
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

    llm = OllamaLLM(
        model="mistral-small3.1:latest",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

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

    return executor


def run_multi_agent_analysis(ticker: str) -> Dict[str, Any]:
    """Run both analysis and critique agents in sequence"""
    print(f"\n🎯 PHASE 1: Initial Financial Analysis for {ticker}")
    print("=" * 60)
    
    # Run primary analysis agent
    primary_result = create_proper_react_agent(ticker)
    primary_analysis = primary_result["output"]
    
    print(f"\n🔍 PHASE 2: Critique and Validation for {ticker}")
    print("=" * 60)
    
    # Run critique agent
    critique_executor = create_critique_agent(ticker)
    critique_result = critique_executor.invoke({
        "input": f"Critique this financial analysis for {ticker}: {primary_analysis}",
        "ticker": ticker,
        "analysis": primary_analysis
    })
    
    # Combine results
    final_result = {
        "ticker": ticker,
        "primary_analysis": primary_analysis,
        "critique": critique_result["output"],
        "primary_steps": primary_result.get("intermediate_steps", []),
        "critique_steps": critique_result.get("intermediate_steps", [])
    }
    
    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run financial ReAct agent with critique")
    parser.add_argument("--single", action="store_true", help="Run single agent only (no critique)")
    parser.add_argument("--test", action="store_true", help="Test ReAct format first")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    if args.single:
        # Run single agent analysis
        print(f"🔥 Single Agent Analysis for {ticker}")
        result = create_proper_react_agent(ticker)
        
        print("\n\n💡 Intermediate Steps:")
        print(json.dumps(result.get("intermediate_steps", {}), indent=2, default=str))
        
        print("\n\n📋 Final Answer:")
        print(result["output"])
    else:
        # Run multi-agent analysis with critique
        print(f"🎭 Multi-Agent Analysis with Critique for {ticker}")
        result = run_multi_agent_analysis(ticker)
        
        print("\n\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📈 PRIMARY ANALYSIS for {result['ticker']}:")
        print("-" * 50)
        print(result['primary_analysis'])
        
        print(f"\n🔍 EXPERT CRITIQUE:")
        print("-" * 50)
        print(result['critique'])
        
        print(f"\n💼 INVESTMENT DECISION SUMMARY:")
        print("-" * 50)
        print("Consider both the primary analysis and expert critique when making investment decisions.")
        print("The critique highlights potential risks and validates the recommendation quality.")