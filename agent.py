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
    except Exception:
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


tools = [
    Tool(name="financial_data_analysis", func=financial_data_analysis_tool,
         description="Get financial metrics for a stock ticker symbol"),
    Tool(name="search", func=search_tool,
         description="Search for recent news about a stock ticker"),
    Tool(name="sentiment_analysis", func=sentiment_analysis_tool,
         description="Analyze sentiment of news text"),
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
Action Input: [input for the tool]
Observation: [tool output]
… (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: [analysis + BUY/SELL/HOLD]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run financial ReAct agent")
    parser.add_argument("--test", action="store_true", help="Test ReAct format first")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker")
    args = parser.parse_args()

    # (Optional) add a ReAct format self-test here if desired

    result = create_proper_react_agent(args.ticker.upper())

    print("\n\n💡 Intermediate Steps Dump:")
    print(json.dumps(result.get("intermediate_steps", {}), indent=2, default=str))

    print("\n\n📋 Final Answer:")
    print(result["output"])