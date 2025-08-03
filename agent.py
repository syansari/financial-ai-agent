from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from typing import List, Dict, Optional
from langchain_ollama.llms import OllamaLLM
import re


# Define tool functions

def extract_ticker(query: str) -> Optional[str]:
    """Extract ticker symbol from query string."""
    # Match common ticker patterns (1-5 uppercase letters, optionally followed by numbers)
    matches = re.findall(r'\b[A-Z]{1,5}(?:\d*)\b', query)
    return matches[0] if matches else None


def get_latest_news(stock: yf.Ticker, max_items: int = 10) -> List[Dict[str, str]]:
    """
    Get the latest news for a stock with improved error handling.
    """
    try:
        news = stock.news
        print(f"***************Fetched {len(news) if news else 0} news items for {stock.ticker}")

        if not news:
            return []

        formatted_news = []
        for item in news[:max_items]:
            try:
                # Handle the nested structure shown in your debug output
                if isinstance(item, dict):
                    # Check if there's a 'content' key (nested structure)
                    content = item.get('content', item)

                    # Extract title
                    title = content.get('title', 'No title')

                    # Extract publisher from provider
                    provider = content.get('provider', {})
                    publisher = provider.get('displayName', 'Unknown')

                    # Extract link from canonicalUrl or clickThroughUrl
                    canonical_url = content.get('canonicalUrl', {})
                    click_url = content.get('clickThroughUrl', {})
                    link = canonical_url.get('url') or click_url.get('url', '#')

                    # Extract date from pubDate
                    pub_date = content.get('pubDate')
                    if pub_date:
                        try:
                            # Handle ISO format date string
                            date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ')
                        except ValueError:
                            try:
                                # Try alternative format
                                date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%S.%fZ')
                            except ValueError:
                                date = datetime.now()
                    else:
                        date = datetime.now()

                    # Get summary
                    summary = content.get('summary', 'No summary available')

                    if title and title != 'No title':
                        formatted_news.append({
                            'title': title,
                            'publisher': publisher,
                            'link': link,
                            'published': date.strftime('%Y-%m-%d %H:%M:%S'),
                            'summary': summary
                        })
                        print(f"Successfully parsed: {title[:50]}...")

            except Exception as e:
                print(f"Error processing news item: {e}")
                continue

        print(f"Successfully formatted {len(formatted_news)} news items")
        return formatted_news

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def search_tool(query: str) -> str:
    """Search for news about a stock with improved error handling."""
    print(f"Searching for: {query}")

    try:
        ticker = extract_ticker(query)
        if not ticker:
            return "Could not find a valid ticker symbol in the query. Please provide a company name or stock symbol."

        ticker = ticker.strip('"\'')
        print(f"Found ticker: {ticker}")

        stock = yf.Ticker(ticker)

        # Validate ticker exists
        try:
            info = stock.info
            if not info or info.get('regularMarketPrice') is None:
                return f"Invalid ticker symbol: {ticker}"
        except:
            return f"Invalid ticker symbol: {ticker}"

        news = get_latest_news(stock, max_items=5)

        if not news:
            return f"No recent news found for {ticker}."

        news_summary = "\n".join([
            f"- {item['title']}\n  Published: {item['published']} by {item['publisher']}\n  Summary: {item['summary'][:200]}{'...' if len(item['summary']) > 200 else ''}\n"
            for item in news
        ])
        return f"Recent news for {ticker}:\n{news_summary}"

    except Exception as e:
        return f"Error searching for news: {str(e)}"


def financial_data_analysis_tool(ticker: str) -> str:
    """Analyze financial data for a given ticker with enhanced metrics."""
    ticker = ticker.strip('"\'').upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('regularMarketPrice') is None:
            return f"Invalid ticker symbol or no data available for {ticker}."

        # Get different time periods for analysis
        hist_3m = stock.history(period="3mo", interval="1d")
        hist_1y = stock.history(period="1y", interval="1d")

        if hist_3m.empty:
            return f"No historical price data available for {ticker}."

        current_price = hist_3m['Close'].iloc[-1]

        # Calculate various metrics
        price_3m_ago = hist_3m['Close'].iloc[0]
        price_change_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100

        # Calculate volatility (standard deviation of daily returns)
        daily_returns = hist_3m['Close'].pct_change().dropna()
        volatility = daily_returns.std() * 100

        avg_volume_3m = hist_3m['Volume'].mean()

        # 52-week high/low if available
        if not hist_1y.empty:
            high_52w = hist_1y['High'].max()
            low_52w = hist_1y['Low'].min()
            price_change_1y = ((current_price - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0]) * 100
        else:
            high_52w = hist_3m['High'].max()
            low_52w = hist_3m['Low'].min()
            price_change_1y = "N/A"

        # Format market cap
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap / 1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        analysis = f"""
Financial Analysis for {info.get('longName', ticker)} ({ticker}):

PRICE METRICS:
• Current Price: ${current_price:.2f} {info.get('currency', 'USD')}
• 3-Month Change: {price_change_3m:.2f}%
• 1-Year Change: {price_change_1y}%
• 52-Week High: ${high_52w:.2f}
• 52-Week Low: ${low_52w:.2f}
• Daily Volatility: {volatility:.2f}%

COMPANY INFO:
• Market Cap: {market_cap_str}
• Sector: {info.get('sector', 'N/A')}
• Industry: {info.get('industry', 'N/A')}
• P/E Ratio: {info.get('trailingPE', 'N/A')}
• Dividend Yield: {info.get('dividendYield', 'N/A')}

TRADING METRICS:
• Average Daily Volume (3M): {avg_volume_3m:,.0f}
• Beta: {info.get('beta', 'N/A')}

RECENT PERFORMANCE (Last 5 Days):
{hist_3m[['Open', 'High', 'Low', 'Close', 'Volume']].tail().round(2).to_string()}
"""

        # Add business summary if available and not too long
        business_summary = info.get('longBusinessSummary', '')
        if business_summary and len(business_summary) < 500:
            analysis += f"\n\nBUSINESS DESCRIPTION:\n{business_summary}"

        return analysis

    except Exception as e:
        return f"Error analyzing financial data for {ticker}: {str(e)}"


def sentiment_analysis_tool(text: str) -> str:
    """Analyze sentiment with more detailed output."""
    try:
        analyzer = SentimentIntensityAnalyzer()

        # If text is very long, focus on first 1000 characters for better analysis
        analysis_text = text[:1000] if len(text) > 1000 else text

        scores = analyzer.polarity_scores(analysis_text)

        # Interpret the compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "Positive"
            interpretation = "Bullish market sentiment - favorable for investment"
        elif compound <= -0.05:
            sentiment = "Negative"
            interpretation = "Bearish market sentiment - exercise caution"
        else:
            sentiment = "Neutral"
            interpretation = "Mixed market sentiment - analyze fundamentals carefully"

        return f"""Sentiment Analysis Results:
Overall Sentiment: {sentiment}
Market Interpretation: {interpretation}
Compound Score: {compound:.3f} (range: -1 to +1)
Detailed Breakdown:
• Positive: {scores['pos']:.1%}
• Neutral: {scores['neu']:.1%} 
• Negative: {scores['neg']:.1%}

Investment Implication: {interpretation}
Text Analyzed: "{analysis_text[:100]}{'...' if len(analysis_text) > 100 else ''}"
"""

    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"


# Register tools for LangChain agent
search = Tool(
    name="search",
    func=search_tool,
    description="Search for recent financial news about a stock. Input should be a ticker symbol or company name."
)

financial_data_analysis = Tool(
    name="financial_data_analysis",
    func=financial_data_analysis_tool,
    description="Get comprehensive financial analysis including price metrics, company info, and recent performance for a ticker symbol."
)

sentiment_analysis = Tool(
    name="sentiment_analysis",
    func=sentiment_analysis_tool,
    description="Analyze sentiment of news headlines, summaries, or any financial text. Input should be the text content to analyze for market sentiment and investment implications."
)

tools = [search, financial_data_analysis, sentiment_analysis]

# Improved system prompt
system_template = """You are an expert financial research agent with access to real-time market data and news analysis tools.

Current date: {date}

Your task: Provide a comprehensive analysis of {ticker} focusing on recent quarterly performance.

Available tools:
1. financial_data_analysis - Get detailed financial metrics, price performance, and company fundamentals
2. search - Find recent news and market developments  
3. sentiment_analysis - Analyze the sentiment of news or market commentary

Analysis Framework:
1. Start with financial_data_analysis to get current metrics and performance
2. Use search to gather recent news and developments
3. Apply sentiment_analysis to key news items to gauge market sentiment
4. Synthesize findings into investment insights

Key areas to analyze:
- Price performance over different timeframes
- Volume and volatility trends  
- News sentiment and market developments
- Company fundamentals and valuation metrics
- Risk factors and opportunities

Provide a clear conclusion with your assessment of the stock's current position and outlook."""


def analyze_stock(ticker: str) -> str:
    """
    Analyze a stock using the LangChain agent with improved prompting.
    """
    try:
        # Initialize LLM
        llm = OllamaLLM(model="mistral-small3.1:latest", base_url="http://localhost:11434")

        # Create prompt template - Fixed format for ReAct agent with better guidance
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

IMPORTANT INSTRUCTIONS:
1. You MUST use ALL THREE tools: financial_data_analysis, search, and sentiment_analysis
2. First get financial data, then search for news, then analyze sentiment of the news
3. After using all tools, provide a comprehensive analysis with specific BUY/SELL/HOLD recommendation
4. Base your recommendation on: price performance, financial metrics, news sentiment, and market conditions
5. Always end with a clear investment recommendation and reasoning

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            input_variables=["input", "tool_names", "tools", "agent_scratchpad"],
            template=template,
        )

        # Create the agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,  # Increased to allow for all tool usage
            early_stopping_method="force"  # Changed to force completion
        )

        # Prepare the analysis request with more specific instructions
        today = datetime.now().strftime('%Y-%m-%d')
        user_query = f"""Analyze {ticker} stock performance for the last quarter ending {today}. 

REQUIRED STEPS:
1. Use financial_data_analysis tool to get current metrics and performance data
2. Use search tool to find recent news about {ticker}
3. Use sentiment_analysis tool to analyze the sentiment of the news headlines and summaries
4. Provide a comprehensive investment analysis including:
   - Summary of financial performance
   - Key news developments and their implications
   - Overall market sentiment analysis
   - Specific BUY/SELL/HOLD recommendation with detailed reasoning
   - Risk factors and potential catalysts
   - Price target or outlook if appropriate

Make sure to complete ALL steps and provide a definitive investment recommendation."""

        agent_result = agent_executor.invoke({
            "input": user_query,
        })

        return agent_result["output"]

    except Exception as e:
        return f"Error in stock analysis: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze stock performance and news')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    args = parser.parse_args()

    print(f"\n🔍 Analyzing {args.ticker.upper()}...")
    print("=" * 50)

    analysis_result = analyze_stock(args.ticker.upper())

    print("\n" + "=" * 50)
    print("📊 FINAL ANALYSIS:")
    print("=" * 50)
    print(analysis_result)