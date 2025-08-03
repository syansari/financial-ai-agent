# Financial AI Agent

This project is an expert financial research agent built with LangChain, yfinance, and sentiment analysis tools. It provides comprehensive stock analysis using real-time market data, news, and sentiment evaluation.

## Features
- **Financial Data Analysis**: Get detailed price metrics, company info, and recent performance for any stock ticker.
- **News Search**: Fetch and summarize the latest financial news for a given stock.
- **Sentiment Analysis**: Analyze the sentiment of news headlines and summaries to gauge market mood.
- **Investment Recommendation**: Synthesizes data and news into actionable BUY/SELL/HOLD recommendations.

## Requirements
- Python 3.12+
- [Ollama](https://ollama.com/) running locally for LLM (default: `http://localhost:11434`)

## Installation

```bash
# Clone the repository
$ git clone <repo-url>

# Navigate to the project directory
$ cd financial-ai-agent

# Create a virtual environment
$ python -m venv .venv   

# lock dependencies using Poetry
$ poetry lock

# Install dependencies using Poetry
$ poetry install --no-root

# Activate the virtual environment
$ source .venv/bin/activate

```

## Usage

```bash
poetry run python agent.py <TICKER>
```
Example:
```bash
poetry run python agent.py AAPL
```

## How It Works
- The agent uses three tools: financial data analysis, news search, and sentiment analysis.
- It combines these to provide a thorough investment analysis and recommendation.

## Configuration
- The LLM model and base URL can be changed in `agent.py`.

## License
MIT
