# Multi-Agent Financial AI Analysis System

This project is a sophisticated multi-agent financial research system built with LangChain, featuring primary analysis and expert critique validation. It provides institutional-quality stock analysis using real-time market data, news, sentiment evaluation, and comprehensive risk assessment.

## Features

### Primary Analysis Agent
- **Financial Data Analysis**: Real-time price metrics, market cap, P/E ratios, and 3-month performance trends
- **News Intelligence**: Latest financial news retrieval and summarization with publication details
- **Sentiment Analysis**: VADER-based sentiment scoring (Bullish/Bearish/Neutral) with confidence scores
- **Investment Recommendations**: Data-driven BUY/SELL/HOLD decisions with supporting rationale

### Expert Critique Agent
- **Risk Assessment**: Volatility analysis, beta calculations, debt-to-equity ratios, and liquidity metrics
- **Analysis Validation**: Completeness checking for required elements (price, trends, valuations)
- **Quality Assurance**: Reasoning validation and recommendation alignment verification
- **Performance Grading**: A-F quality scoring with specific improvement suggestions

### Multi-Agent Architecture
- **Sequential Processing**: Primary analysis → Expert critique → Comprehensive report
- **Real-time Streaming**: Live reasoning process visibility with token-by-token output
- **Error Handling**: Robust input parsing and API failure management
- **Flexible Execution**: Choice between single-agent or multi-agent analysis modes

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

### Multi-Agent Analysis (Default)
Runs both primary analysis and expert critique for comprehensive evaluation:

```bash
poetry run python agent.py <TICKER>
```

### Single Agent Analysis
Runs only the primary analysis agent (legacy mode):

```bash
poetry run python agent.py --single <TICKER>
```

### Examples

```bash
# Multi-agent analysis with critique validation
poetry run python agent.py AAPL

# Single agent analysis only
poetry run python agent.py --single TSLA

# Test different tickers
poetry run python agent.py MSFT
poetry run python agent.py GOOGL
```

## How It Works

### Phase 1: Primary Analysis
The primary agent uses **ReAct framework** (Reason-Act-Observe) with these tools:
- **financial_data_analysis**: Real-time metrics from yfinance API
- **search**: Recent news extraction and summarization
- **sentiment_analysis**: VADER sentiment scoring of news content

### Phase 2: Expert Critique (Multi-Agent Mode)
The critique agent validates the primary analysis using:
- **risk_assessment**: Volatility, beta, and financial ratio analysis
- **validate_analysis**: Completeness and reasoning quality checks

### Output Format
- **Primary Analysis**: Initial investment recommendation with supporting data
- **Expert Critique**: Risk validation and quality assessment with A-F grading
- **Investment Summary**: Consolidated decision guidance considering both perspectives

## Configuration

### LLM Settings
```python
# In agent.py - lines 257-261 and 308-312
llm = OllamaLLM(
    model="mistral-small3.1:latest",    # Change model here
    base_url="http://localhost:11434",  # Change Ollama URL
    temperature=0.1,                    # Adjust creativity (0.0-1.0)
)
```

### Tool Parameters
- **News items**: Default 3 recent articles (configurable in `get_latest_news`)
- **Historical data**: 3-month performance analysis, 1-year risk assessment
- **Agent iterations**: 8 max for primary, 6 max for critique agent
- **Sentiment threshold**: ±0.05 for bullish/bearish classification

### Callback Customization
The `VerboseCallback` class provides real-time logging. Disable by removing from `callbacks=[]` parameter.

## Architecture Notes

### ReAct Format
Both agents follow strict ReAct formatting:
```
Question: [input question]
Thought: [reasoning process]
Action: [tool selection]
Action Input: [tool parameter]
Observation: [tool output]
... (repeat as needed)
Final Answer: [conclusion]
```

### Error Handling
- **Input parsing**: Handles "ticker: AAPL" and "AAPL" formats
- **API failures**: Graceful degradation with error messages
- **Missing data**: Validation with appropriate fallbacks
- **Parse errors**: Automatic retry with `handle_parsing_errors=True`

### Dependencies
- **yfinance**: Real-time financial data
- **vaderSentiment**: News sentiment analysis
- **langchain**: Agent framework and LLM integration
- **langchain-ollama**: Local LLM connection

## License
MIT
