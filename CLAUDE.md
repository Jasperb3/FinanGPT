# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinanGPT is a Python-based financial data pipeline with an intelligent conversational query interface. It combines comprehensive data ingestion, smart caching, and natural language querying to provide a powerful financial analysis platform.

### Core Capabilities

**Data Ingestion**:
- US-only stock financials from yfinance (annual & quarterly statements)
- Price history (OHLCV), dividends, stock splits, and company metadata
- MongoDB storage with strict validation (non-ETF, USD-only, US-listed)
- Smart caching with incremental updates for fast refresh
- Automated freshness tracking for all data types

**Data Analytics**:
- DuckDB transformation for high-performance analytics
- Derived financial ratios (ROE, ROA, profit margins, cash conversion, etc.)
- Year-over-year growth calculations (revenue, income)
- Multiple data schemas (financials, prices, dividends, splits, metadata)

**Query Interface**:
- Natural language to SQL via LLM (Ollama)
- **One-shot queries** (`query.py`) - Single question with freshness checking
- **Conversational mode** (`chat.py`) - Multi-turn dialogue with context memory
- Intelligent error recovery with automatic retry
- Built-in freshness warnings and data validation

### Enhancement Phases (Completed)

**Phase 1: Rich Data Sources** ✅
- Extended beyond basic financials to include:
  - Daily stock prices (OHLCV)
  - Dividend payment history
  - Stock split events
  - Company metadata (sector, industry, employees, etc.)
- Derived analytics tables:
  - Financial ratios (9 key metrics)
  - Year-over-year growth view

**Phase 2: Smart Caching & Incremental Updates** ✅
- Data freshness tracking with MongoDB metadata
- Three ingestion modes: Normal, Refresh (smart caching), Force
- Incremental price updates (10-100x faster)
- Pre-query staleness warnings
- Automated refresh workflows

**Phase 3: Conversational Query Interface** ✅
- Interactive chat mode with multi-turn context
- Conversation history management (20 message rolling window)
- Intelligent error recovery (3 automatic retries with LLM feedback)
- Query suggestions and examples at startup
- Special commands: `/help`, `/clear`, `/exit`

**Phase 4: Visual Analytics & Charting** ✅
- Automatic chart generation from query results
- Intelligent chart type detection (line, bar, scatter, candlestick)
- Financial value formatting ($1.50B, 25.00%, etc.)
- Multiple export formats (CSV, JSON, Excel)
- Enhanced terminal output with financial formatting

**Phase 5: Advanced Query Capabilities** ✅
- Peer group analysis with predefined industry groups (FAANG, Semiconductors, etc.)
- Natural language date parsing ("last year", "YTD", "2023")
- Window functions support (RANK, ROW_NUMBER, LAG, LEAD)
- Statistical aggregations (AVG, STDDEV, MEDIAN)
- Portfolio tracking table for investment analysis

**Phase 6: Error Resilience & UX Polish** ✅
- Graceful degradation when Ollama is unavailable (direct SQL, templates, exit)
- Query template library with 10+ pre-built templates and parameter substitution
- Ticker validation with autocomplete and spell-check capabilities
- Debug mode with comprehensive logging and query timing

**Phase 7: Unified Workflow & Automation** ✅
- Unified CLI entry point (`finangpt.py`) for all operations
- Configuration file support (`config.yaml`) with environment variable fallback
- Status command for system health and data freshness monitoring
- Scheduled update scripts for automated daily refresh (cron-ready)

## Environment Setup

**Virtual Environment (Required)**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

**Dependencies**:
```bash
pip install -r requirements.txt
# Includes: yfinance, pandas, duckdb, pymongo, requests, python-dotenv, pytest
# Visualization: matplotlib, mplfinance, openpyxl
```

**Environment Variables** (`.env`):
```bash
MONGO_URI=mongodb://localhost:27017/financial_data  # Must include database name
OLLAMA_URL=http://localhost:11434
MODEL_NAME=gpt-oss:latest  # or phi4:latest
PRICE_LOOKBACK_DAYS=365  # Default: 1 year of price history
```

**Required Services**:
- MongoDB running locally (mongod)
- Ollama running with model pulled: `ollama pull gpt-oss:latest`

## Core Commands

**Run full pipeline**:
```bash
# 1. Ingest raw data
python ingest.py --tickers AAPL,MSFT,GOOGL
# or from file
python ingest.py --tickers-file tickers.csv

# 2. Transform to DuckDB
python transform.py

# 3. Query via natural language
python query.py "annual net income for AAPL over the last 5 years"
```

**Smart Caching & Incremental Updates (Phase 2)**:
```bash
# Refresh mode: only update data older than 7 days (default threshold)
python ingest.py --refresh --tickers AAPL,MSFT

# Custom refresh threshold (e.g., 3 days)
python ingest.py --refresh --refresh-days 3 --tickers-file tickers.csv

# Force mode: re-fetch all data regardless of freshness
python ingest.py --force --tickers AAPL,MSFT

# Query with automatic freshness checking (warns if data is stale)
python query.py "show AAPL stock price trends"

# Skip freshness check in query
python query.py --skip-freshness-check "show AAPL stock price trends"
```

**Conversational Query Interface (Phase 3)**:
```bash
# Interactive chat mode with context memory
python chat.py

# Example conversation:
# You: Show AAPL revenue for last 5 years
# AI: [Shows revenue data]
# You: Now compare to MSFT
# AI: [Shows comparison with same time range]

# Chat with freshness check disabled
python chat.py --skip-freshness-check

# Special commands within chat:
# /help    - Show help message
# /clear   - Clear conversation history
# /exit    - Exit chat mode
```

**Visual Analytics & Charting (Phase 4)**:
```bash
# Query with automatic chart generation
python query.py "plot AAPL stock price over the last year"

# Query with enhanced financial formatting
python query.py "show AAPL revenue and profit margins"
# Output: $394.00B, 23.76%, etc.

# Disable chart generation
python query.py --no-chart "show revenue trends"

# Disable financial formatting
python query.py --no-formatting "show raw data"

# Chat mode also includes automatic visualization
python chat.py
# You: Compare AAPL and MSFT revenue
# AI: [Shows table + generates bar chart automatically]

# Charts saved to: charts/Query_Result_YYYYMMDD_HHMMSS.png
```

**Testing**:
```bash
# Run all tests
python -m pytest tests

# Run specific test file
python -m pytest tests/test_ingest_filters.py -v

# Run with output
python -m pytest tests -v -s
```

## Architecture & Data Flow

### Three-Stage Pipeline

1. **Ingestion (`ingest.py`)**:
   - Fetches financial statements from yfinance with exponential backoff (3 attempts)
   - Fetches price history (OHLCV), dividends, stock splits, and company metadata
   - Validates: US-listed, non-ETF, USD currency
   - Normalizes dates to UTC (16:00 US/Eastern)
   - Merges Income Statement + Balance Sheet + Cash Flow per reporting period
   - Upserts to MongoDB collections:
     - `raw_annual` / `raw_quarterly` (financial statements)
     - `stock_prices_daily` (OHLCV data)
     - `dividends_history` (dividend payments)
     - `splits_history` (stock split events)
     - `company_metadata` (sector, industry, description, etc.)
     - `ingestion_metadata` (freshness tracking)
   - Compound index: `{ticker: 1, date: 1}` (unique) on time-series collections

2. **Transformation (`transform.py`)**:
   - Reads from MongoDB raw collections
   - Flattens nested payloads into numeric-only columns
   - Loads into DuckDB tables:
     - `financials.annual` / `financials.quarterly` (statements)
     - `prices.daily` (OHLCV price data)
     - `dividends.history` (dividend payments)
     - `splits.history` (stock splits)
     - `company.metadata` (company information)
     - `ratios.financial` (derived metrics: ROE, ROA, margins, etc.)
   - Creates `growth.annual` view (YoY revenue/income growth)
   - Idempotent via delete-then-insert on `{ticker, date}`

3. **Query (`query.py`)**:
   - Introspects DuckDB schema to build dynamic system prompt
   - Calls Ollama `/api/chat` endpoint with schema-aware prompt
   - Validates SQL: table allow-list, column verification, SELECT only
   - Supports 8 table schemas (financials, prices, dividends, splits, metadata, ratios, growth)
   - Enforces LIMIT 25 (default) / 100 (max)
   - Executes against DuckDB and pretty-prints results

### Key Validation Rules

**Ingestion Filters** (fail closed for data quality):
- `is_etf()`: Rejects ETFs via quoteType, fundFamily, or name patterns
- `is_us_listing()`: country="United States" OR market="us_market"
- `has_usd_financials()`: financialCurrency or currency == "USD"
- Missing metadata → rejection (no guessing allowed)

**SQL Guardrails** (`query.py`):
- Only SELECT statements allowed (no DDL/DML/multi-statement); top-level `WITH` clauses and CTEs are supported so long as the final statement is a SELECT.
- Table allow-list:
  - `financials.annual`, `financials.quarterly`
  - `prices.daily`
  - `dividends.history`
  - `splits.history`
  - `company.metadata`
  - `ratios.financial`
  - `growth.annual`
- All columns must exist in schema (CTEs inherit the same validation; references must resolve to a prior CTE or allowed table)
- LIMIT ≤ 100 (auto-adds LIMIT 25 if missing)

### Field Normalization

`FIELD_MAPPINGS` in `ingest.py` (lines 45-101) canonicalizes variable field names:
```python
"netIncome" → ["Net Income", "NetIncome", "Net Income Applicable To Common Shares"]
"totalRevenue" → ["Total Revenue", "Revenue", "Revenues"]
```

When adding new fields, extend this mapping to handle yfinance variations.

## Data Structures

**MongoDB Document** (`raw_annual` / `raw_quarterly`):
```json
{
  "ticker": "AAPL",
  "date": "2024-12-31T21:00:00Z",  // UTC, normalized from US/Eastern 16:00
  "period": "annual",
  "currency": "USD",
  "payload": {
    "income_statement": { "netIncome": 123456000000 },
    "balance_sheet": { "totalAssets": 394000000000 },
    "cash_flow": { "operatingCashFlow": 87654000000 }
  },
  "source": "yfinance",
  "fetched_at": "2024-11-09T10:30:00Z",
  "instrument": { "longName": "Apple Inc.", "exchange": "NASDAQ" }
}
```

**DuckDB Schemas**:

`financials.annual` / `financials.quarterly`:
```sql
ticker VARCHAR
date DATE
netIncome DOUBLE
totalAssets DOUBLE
totalLiabilities DOUBLE
operatingCashFlow DOUBLE
totalRevenue DOUBLE
shareholderEquity DOUBLE
freeCashFlow DOUBLE
grossProfit DOUBLE
ebitda DOUBLE
-- [additional numeric columns sorted alphabetically]
```

`prices.daily`:
```sql
ticker VARCHAR
date DATE
open DOUBLE
high DOUBLE
low DOUBLE
close DOUBLE
adj_close DOUBLE
volume BIGINT
```

`dividends.history`:
```sql
ticker VARCHAR
date DATE
amount DOUBLE
```

`splits.history`:
```sql
ticker VARCHAR
date DATE
ratio DOUBLE
```

`company.metadata`:
```sql
ticker VARCHAR
longName VARCHAR
shortName VARCHAR
sector VARCHAR
industry VARCHAR
website VARCHAR
country VARCHAR
exchange VARCHAR
quoteType VARCHAR
marketCap BIGINT
employees INTEGER
description TEXT
currency VARCHAR
financialCurrency VARCHAR
```

`ratios.financial`:
```sql
ticker VARCHAR
date DATE
net_margin DOUBLE          -- netIncome / totalRevenue
roe DOUBLE                 -- Return on Equity: netIncome / shareholderEquity
roa DOUBLE                 -- Return on Assets: netIncome / totalAssets
debt_ratio DOUBLE          -- totalLiabilities / totalAssets
cash_conversion DOUBLE     -- operatingCashFlow / netIncome
fcf_margin DOUBLE          -- freeCashFlow / totalRevenue
asset_turnover DOUBLE      -- totalRevenue / totalAssets
gross_margin DOUBLE        -- grossProfit / totalRevenue
ebitda_margin DOUBLE       -- ebitda / totalRevenue
```

`growth.annual` (VIEW):
```sql
ticker VARCHAR
date DATE
totalRevenue DOUBLE
netIncome DOUBLE
prior_revenue DOUBLE
prior_income DOUBLE
revenue_growth_yoy DOUBLE  -- (current - prior) / prior revenue
income_growth_yoy DOUBLE   -- (current - prior) / prior income
```

## Logging

All scripts emit structured JSON logs to `logs/`:
```json
{
  "ts": "2024-11-09T10:30:00Z",
  "phase": "ingest.annual",
  "ticker": "AAPL",
  "attempts": 1,
  "rows": 5,
  "duration_ms": 1234
}
```

## Phase 2: Smart Caching & Incremental Updates

### Data Freshness Tracking

FinanGPT tracks the freshness of all ingested data using the `ingestion_metadata` collection in MongoDB:

```json
{
  "ticker": "AAPL",
  "data_type": "prices_daily",
  "last_fetched": "2025-11-09T10:30:00Z",
  "status": "success",
  "record_count": 365
}
```

**Tracked data types**:
- `financials_annual` - Annual financial statements
- `financials_quarterly` - Quarterly financial statements
- `prices_daily` - Daily stock prices (OHLCV)
- `dividends_history` - Dividend payment history
- `splits_history` - Stock split history
- `company_metadata` - Company information

### Ingestion Modes

**Normal Mode** (default):
```bash
python ingest.py --tickers AAPL,MSFT
```
- Fetches all data for specified tickers
- Updates existing records (upsert behavior)
- Uses full lookback period for prices (365 days default)

**Refresh Mode** (`--refresh`):
```bash
python ingest.py --refresh --refresh-days 7 --tickers AAPL,MSFT
```
- Only processes tickers with data older than threshold (default: 7 days)
- Skips tickers with fresh data to reduce API load
- Uses incremental price updates (only new data since last fetch)
- Significantly faster for frequent updates

**Force Mode** (`--force`):
```bash
python ingest.py --force --tickers AAPL,MSFT
```
- Re-fetches all data regardless of freshness
- Uses full lookback period (no incremental updates)
- Use when data quality issues are suspected

### Incremental Price Updates

When not in force mode, price fetching is incremental:

```python
# Gets last stored price date from MongoDB
last_price_date = get_last_price_date(collection, "AAPL")
# Only fetches prices after last_price_date
price_df = fetch_price_history(ticker_obj, "AAPL", last_date=last_price_date)
```

**Benefits**:
- Reduces yfinance API load
- Faster ingestion (seconds instead of minutes)
- Enables frequent automated updates (hourly/daily via cron)

### Query Freshness Checking

`query.py` automatically checks data freshness before executing queries:

```python
# Extracts tickers from SQL: WHERE ticker = 'AAPL'
tickers = extract_tickers_from_sql(sql)

# Checks MongoDB metadata for staleness
freshness = check_data_freshness(mongo_db, tickers, threshold_days=7)

if freshness["is_stale"]:
    print("⚠️  Warning: Data may be stale")
    print(f"Stale tickers: {', '.join(freshness['stale_tickers'])}")
    user_input = input("Continue with stale data? [y/N]: ")
```

**Skip freshness check**:
```bash
python query.py --skip-freshness-check "show AAPL revenue"
```

### Staleness Detection Logic

A ticker's data is considered stale if:
1. **Never fetched**: No metadata record exists for the ticker
2. **Threshold exceeded**: `last_fetched` is older than `threshold_days` (default: 7)

The staleness check uses the most recent fetch across all data types:
- If any data type is stale, the ticker is flagged
- Provides detailed freshness info per ticker

### Automated Refresh Workflow

**Daily scheduled update** (via cron):
```bash
# crontab entry: Run at 6 PM weekdays
0 18 * * 1-5 /path/to/.venv/bin/python /path/to/ingest.py --refresh --tickers-file tickers.csv

# After ingestion, transform to DuckDB
0 18 * * 1-5 /path/to/.venv/bin/python /path/to/transform.py
```

**Manual refresh** (on-demand):
```bash
# Update all tickers that are >7 days old
python ingest.py --refresh --tickers-file tickers.csv

# Update specific ticker if stale
python ingest.py --refresh --tickers AAPL

# Transform updated data
python transform.py
```

### Monitoring Data Freshness

**Check freshness via MongoDB**:
```javascript
// MongoDB shell
use financial_data
db.ingestion_metadata.find({"ticker": "AAPL"}).sort({"last_fetched": -1})

// Find stale data (>7 days)
var threshold = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
db.ingestion_metadata.find({"last_fetched": {$lt: threshold.toISOString()}})
```

**Freshness in logs**:
```json
{
  "ts": "2025-11-09T10:00:00Z",
  "phase": "skip.fresh",
  "ticker": "AAPL",
  "message": "Data is fresh (less than 7 days old), skipping."
}
```

## Phase 3: Conversational Query Interface

### Overview

Phase 3 introduces an interactive chat interface (`chat.py`) that provides a conversational experience for querying financial data. Unlike the one-shot `query.py`, the chat interface maintains conversation context, remembers previous queries, and can handle follow-up questions naturally.

### Key Features

**Multi-Turn Conversation**:
- Maintains conversation history with the LLM
- Remembers previous queries and results
- Supports contextual follow-ups without repeating information
- Rolling window of 20 most recent messages to prevent token overflow

**Intelligent Error Recovery**:
- Automatic retry on query failures (up to 3 attempts)
- Feeds error messages back to LLM for self-correction
- Users never see internal SQL validation errors
- LLM learns from mistakes within the session

**Query Suggestions**:
- Welcome screen with example queries
- Built-in help system (`/help` command)
- Contextual tips on failures

**Special Commands**:
- `/help` - Display help message and usage tips
- `/clear` - Reset conversation history
- `/exit` or `/quit` - Exit the chat interface

### Usage

**Start Interactive Session**:
```bash
python chat.py
```

**Example Conversation**:
```
💬 Query> Show AAPL revenue for last 5 years

📊 Generated SQL: SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 5

✅ Results (5 rows):

ticker | date       | totalRevenue
-------|------------|---------------
AAPL   | 2024-09-30 | 394328000000
AAPL   | 2023-09-30 | 383285000000
...

💬 Query> Now compare to MSFT

📊 Generated SQL: SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker IN ('AAPL', 'MSFT') ORDER BY ticker, date DESC LIMIT 10

✅ Results (10 rows):
[Shows both AAPL and MSFT with same time range]
```

### Conversation History Management

The chat interface maintains conversation context using a rolling window:

```python
conversation_history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Show AAPL revenue"},
    {"role": "assistant", "content": "Query executed..."},
    {"role": "system", "content": "Query returned 5 rows"},
    {"role": "user", "content": "Now compare to MSFT"},
    # ... continues
]
```

**History Limits**:
- Maximum 20 messages retained (excluding system prompt)
- Older messages automatically trimmed to prevent token overflow
- System prompt always preserved
- Most recent context prioritized

### Error Recovery Flow

When a query fails, the system:

1. **Attempt 1**: Generate SQL from user query
2. **Validation Fail**: Detect error (e.g., invalid table)
3. **Feedback**: Send error details to LLM
4. **Attempt 2**: LLM generates revised SQL
5. **Success**: Execute and return results

**Example**:
```
User: "Show me all the stocks"
Attempt 1: SELECT * FROM stocks  ← Invalid table
Feedback: "Table 'stocks' not in allow-list"
Attempt 2: SELECT * FROM company.metadata LIMIT 25  ← Success
```

### Integration with Phase 2

The chat interface inherits all Phase 2 features:
- **Freshness checking**: Warns about stale data before execution
- **Smart caching**: Benefits from incremental updates
- **Skip option**: `--skip-freshness-check` flag available

### Commands Reference

| Command | Action |
|---------|--------|
| `/help` | Show help message and tips |
| `/clear` | Clear conversation history |
| `/exit` | Exit chat mode |
| `/quit` | Exit chat mode (alias) |

### Chat Logs

All chat sessions are logged to `logs/chat_YYYYMMDD.log`:

```json
{
  "ts": "2025-11-09T15:30:00Z",
  "phase": "chat.start",
  "model": "phi4:latest"
}
{
  "ts": "2025-11-09T15:30:15Z",
  "phase": "query.success",
  "attempt": 1,
  "sql": "SELECT ticker, totalRevenue FROM financials.annual LIMIT 10",
  "rows": 10
}
{
  "ts": "2025-11-09T15:35:00Z",
  "phase": "chat.end"
}
```

### Best Practices

**For Better Results**:
1. Be specific with ticker symbols (AAPL vs "Apple")
2. Mention time ranges when relevant
3. Use action verbs: "show", "compare", "list"
4. Build on previous queries with follow-ups

**Example Good Conversation**:
```
✅ "Show AAPL revenue trends for last 5 years"
✅ "Now add profit margins for the same period"
✅ "How does this compare to MSFT?"
```

**Example Poor Conversation**:
```
❌ "Tell me about Apple"  (too vague)
❌ "What's the data?"  (no context)
❌ "Show everything"  (no specificity)
```

### Troubleshooting

**Chat doesn't remember context**:
- Check if conversation history is being trimmed too aggressively
- Try `/clear` to reset and start fresh
- Ensure Ollama is running and model supports conversation

**Queries fail repeatedly**:
- Use `/help` to see valid query patterns
- Check `logs/chat_YYYYMMDD.log` for error details
- Try one-shot mode: `python query.py "your question"`

**Ollama timeout**:
- Increase timeout in code if needed (default: 60s)
- Check Ollama server status: `ollama list`
- Verify model is loaded: `ollama run phi4:latest`

## Phase 4: Visual Analytics & Charting

### Overview

Phase 4 introduces automatic visualization capabilities and enhanced financial formatting across all query interfaces. The system intelligently detects when a query would benefit from visual representation and automatically generates appropriate charts.

### Key Features

**Intelligent Chart Detection**:
- Automatic detection from query keywords ("plot", "chart", "compare", "trend")
- Data structure analysis (time-series, OHLC, multi-ticker comparisons)
- Support for 4 chart types: line, bar, scatter, candlestick
- Works with both `query.py` and `chat.py` interfaces

**Financial Value Formatting**:
- Large numbers: $1.50B, $250.00M, $3.45K
- Percentages: 25.00% (for margins, ratios, growth)
- Prices: $150.25 (for stock prices)
- Volume: 1,500,000 (comma-separated integers)
- Smart detection based on column names

**Export Capabilities**:
- CSV export with proper encoding
- JSON export with date serialization
- Excel export with formatted sheets
- All exports preserve data types

**Enhanced Terminal Output**:
- Pretty-printed tables with financial formatting
- Column name beautification (camelCase → Title Case)
- Alignment and spacing optimized for readability

### Chart Types

**Line Chart** (Time-Series Data):
```python
# Automatically selected for queries with:
# - Date/time column + numeric values
# - Keywords: "trend", "over time", "history"

# Example query:
"Plot AAPL revenue over the last 5 years"
```

**Bar Chart** (Comparisons):
```python
# Automatically selected for:
# - Multiple tickers with single metric
# - Keywords: "compare", "comparison"

# Example query:
"Compare revenue across AAPL, MSFT, GOOGL"
```

**Scatter Plot** (Correlations):
```python
# Automatically selected for:
# - Two numeric columns without date
# - Keywords: "correlation", "relationship"

# Example query:
"Show relationship between ROE and debt ratio"
```

**Candlestick Chart** (OHLC Data):
```python
# Automatically selected for:
# - Data with open, high, low, close columns
# - Keywords: "candlestick", "ohlc"

# Example query:
"Show candlestick chart for AAPL prices in Q4 2024"
```

### Financial Formatting Rules

The system applies intelligent formatting based on column names:

**Revenue/Income Columns** → Large Number Format:
- `totalRevenue`, `netIncome`, `operatingIncome`, `grossProfit`, `ebitda`
- `totalAssets`, `totalLiabilities`, `shareholderEquity`
- `operatingCashFlow`, `freeCashFlow`, `marketCap`
- Format: $1.50T, $394.00B, $25.00M, $750.00K

**Ratio/Margin Columns** → Percentage Format:
- `net_margin`, `gross_margin`, `ebitda_margin`, `fcf_margin`
- `roe`, `roa`, `debt_ratio`, `asset_turnover`
- `revenue_growth_yoy`, `income_growth_yoy`
- Format: 25.00%, 15.50%, -3.25%

**Price Columns** → Currency Format:
- `close`, `open`, `high`, `low`, `adj_close`
- Format: $150.25, $3,247.89

**Volume Columns** → Integer Format:
- `volume`, `shares`, `sharesOutstanding`
- Format: 1,500,000 (comma-separated)

**Default** → Decimal Format:
- Unknown columns: 123.46 (2 decimal places)

### Integration with Query Interfaces

**One-Shot Queries (`query.py`)**:
```python
# Automatic visualization + formatting (default)
python query.py "plot AAPL stock price trends"

# Disable visualization
python query.py --no-chart "show revenue data"

# Disable formatting (raw numbers)
python query.py --no-formatting "show raw revenue"

# Disable both
python query.py --no-chart --no-formatting "show data"
```

**Conversational Mode (`chat.py`)**:
```python
# Visualization and formatting always enabled in chat
python chat.py

# Example session:
You: Show AAPL revenue for last 3 years
AI: [Displays formatted table with $B values]

You: Plot this as a chart
AI: [Generates line chart automatically]
    📈 Chart saved: charts/Query_Result_20251109_143022.png
```

### Chart Output

All charts are saved to the `charts/` directory with timestamped filenames:

```bash
charts/
├── Query_Result_20251109_143022.png
├── Query_Result_Line_Chart_20251109_143155.png
├── Query_Result_Bar_Chart_20251109_144312.png
└── ...
```

**Chart Features**:
- Professional styling with grid lines
- Automatic axis labeling from column names
- Legend with ticker symbols (multi-ticker charts)
- Date formatting on x-axis (time-series)
- High-resolution PNG output (300 DPI)
- Automatic figure sizing based on data

### Export Functions

**CSV Export**:
```python
from visualize import export_to_csv
export_to_csv(df, "output.csv")
# Creates: CSV file with UTF-8 encoding
```

**JSON Export**:
```python
from visualize import export_to_json
export_to_json(df, "output.json")
# Creates: JSON with date serialization
```

**Excel Export**:
```python
from visualize import export_to_excel
export_to_excel(df, "output.xlsx")
# Creates: Excel file with auto-sized columns
```

### Auto-Detection Algorithm

The chart type is determined by:

1. **Keyword Detection** (highest priority):
   - "plot", "chart", "graph" → Enable visualization
   - "candlestick", "ohlc" → Candlestick chart
   - "scatter", "correlation" → Scatter plot
   - "compare", "comparison" → Bar chart
   - "trend", "over time" → Line chart

2. **Data Structure Analysis**:
   - Has OHLC columns? → Candlestick
   - Has date column? → Line chart
   - Multiple tickers, no date? → Bar chart
   - Two numeric columns? → Scatter plot

3. **Fallback**:
   - If no clear match → No chart generated
   - User can explicitly request with keywords

### Formatting in Action

**Before (Phase 1-3)**:
```
ticker | totalRevenue  | net_margin
-------|---------------|------------
AAPL   | 394328000000  | 0.2376
MSFT   | 245122000000  | 0.3596
```

**After (Phase 4)**:
```
Ticker | Total Revenue | Net Margin
-------|---------------|------------
AAPL   | $394.00B      | 23.76%
MSFT   | $245.00B      | 35.96%
```

### Error Handling

**Missing Matplotlib**:
- Optional import pattern: visualization gracefully disabled if not installed
- Query continues to work, but charts are not generated
- User sees informational message

**Invalid Chart Data**:
- Empty DataFrames → No chart generated
- Missing required columns → Falls back to table output
- Invalid chart type → Logs warning, continues

**File System Errors**:
- `charts/` directory created automatically if missing
- Filename sanitization removes invalid characters
- Long filenames truncated to 200 characters

### Performance Considerations

**Chart Generation**:
- Minimal overhead (~100-200ms per chart)
- Charts generated asynchronously (non-blocking)
- Memory efficient (figures closed after save)

**Formatting**:
- Negligible performance impact (<10ms)
- Applied only to display, not to underlying data
- No impact on SQL query execution

### Testing

Phase 4 includes comprehensive test coverage (`tests/test_visualizations.py`):

```bash
# Run visualization tests
python -m pytest tests/test_visualizations.py -v

# Test categories:
# - Visualization intent detection (7 tests)
# - Financial formatting (11 tests)
# - Column name formatting (3 tests)
# - Chart creation (3 tests)
# - Export functions (2 tests)
# - Integration workflows (2 tests)
```

### Usage Examples

**Time-Series Analysis**:
```bash
python query.py "plot AAPL closing price over the last 6 months"
# → Line chart with date on x-axis, price on y-axis
# → Table with formatted prices: $150.25
```

**Multi-Company Comparison**:
```bash
python chat.py
You: Compare net margins for AAPL, MSFT, GOOGL
# → Bar chart comparing the three companies
# → Table with percentages: 23.76%, 35.96%, 28.12%
```

**Financial Ratios**:
```bash
python query.py "show ROE and ROA for tech companies"
# → Table with percentage formatting
# → Scatter plot if both metrics requested
```

**OHLC Stock Data**:
```bash
python query.py "show TSLA daily prices for October 2024 as candlestick"
# → Candlestick chart with OHLC data
# → Table with price formatting
```

### Customization

**Disable Visualization Globally**:
```python
# In query.py or chat.py, set:
VISUALIZATION_AVAILABLE = False
```

**Custom Formatting Rules**:
```python
# In visualize.py, update FORMAT_RULES dict:
FORMAT_RULES = {
    'revenue': lambda v: format_large_number(v),
    'custom_metric': lambda v: f"{v:.3f}x",  # Custom format
}
```

**Chart Styling**:
```python
# In visualize.py chart functions, modify matplotlib settings:
plt.style.use('seaborn-v0_8')  # Different style
plt.rcParams['figure.dpi'] = 150  # Different resolution
```

### Troubleshooting

**Charts not generating**:
- Check if `charts/` directory has write permissions
- Verify matplotlib is installed: `pip list | grep matplotlib`
- Look for error messages in terminal output
- Check query includes visualization keywords

**Formatting looks wrong**:
- Verify column names match formatting rules (see Financial Formatting Rules)
- Check for NULL/None values in data (displayed as "N/A")
- Disable formatting with `--no-formatting` to see raw values

**Chart quality issues**:
- Increase DPI in chart creation functions (default: 300)
- Adjust figure size in code (default: 10x6 inches)
- Use higher-quality output format (PNG with anti-aliasing)

**Export failures**:
- Ensure write permissions for output directory
- Check disk space availability
- Verify filename doesn't contain invalid characters

## Phase 5: Advanced Query Capabilities

### Overview

Phase 5 introduces powerful advanced query capabilities that enable sophisticated financial analysis. These features include peer group comparisons, natural language date parsing, window functions for rankings and trends, and portfolio tracking.

### Key Features

**Peer Group Analysis**:
- Predefined industry peer groups for comparative analysis
- 16+ peer groups including FAANG, Semiconductors, Cloud Computing, etc.
- Automatic ticker expansion for group queries
- Cross-group comparisons and rankings

**Natural Language Date Parsing**:
- Intelligent interpretation of relative dates
- Support for "last year", "last 5 years", "YTD", "2023", etc.
- Dynamic date context injected into system prompt
- Automatic date filtering without explicit WHERE clauses

**Window Functions & Statistical Aggregations**:
- Ranking functions: RANK(), ROW_NUMBER(), DENSE_RANK()
- Analytical functions: LAG(), LEAD(), NTILE()
- Statistical functions: AVG(), STDDEV(), MEDIAN(), PERCENTILE_CONT()
- Support for PARTITION BY and ORDER BY

**Portfolio Tracking**:
- User portfolio table for investment tracking
- Support for multiple portfolios
- Purchase tracking with dates and prices
- Portfolio valuation and performance analysis

### Peer Groups Reference

**Available Peer Groups** (`company.peers` table):

| Peer Group | Tickers | Description |
|------------|---------|-------------|
| FAANG | META, AAPL, AMZN, NFLX, GOOGL | Original tech giants |
| Magnificent Seven | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA | Current tech leaders |
| Semiconductors | NVDA, AMD, INTC, TSM, QCOM, AVGO, MU | Chip manufacturers |
| Cloud Computing | AMZN, MSFT, GOOGL, CRM, ORCL, IBM | Cloud service providers |
| Social Media | META, SNAP, PINS, TWTR, RDDT | Social platforms |
| Streaming | NFLX, DIS, PARA, WBD | Video streaming services |
| E-commerce | AMZN, EBAY, SHOP, ETSY, MELI | Online retailers |
| Payment Processors | V, MA, PYPL, SQ, AXP | Payment networks |
| Electric Vehicles | TSLA, RIVN, LCID, F, GM | EV manufacturers |
| Airlines | AAL, DAL, UAL, LUV, JBLU | Major airlines |
| Banks | JPM, BAC, WFC, C, GS, MS | Banking institutions |
| Oil & Gas | XOM, CVX, COP, SLB, BP, SHEL | Energy companies |
| Defense | LMT, RTX, BA, NOC, GD | Defense contractors |
| Retail | WMT, TGT, COST, HD, LOW | Major retailers |
| Pharma | JNJ, PFE, MRK, ABBV, LLY, BMY | Pharmaceutical companies |
| Telecom | T, VZ, TMUS, CMCSA | Telecommunications |

### Database Schema Additions

**company.peers**:
```sql
CREATE TABLE company.peers (
    ticker VARCHAR,        -- Stock ticker symbol
    peer_group VARCHAR     -- Group name (e.g., "FAANG", "Semiconductors")
);
```

**user.portfolios**:
```sql
CREATE TABLE user.portfolios (
    portfolio_name VARCHAR,   -- Name of portfolio (e.g., "Retirement", "Tech")
    ticker VARCHAR,          -- Stock ticker
    shares DOUBLE,           -- Number of shares owned
    purchase_date DATE,      -- Date of purchase
    purchase_price DOUBLE,   -- Price per share at purchase
    notes VARCHAR            -- Optional notes
);
```

### Natural Language Date Queries

The system now intelligently parses relative dates in natural language:

**Supported Date Patterns**:
- **"last year"** or **"past year"** → `WHERE date >= '2024-11-09'` (365 days ago)
- **"last 5 years"** → `WHERE date >= '2020-11-09'` (5 years ago)
- **"recent"** or **"latest"** → `ORDER BY date DESC LIMIT 1`
- **"2023"** → `WHERE YEAR(date) = 2023`
- **"YTD"** or **"year to date"** → `WHERE YEAR(date) = 2025`

**Example Queries**:
```
"Show AAPL revenue for the last 5 years"
→ Automatically adds: WHERE date >= '2020-11-09'

"What's Apple's latest net income?"
→ Automatically adds: ORDER BY date DESC LIMIT 1

"Compare tech stocks in 2023"
→ Automatically adds: WHERE YEAR(date) = 2023
```

### Window Functions

Phase 5 enables sophisticated analytical queries using window functions:

**Ranking Functions**:
```sql
-- Rank companies by revenue within their peer group
SELECT
    ticker,
    peer_group,
    totalRevenue,
    RANK() OVER (PARTITION BY peer_group ORDER BY totalRevenue DESC) as rank
FROM financials.annual f
JOIN company.peers p ON f.ticker = p.ticker
WHERE YEAR(date) = 2024;
```

**Trend Analysis**:
```sql
-- Calculate revenue growth using LAG
SELECT
    ticker,
    date,
    totalRevenue,
    LAG(totalRevenue) OVER (PARTITION BY ticker ORDER BY date) as prior_revenue,
    (totalRevenue - LAG(totalRevenue) OVER (PARTITION BY ticker ORDER BY date)) /
    LAG(totalRevenue) OVER (PARTITION BY ticker ORDER BY date) as growth_rate
FROM financials.annual;
```

**Percentile Analysis**:
```sql
-- Find companies in top quartile by ROE
SELECT
    ticker,
    roe,
    NTILE(4) OVER (ORDER BY roe DESC) as quartile
FROM ratios.financial
WHERE date = (SELECT MAX(date) FROM ratios.financial);
```

### Peer Group Query Examples

**Compare FAANG Revenue**:
```
Natural language: "Compare FAANG companies by revenue"

Generated SQL:
SELECT f.ticker, f.date, f.totalRevenue
FROM financials.annual f
JOIN company.peers p ON f.ticker = p.ticker
WHERE p.peer_group = 'FAANG'
ORDER BY f.date DESC, f.totalRevenue DESC
LIMIT 25;
```

**Rank Semiconductor Companies**:
```
Natural language: "Rank semiconductor companies by profit margin"

Generated SQL:
SELECT
    r.ticker,
    r.net_margin,
    RANK() OVER (ORDER BY r.net_margin DESC) as margin_rank
FROM ratios.financial r
JOIN company.peers p ON r.ticker = p.ticker
WHERE p.peer_group = 'Semiconductors'
AND r.date = (SELECT MAX(date) FROM ratios.financial)
LIMIT 25;
```

**Cross-Group Comparison**:
```
Natural language: "Compare average ROE for FAANG vs Semiconductors"

Generated SQL:
SELECT
    p.peer_group,
    AVG(r.roe) as avg_roe,
    STDDEV(r.roe) as stddev_roe
FROM ratios.financial r
JOIN company.peers p ON r.ticker = p.ticker
WHERE p.peer_group IN ('FAANG', 'Semiconductors')
AND YEAR(r.date) = 2024
GROUP BY p.peer_group;
```

### Portfolio Tracking

**Add Holdings to Portfolio**:
```sql
INSERT INTO user.portfolios
    (portfolio_name, ticker, shares, purchase_date, purchase_price, notes)
VALUES
    ('Tech Growth', 'AAPL', 100, '2023-01-15', 150.50, 'Long-term hold'),
    ('Tech Growth', 'MSFT', 50, '2023-02-20', 280.00, 'Cloud play');
```

**Portfolio Value Query**:
```
Natural language: "What's my Tech Growth portfolio worth today?"

Generated SQL:
SELECT
    port.ticker,
    port.shares,
    port.purchase_price,
    prices.close as current_price,
    port.shares * port.purchase_price as cost_basis,
    port.shares * prices.close as current_value,
    (prices.close - port.purchase_price) / port.purchase_price as return_pct
FROM user.portfolios port
JOIN prices.daily prices ON port.ticker = prices.ticker
WHERE port.portfolio_name = 'Tech Growth'
AND prices.date = (SELECT MAX(date) FROM prices.daily)
LIMIT 25;
```

**Portfolio Performance**:
```
Natural language: "Show unrealized gains for my holdings"

Generated SQL:
SELECT
    ticker,
    shares,
    purchase_price,
    shares * (
        (SELECT close FROM prices.daily p
         WHERE p.ticker = portfolios.ticker
         ORDER BY date DESC LIMIT 1) - purchase_price
    ) as unrealized_gain
FROM user.portfolios
WHERE portfolio_name = 'Tech Growth';
```

### Statistical Aggregations

**Sector Statistics**:
```sql
-- Calculate median revenue and standard deviation by sector
SELECT
    m.sector,
    MEDIAN(f.totalRevenue) as median_revenue,
    AVG(f.totalRevenue) as mean_revenue,
    STDDEV(f.totalRevenue) as stddev_revenue,
    COUNT(*) as company_count
FROM financials.annual f
JOIN company.metadata m ON f.ticker = m.ticker
WHERE YEAR(f.date) = 2024
GROUP BY m.sector;
```

**Percentile Distributions**:
```sql
-- Find 25th, 50th, and 75th percentile for ROE
SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY roe) as p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY roe) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY roe) as p75
FROM ratios.financial
WHERE date = (SELECT MAX(date) FROM ratios.financial);
```

### System Prompt Enhancements

The system prompt now includes:

1. **Dynamic Date Context**: Automatically calculated relative to today's date
2. **Peer Group Reference**: Lists all available peer groups with examples
3. **Window Functions Documentation**: Explains available analytical functions
4. **Example Queries**: Demonstrates peer group and portfolio queries

This context enables the LLM to:
- Parse natural language dates accurately
- Recognize peer group names and expand them to tickers
- Use window functions appropriately for rankings and trends
- Generate portfolio valuation queries

### Usage Examples

**Time-Based Analysis**:
```
"Show AAPL revenue trends for the last 5 years"
"What was MSFT's latest quarterly income?"
"Compare 2023 vs 2024 performance for tech stocks"
```

**Peer Group Analysis**:
```
"Compare FAANG companies by market cap"
"Rank semiconductor companies by revenue growth"
"Show average profit margins for Cloud Computing companies"
"Which Magnificent Seven stock has the highest ROE?"
```

**Advanced Analytics**:
```
"Show top 10 companies by revenue with their rank"
"Calculate 3-period moving average for AAPL revenue"
"Find companies in the top quartile by ROE"
"Show revenue growth rate vs prior year for each company"
```

**Portfolio Queries**:
```
"What's my portfolio value today?"
"Show unrealized gains for my tech holdings"
"Calculate portfolio allocation by sector"
"Which of my holdings have positive returns?"
```

### Troubleshooting

**Peer group not recognized**:
- Verify the group name matches exactly (case-insensitive)
- Check available groups: See peer_groups.py for complete list
- Use `SELECT DISTINCT peer_group FROM company.peers` to see all groups

**Window function errors**:
- Ensure proper PARTITION BY and ORDER BY clauses
- Check that window functions are in SELECT clause, not WHERE
- Verify column references are valid

**Portfolio queries failing**:
- Ensure portfolio table has data: `SELECT * FROM user.portfolios`
- Add holdings using INSERT statements
- Check portfolio_name matches exactly

**Date parsing issues**:
- Use explicit dates if relative parsing isn't working
- Check that date column exists in the table
- Verify date format is YYYY-MM-DD

## Phase 6: Error Resilience & UX Polish

### Overview

Phase 6 introduces robust error handling and user experience improvements, making FinanGPT more resilient to failures and easier to use. Key features include graceful degradation when services are unavailable, pre-built query templates, ticker validation, and comprehensive debug logging.

### Key Features

**Graceful Degradation**:
- Fallback options when Ollama is unavailable
- Direct SQL input mode (expert users)
- Query template library access
- User choice-driven recovery

**Query Templates**:
- 10+ pre-built templates for common queries
- Parameter substitution with defaults
- YAML-based template library
- Command-line and programmatic access

**Ticker Validation**:
- Check ticker existence before query
- Auto-complete suggestions
- Spell-check and typo detection
- Full ticker list retrieval

**Debug Mode**:
- Comprehensive logging of LLM interactions
- SQL generation and validation visibility
- Query timing and performance metrics
- System prompt inspection

### Query Templates

**Template Library** (`templates/queries.yaml`):

Templates provide reusable query patterns with parameter substitution:

```yaml
top_revenue:
  description: "Top N companies by revenue in a specific year"
  sql: "SELECT ticker, date, totalRevenue FROM financials.annual WHERE YEAR(date) = {year} ORDER BY totalRevenue DESC LIMIT {limit}"
  params:
    - year
    - limit
  defaults:
    limit: 10
```

**Available Templates**:
- `top_revenue`: Top companies by revenue in a year
- `ticker_comparison`: Compare metrics across tickers
- `revenue_trends`: Revenue trends for a ticker
- `profit_margins`: Profit margins for tickers
- `peer_group_comparison`: Compare metrics across peer groups
- `top_roe`: Top companies by Return on Equity
- `dividend_history`: Dividend payment history
- `stock_price_range`: Stock prices in a date range
- `growth_leaders`: Highest revenue growth companies
- `sector_analysis`: Average metrics by sector

**Usage**:

```bash
# List all templates
python query.py --list-templates

# Execute a template
python query.py --template top_revenue --template-params "year=2023,limit=10"

# Complex template with multiple params
python query.py --template ticker_comparison \
  --template-params "metric=netIncome,tickers='AAPL','MSFT','GOOGL',limit=25"
```

**Programmatic Access**:

```python
from resilience import load_query_templates, execute_template

# Load templates
templates = load_query_templates()

# Execute template
params = {"year": 2023, "limit": 10}
columns, rows, sql = execute_template("top_revenue", params, conn)
```

### Graceful Degradation

When Ollama connection fails, users get three options:

**Option 1: Direct SQL Entry (Expert Mode)**:
```
⚠️  Ollama is not reachable. Fallback options:
   1. Enter SQL directly (expert mode)
   2. Use saved query templates
   3. Exit and fix connection

Select [1/2/3]: 1

📝 Expert Mode: Enter your SQL query directly
SQL> SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 5
```

**Option 2: Query Templates**:
```
Select [1/2/3]: 2

📚 Available query templates:
   • top_revenue: Top N companies by revenue in a specific year
   • ticker_comparison: Compare a specific metric across multiple tickers
   [...]

Template name> top_revenue

📋 Template: Top N companies by revenue in a specific year
   Required parameters: year, limit

year> 2023
limit (default: 10)> 5

📊 Generated SQL: SELECT ticker, date, totalRevenue FROM financials.annual WHERE YEAR(date) = 2023 ORDER BY totalRevenue DESC LIMIT 5
```

**Option 3: Exit**:
```
Select [1/2/3]: 3

Exiting. Please check Ollama connection and try again.
```

**Implementation** (`resilience.py:handle_ollama_failure()`):
- Detects `requests.ConnectionError` when calling Ollama
- Presents interactive menu to user
- Returns SQL string or None based on choice
- Integrates with both `query.py` and `chat.py`

### Ticker Validation

**Validation Functions** (`resilience.py`):

```python
def validate_ticker(ticker: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if ticker exists in company.metadata table."""
    result = conn.execute(
        "SELECT COUNT(*) FROM company.metadata WHERE ticker = ?",
        [ticker.upper()]
    ).fetchone()
    return result[0] > 0

def suggest_tickers(partial: str, conn: duckdb.DuckDBPyConnection, limit: int = 10) -> List[str]:
    """Autocomplete ticker symbols."""
    results = conn.execute(
        "SELECT ticker FROM company.metadata WHERE ticker LIKE ? ORDER BY ticker LIMIT ?",
        [f"{partial.upper()}%", limit]
    ).fetchall()
    return [r[0] for r in results]

def get_all_tickers(conn: duckdb.DuckDBPyConnection) -> List[str]:
    """Get all available tickers."""
    results = conn.execute("SELECT DISTINCT ticker FROM company.metadata ORDER BY ticker").fetchall()
    return [r[0] for r in results]
```

**Usage Examples**:

```python
# Validate before querying
if not validate_ticker("AAPL", conn):
    print("Invalid ticker")

# Auto-complete in interactive UI
user_input = "A"
suggestions = suggest_tickers(user_input, conn, limit=5)
print(f"Did you mean: {', '.join(suggestions)}")
# Output: AAPL, AMD, AMZN, ABBV, ABT

# Build ticker picker
all_tickers = get_all_tickers(conn)
print(f"Available tickers: {len(all_tickers)}")
```

### Debug Mode

**Command-Line Usage**:

```bash
# One-shot query with debug
python query.py --debug "Show AAPL revenue for last 3 years"

# Interactive chat with debug
python chat.py --debug
```

**Debug Output Example**:

```
[DEBUG] System Prompt (1245 chars):
You are FinanGPT, a disciplined financial data analyst...
[truncated for brevity]

[DEBUG] User Query: Show AAPL revenue for last 3 years

[DEBUG] LLM Response:
SELECT ticker, date, totalRevenue
FROM financials.annual
WHERE ticker = 'AAPL'
  AND date >= '2022-11-09'
ORDER BY date DESC
LIMIT 3

[DEBUG] Validated SQL:
SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker = 'AAPL' AND date >= '2022-11-09' ORDER BY date DESC LIMIT 3

[DEBUG] Query Time: 0.043s
[DEBUG] Rows Returned: 3
```

**Debug Functions** (`resilience.py`):

```python
def print_debug_info(
    system_prompt: str,
    user_query: str,
    llm_response: str,
    validated_sql: str,
    query_time: float,
    rows: int,
    enabled: bool = False,
) -> None:
    """Print comprehensive debug information."""
    if not enabled:
        return

    print("\n" + "=" * 70)
    print(f"[DEBUG] System Prompt ({len(system_prompt)} chars):")
    print("=" * 70)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    # ... [additional debug output]
```

### Integration with Existing Features

**Phase 6 enhances previous phases**:

1. **Query Pipeline** (`query.py`):
   - Template execution bypasses LLM for known patterns
   - Graceful degradation maintains uptime
   - Debug mode aids troubleshooting

2. **Chat Interface** (`chat.py`):
   - Conversation continues with fallback options
   - Templates provide quick answers
   - Debug mode traces multi-turn context

3. **Error Recovery** (Phase 3):
   - Graceful degradation is first line of defense
   - Auto-retry is second line (existing)
   - Templates provide third option

### Files Added/Modified

**New Files**:
- `resilience.py` (350+ lines): Core Phase 6 functionality
- `templates/queries.yaml`: Query template library
- `tests/test_error_resilience.py` (400+ lines): Comprehensive test suite

**Modified Files**:
- `query.py`: Added `--template`, `--list-templates`, `--debug` flags
- `chat.py`: Added `--debug` flag and graceful degradation
- `requirements.txt`: Added `pyyaml` dependency

### Testing

**Test Coverage** (`tests/test_error_resilience.py`):

```bash
# Run Phase 6 tests
python -m pytest tests/test_error_resilience.py -v

# Test categories:
# - Query template loading and execution (6 tests)
# - Ticker validation and autocomplete (4 tests)
# - Template integration (2 tests)
# - Graceful degradation (2 tests)
# - Debug logging (2 tests)
```

**Key Test Cases**:
- Template loading from YAML
- Template execution with parameter substitution
- Missing parameter error handling
- Ticker validation (exists/not exists)
- Autocomplete suggestions
- Graceful degradation user choices
- Debug logging enabled/disabled

### Usage Patterns

**For End Users**:
1. Templates provide quick access to common queries without LLM
2. Graceful degradation ensures uptime during Ollama outages
3. Debug mode helps understand query failures

**For Developers**:
1. Templates simplify testing and CI/CD
2. Ticker validation prevents common errors
3. Debug mode aids development and troubleshooting

**For System Administrators**:
1. Graceful degradation reduces support burden
2. Templates enable scripting and automation
3. Debug logs facilitate issue diagnosis

### Troubleshooting

**Templates not loading**:
- Check `templates/queries.yaml` exists and is valid YAML
- Install `pyyaml`: `pip install pyyaml`
- Verify file permissions

**Ticker validation fails**:
- Ensure `company.metadata` table exists
- Run `python transform.py` to create/update table
- Check DuckDB database file is accessible

**Debug output not showing**:
- Verify `--debug` flag is passed
- Check that `resilience.py` is importable
- Ensure no stdout redirection is interfering

**Graceful degradation not working**:
- Confirm `resilience.py` is in Python path
- Check for import errors at startup
- Verify Ollama connection failure is actually occurring

## Phase 7: Unified Workflow & Automation

### Overview

Phase 7 introduces a unified command-line interface and configuration management system, simplifying the FinanGPT workflow into a single entry point. This phase also adds system status monitoring and automated refresh capabilities for production deployments.

### Key Features

**Unified CLI (`finangpt.py`)**:
- Single entry point for all FinanGPT operations
- Consistent command interface across all workflows
- Built-in help and usage documentation
- Subprocess management for existing scripts

**Configuration File Support**:
- YAML-based configuration (`config.yaml`)
- Environment variable fallback for flexibility
- Centralized settings management
- Priority: env vars → config file → defaults

**Status Monitoring**:
- System health checks (MongoDB, DuckDB)
- Data freshness statistics
- Ticker count and table row counts
- JSON output for programmatic access

**Automation Support**:
- Scheduled update scripts (daily_refresh.sh)
- Cron-ready with error handling
- Email notifications on failure (optional)
- Logging and status reporting

### Unified CLI Commands

The `finangpt.py` script provides a single entry point for all operations:

```bash
# Ingest data
finangpt.py ingest --tickers AAPL,MSFT
finangpt.py ingest --tickers-file tickers.csv --refresh
finangpt.py ingest --force --tickers AAPL  # Force re-fetch

# Transform data
finangpt.py transform

# Query (one-shot)
finangpt.py query "Show AAPL revenue trends"
finangpt.py query --template top_revenue --template-params "year=2023,limit=10"
finangpt.py query --list-templates

# Interactive chat
finangpt.py chat
finangpt.py chat --debug

# System status
finangpt.py status
finangpt.py status --json  # JSON output

# Full refresh workflow
finangpt.py refresh --tickers-file tickers.csv
finangpt.py refresh --tickers AAPL,MSFT,GOOGL
```

### Configuration File

**File**: `config.yaml`

```yaml
# FinanGPT Configuration File

database:
  mongo_uri: mongodb://localhost:27017/financial_data
  duckdb_path: financial_data.duckdb

ollama:
  url: http://localhost:11434
  model: phi4:latest
  timeout: 60
  max_retries: 3

ingestion:
  price_lookback_days: 365
  auto_refresh_threshold_days: 7
  batch_size: 50
  retry_backoff: [1, 2, 4]

query:
  default_limit: 25
  max_limit: 100
  enable_visualizations: true
  chart_output_dir: charts/
  export_formats: [csv, json, excel]

features:
  conversational_mode: true
  auto_error_recovery: true
  query_suggestions: true
  portfolio_tracking: false

logging:
  level: INFO
  directory: logs/
  format: json
```

**Configuration Priority**:
1. **Environment variables** (highest priority)
   - `MONGO_URI`, `OLLAMA_URL`, `MODEL_NAME`, `PRICE_LOOKBACK_DAYS`
2. **config.yaml file**
   - Located in project root directory
3. **Default values** (fallback)
   - Hard-coded defaults in `config_loader.py`

**Programmatic Access**:

```python
from config_loader import load_config

# Load configuration
config = load_config()

# Access settings
print(config.mongo_uri)
print(config.model_name)
print(config.default_limit)

# Custom config path
config = load_config("/path/to/custom/config.yaml")
```

### Status Command

The status command provides system health monitoring and data freshness information:

**Text Output**:
```bash
$ python finangpt.py status

======================================================================
FinanGPT System Status
======================================================================

Timestamp: 2025-11-09T12:00:00Z

📊 Database Status:
  MongoDB: connected
  DuckDB: connected
  Ticker Count: 150

📅 Data Freshness:
  Average Age: 3.5 days
  Oldest: 7 days
  Newest: 1 days
  Stale Tickers: 10/150 (>7 days)

📁 Table Row Counts:
  financials.annual: 750
  financials.quarterly: 3000
  prices.daily: 54750
  dividends.history: 1200
  splits.history: 45
  company.metadata: 150
  company.peers: 240
  ratios.financial: 750
  user.portfolios: 0

⚙️  Configuration:
  MongoDB: mongodb://localhost:27017/financial_data
  DuckDB: financial_data.duckdb
  Ollama: http://localhost:11434
  Model: phi4:latest

======================================================================
```

**JSON Output**:
```bash
$ python finangpt.py status --json

{
  "timestamp": "2025-11-09T12:00:00Z",
  "database": {
    "mongodb": "connected",
    "duckdb": "connected",
    "ticker_count": 150,
    "table_counts": {
      "financials.annual": 750,
      "financials.quarterly": 3000,
      ...
    }
  },
  "data_freshness": {
    "average_age_days": 3.5,
    "oldest_age_days": 7,
    "newest_age_days": 1,
    "stale_ticker_count": 10,
    "stale_threshold_days": 7
  },
  "configuration": {
    "mongo_uri": "mongodb://localhost:27017/financial_data",
    "ollama_url": "http://localhost:11434",
    "model": "phi4:latest",
    "duckdb_path": "financial_data.duckdb"
  }
}
```

### Full Refresh Workflow

The `refresh` command runs a complete update workflow:

```bash
$ python finangpt.py refresh --tickers-file tickers.csv

======================================================================
Full Refresh Workflow
======================================================================

Step 1: Ingesting data...
Running: python ingest.py --refresh --tickers-file tickers.csv
[ingestion output...]

Step 2: Transforming data...
Running: python transform.py
[transformation output...]

Step 3: Checking status...
[status output...]

✅ Full refresh completed successfully!
```

This command:
1. Runs incremental data ingestion (`--refresh` mode)
2. Transforms data to DuckDB
3. Generates and displays status report

### Scheduled Updates

**Script**: `scripts/daily_refresh.sh`

Automated daily refresh script with error handling and logging:

```bash
#!/bin/bash
# Daily refresh script for FinanGPT
# Runs incremental data refresh and transformation

# Features:
# - Virtual environment activation
# - Error handling with exit on failure
# - Optional email notifications
# - Status report generation
# - Logging to logs/ directory

# Usage:
./scripts/daily_refresh.sh

# Cron example (weekdays at 6 PM):
# 0 18 * * 1-5 /path/to/FinanGPT/scripts/daily_refresh.sh >> /path/to/logs/cron.log 2>&1
```

**Script Features**:
- Activates virtual environment automatically
- Runs incremental refresh (only updates stale data)
- Transforms data to DuckDB
- Generates daily status report (JSON)
- Email notifications on failure (optional)
- Comprehensive error handling
- Colored logging output

**Configuration**:

Edit `scripts/daily_refresh.sh` to customize:
```bash
# Set tickers file location
TICKERS_FILE="$PROJECT_DIR/tickers.csv"

# Enable email notifications on failure
EMAIL_ON_FAILURE="admin@example.com"
```

**Cron Setup**:

```bash
# Edit crontab
crontab -e

# Add daily refresh (weekdays at 6 PM)
0 18 * * 1-5 /path/to/FinanGPT/scripts/daily_refresh.sh >> /path/to/FinanGPT/logs/cron.log 2>&1

# Add weekly full refresh (Sundays at 2 AM)
0 2 * * 0 /path/to/FinanGPT/scripts/daily_refresh.sh >> /path/to/FinanGPT/logs/cron.log 2>&1
```

### Command Reference

**finangpt.py ingest**:
- `--tickers AAPL,MSFT` - Comma-separated ticker list
- `--tickers-file path` - File with tickers (one per line)
- `--refresh` - Smart refresh mode (only stale data)
- `--refresh-days N` - Custom staleness threshold
- `--force` - Force re-fetch all data

**finangpt.py transform**:
- No arguments - transforms all data

**finangpt.py query**:
- `question` - Natural language query
- `--template name` - Use query template
- `--template-params key=val,key=val` - Template parameters
- `--list-templates` - Show available templates
- `--skip-freshness-check` - Skip data freshness check
- `--no-chart` - Disable chart generation
- `--no-formatting` - Disable financial formatting
- `--debug` - Enable debug logging

**finangpt.py chat**:
- `--skip-freshness-check` - Skip data freshness check
- `--debug` - Enable debug logging

**finangpt.py status**:
- `--json` - Output as JSON
- `--config path` - Custom config file path

**finangpt.py refresh**:
- `--tickers AAPL,MSFT` - Ticker list
- `--tickers-file path` - Tickers file
- `--all-data-types` - Refresh all data types (default)
- `--config path` - Custom config file path

### Files Added/Modified

**New Files**:
- `config.yaml` - Default configuration file
- `config_loader.py` - Configuration loader module (200+ lines)
- `finangpt.py` - Unified CLI entry point (400+ lines)
- `scripts/daily_refresh.sh` - Automated refresh script (100+ lines)
- `tests/test_unified_cli.py` - Test suite (200+ lines)

**Modified Files**:
- `CLAUDE.md` - Added Phase 7 documentation
- `README.md` - Updated with unified CLI usage

### Testing

**Test Coverage** (`tests/test_unified_cli.py`):

```bash
# Run Phase 7 tests
python -m pytest tests/test_unified_cli.py -v

# Test categories:
# - Configuration loading (8 tests)
# - Environment variable override (1 test)
# - File fallback handling (2 tests)
# - Status command (2 tests)
```

**Key Test Cases**:
- Configuration loading from file
- Configuration loading from dictionary
- Default value fallback
- Environment variable override
- Invalid YAML handling
- Missing config file handling
- Status command structure
- Status output formatting

### Migration from Separate Scripts

**Before (Phases 1-6)**:
```bash
# Multi-step workflow
python ingest.py --tickers AAPL,MSFT
python transform.py
python query.py "Show revenue trends"
python chat.py
```

**After (Phase 7)**:
```bash
# Unified workflow
python finangpt.py ingest --tickers AAPL,MSFT
python finangpt.py transform
python finangpt.py query "Show revenue trends"
python finangpt.py chat

# Or single refresh command
python finangpt.py refresh --tickers AAPL,MSFT
```

**Both approaches work** - existing scripts remain functional for backward compatibility.

### Troubleshooting

**Config file not found**:
- Place `config.yaml` in project root directory
- Or specify path: `finangpt.py --config /path/to/config.yaml`
- System falls back to defaults if file missing

**Status command shows errors**:
- Check MongoDB is running: `mongod --version`
- Check DuckDB file exists: `ls -lh financial_data.duckdb`
- Verify config.yaml has correct URIs

**Scheduled script fails**:
- Check virtual environment path in script
- Verify cron has correct file paths (use absolute paths)
- Check logs in `logs/cron.log`
- Test script manually first: `./scripts/daily_refresh.sh`

**Import errors**:
- Ensure all Phase 7 files are in project root
- Check `config_loader.py` is importable: `python -c "import config_loader"`
- Verify `pyyaml` is installed: `pip install pyyaml`

## Common Issues

**MongoDB connection fails**:
- Ensure `MONGO_URI` includes database name: `mongodb://localhost:27017/financial_data`
- Unique indexes are created per database, so URI must be complete

**Ollama not reachable**:
- Check `OLLAMA_URL` points to running instance
- Verify model is pulled: `ollama list`

**Schema mismatch**:
- If `query.py` says "No DuckDB tables found", run `transform.py`
- To rebuild schema: delete `financial_data.duckdb` and re-run `transform.py`

**ETF/Non-USD rejection**:
- By design; only US equity tickers with USD statements are allowed
- Filter fails closed to protect downstream data quality

## Development Patterns

**Date Handling**:
- All dates normalized to `US/Eastern 16:00` → UTC before storage
- MongoDB stores ISO 8601 strings, DuckDB uses DATE type
- Use `normalise_reporting_date()` in `ingest.py:438` for consistency

**Error Handling**:
- Custom exceptions: `UnsupportedInstrument`, `StatementDownloadError`
- Exponential backoff: `ESTIMATED_BACKOFF_SECONDS = (1, 2, 4)`
- Failures are logged but don't abort entire batch

**Numeric-Only Columns**:
- `is_numeric()` excludes booleans and NaN/Inf values
- Transform stage filters to numeric types only (no strings, dates, etc.)
- Columns sorted alphabetically after `ticker` and `date`

## Testing Strategy

**Test Coverage** (`tests/`):
- `test_ingest_filters.py`: ETF detection, US listing, USD currency validation
- `test_transform_schema.py`: DataFrame flattening, date parsing, numeric filtering
- `test_query_sql_guardrails.py`: SQL injection prevention, table allow-list, LIMIT enforcement

**Run individual test functions**:
```bash
python -m pytest tests/test_query_sql_guardrails.py::test_rejects_drop_table -v
```

## LLM Query Prompt Architecture

The system prompt is **dynamically generated** from DuckDB schema (query.py:67-86):
- Introspects tables at runtime to capture current columns
- Provides examples in prompt: "annual net income for AAPL"
- Enforces rules: "Always project date", "Default LIMIT 25", "Read-only"

When schema changes (new fields added), no code changes needed—prompt auto-updates.

## Example Natural Language Queries

With the enhanced data sources, you can now ask:

**Financial Analysis**:
- "Show AAPL's profit margins over the last 5 years" → Uses `ratios.financial`
- "Which companies have the highest ROE in the tech sector?" → Joins `ratios.financial` + `company.metadata`
- "Compare MSFT and GOOGL revenue growth year-over-year" → Uses `growth.annual`

**Price & Market Data**:
- "What was TSLA's closing price on 2024-01-15?" → Uses `prices.daily`
- "Show AAPL's stock performance for the last quarter" → Uses `prices.daily` with date filters
- "Which stocks have paid dividends in 2024?" → Uses `dividends.history`

**Company Information**:
- "List all companies in the semiconductors industry" → Uses `company.metadata`
- "What is Microsoft's market cap?" → Uses `company.metadata`
- "Show me all stocks in the Technology sector" → Uses `company.metadata`

**Combined Analysis**:
- "Show companies with ROE > 20% and dividend yield > 2%" → Joins `ratios.financial` + `dividends.history` + `prices.daily`
- "Find stocks with positive revenue growth and negative debt ratio changes" → Joins `growth.annual` + `ratios.financial`

## Extension Points

**Adding More Derived Ratios**:
- Edit `create_ratios_table()` in `transform.py:254`
- Add new CASE WHEN calculations (e.g., P/E ratio with price data)
- Re-run `transform.py` to rebuild the table

**Expanding Field Coverage**:
- Update `FIELD_MAPPINGS` in `ingest.py:45-101` to handle new yfinance field names
- No schema migration needed; just re-run `ingest.py` + `transform.py`

**Alternative LLM Providers**:
- Replace `call_ollama()` in `query.py:89` with OpenAI/Anthropic client
- Keep `validate_sql()` guardrails intact regardless of provider

**Data Freshness Monitoring**:
- Query `ingestion_metadata` collection in MongoDB to check last fetch times
- Filter by `data_type` to see which data sources need refreshing
- Use `last_successful_date` to identify gaps in historical data
