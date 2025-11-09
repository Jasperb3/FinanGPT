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
