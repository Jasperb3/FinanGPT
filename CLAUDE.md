# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinanGPT is a Python-based financial data pipeline that:
- Ingests US-only stock financials from yfinance (annual & quarterly)
- Stores raw data in MongoDB with strict validation (non-ETF, USD-only, US-listed)
- Transforms data into DuckDB for analytics
- Provides a natural language query interface via LLM→SQL using Ollama

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
   - Fetches statements from yfinance with exponential backoff (3 attempts)
   - Validates: US-listed, non-ETF, USD currency
   - Normalizes dates to UTC (16:00 US/Eastern)
   - Merges Income Statement + Balance Sheet + Cash Flow per reporting period
   - Upserts to MongoDB `raw_annual` / `raw_quarterly` collections
   - Compound index: `{ticker: 1, date: 1}` (unique)

2. **Transformation (`transform.py`)**:
   - Reads from MongoDB raw collections
   - Flattens nested payloads into numeric-only columns
   - Loads into DuckDB tables: `financials.annual` / `financials.quarterly`
   - Idempotent via delete-then-insert on `{ticker, date}`

3. **Query (`query.py`)**:
   - Introspects DuckDB schema to build dynamic system prompt
   - Calls Ollama `/api/chat` endpoint with schema-aware prompt
   - Validates SQL: table allow-list, column verification, single SELECT only
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
- Table allow-list: `financials.annual`, `financials.quarterly`
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

**DuckDB Schema** (`financials.annual` / `financials.quarterly`):
```sql
ticker VARCHAR
date DATE
netIncome DOUBLE
totalAssets DOUBLE
totalLiabilities DOUBLE
operatingCashFlow DOUBLE
totalRevenue DOUBLE
-- [additional numeric columns sorted alphabetically]
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

## Extension Points

**Adding Derived Ratios**:
- Option 1: Compute in `transform.py` before DuckDB insert
- Option 2: Define as virtual columns in DuckDB schema
- Option 3: Let LLM compute at query time (e.g., `netIncome / totalRevenue AS margin`)

**Expanding Field Coverage**:
- Update `FIELD_MAPPINGS` in `ingest.py` to handle new yfinance field names
- No schema migration needed; just re-run `ingest.py` + `transform.py`

**Alternative LLM Providers**:
- Replace `call_ollama()` in `query.py:89` with OpenAI/Anthropic client
- Keep `validate_sql()` guardrails intact regardless of provider
