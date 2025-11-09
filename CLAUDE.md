# CLAUDE.md

**Project**: FinanGPT - AI-Powered Financial Data Analysis Platform
**Status**: Production-ready (Phase 9 implemented)
**Version**: 2.2 (Updated 2025-11-09)

## Quick Reference

**Tech Stack**: Python 3.x | MongoDB | DuckDB | Ollama (LLM) | yfinance
**Lines of Code**: ~5,700 Python | 11 core modules
**Data**: US equities only (non-ETF, USD-denominated)
**Latest**: Phase 9 - Analyst intelligence (recommendations, price targets, consensus)

### Core Workflows

```bash
# Unified CLI (Recommended)
python finangpt.py ingest --tickers-file tickers.csv --refresh
python finangpt.py transform
python finangpt.py query "Show AAPL revenue trends"
python finangpt.py chat
python finangpt.py status

# Legacy commands (still supported)
python ingest.py --tickers AAPL,MSFT,GOOGL
python transform.py
python query.py "your question here"
python chat.py
```

### Environment Setup

```bash
# Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Dependencies
pip install -r requirements.txt
# yfinance pandas duckdb pymongo requests python-dotenv pyyaml
# matplotlib mplfinance openpyxl pytest

# Configuration (.env)
MONGO_URI=mongodb://localhost:27017/financial_data
OLLAMA_URL=http://localhost:11434
MODEL_NAME=gpt-oss:latest
PRICE_LOOKBACK_DAYS=365

# Services
mongod  # MongoDB server
ollama pull gpt-oss:latest
```

---

## Architecture Overview

### Three-Stage Pipeline

**1. Ingestion** (`ingest.py` - 934 lines)
- Fetches from yfinance API with exponential backoff (1, 2, 4 sec)
- **Validation**: US-listed, non-ETF, USD currency (fail-closed)
- **Data types**: Financials (annual/quarterly), prices (OHLCV), dividends, splits, metadata
- **Smart caching**: Incremental updates, freshness tracking
- **Storage**: MongoDB (7 collections with compound indexes)

**2. Transformation** (`transform.py` - 502 lines)
- Flattens MongoDB documents → DuckDB analytics tables
- **Derives**: 9 financial ratios (ROE, ROA, margins), YoY growth view
- **Creates**: Peer groups table (16 groups), portfolio tracking table
- **Idempotent**: Delete-then-insert on `{ticker, date}`

**3. Query** (`query.py` - 691 lines | `chat.py` - 466 lines)
- **LLM-to-SQL**: Dynamic schema introspection → system prompt
- **Validation**: Table allow-list, column existence, SELECT-only, LIMIT ≤ 100
- **Features**: Freshness checks, auto-visualization, financial formatting, templates

### Database Schemas

**MongoDB Collections** (13):
```
raw_annual, raw_quarterly         # Financial statements
stock_prices_daily                # OHLCV data
dividends_history, splits_history # Corporate actions
company_metadata                  # Company info
ingestion_metadata                # Freshness tracking
earnings_history, earnings_calendar  # Phase 8: Earnings data
analyst_recommendations           # Phase 9: Analyst upgrades/downgrades
price_targets                     # Phase 9: Price target consensus
analyst_consensus                 # Phase 9: Buy/hold/sell ratings
growth_estimates                  # Phase 9: Growth forecasts
```

**DuckDB Tables** (17 tables/views):
```
financials.annual / financials.quarterly  # Statements
prices.daily                              # OHLCV
dividends.history / splits.history        # Corporate actions
company.metadata / company.peers          # Company info + peer groups
ratios.financial                          # 9 derived ratios
growth.annual (VIEW)                      # YoY growth
user.portfolios                           # Portfolio tracking
valuation.metrics                         # Phase 8: Valuation ratios
earnings.history / earnings.calendar      # Phase 8: Earnings data
analyst.recommendations                   # Phase 9: Analyst ratings
analyst.price_targets                     # Phase 9: Price targets
analyst.consensus                         # Phase 9: Rating consensus
analyst.growth_estimates                  # Phase 9: Growth forecasts
```

### Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `ingest.py` | 1445 | Data fetching, validation, MongoDB storage |
| `transform.py` | 815 | MongoDB → DuckDB, derived metrics |
| `query.py` | 730 | One-shot NL queries, SQL generation |
| `chat.py` | 466 | Conversational interface (20-msg history) |
| `visualize.py` | 463 | Charts (4 types), financial formatting, exports |
| `resilience.py` | 306 | Graceful degradation, templates, validation |
| `finangpt.py` | 420 | Unified CLI, status monitoring |
| `config_loader.py` | 204 | YAML config + env var fallback |
| `peer_groups.py` | 79 | 16 predefined peer groups |
| `valuation.py` | 230 | Phase 8: Valuation metrics & earnings tables |
| `analyst.py` | 250 | Phase 9: Analyst intelligence tables |

---

## Feature Capabilities

### Phase 1: Rich Data Sources ✅
- 5 data types ingested (financials, prices, dividends, splits, metadata)
- 9 derived financial ratios (ROE, ROA, net_margin, gross_margin, ebitda_margin, fcf_margin, debt_ratio, cash_conversion, asset_turnover)
- YoY growth view (revenue/income trends)

### Phase 2: Smart Caching ✅
- **3 modes**: Normal (full fetch), Refresh (incremental), Force (re-fetch)
- Freshness tracking per ticker+data_type
- Incremental price updates (10-100x faster)
- Pre-query staleness warnings (7-day threshold)

### Phase 3: Conversational Interface ✅
- Multi-turn context (20-message rolling window)
- Auto-retry with LLM feedback (3 attempts)
- Special commands: `/help`, `/clear`, `/exit`

### Phase 4: Visual Analytics ✅
- Auto-detection: line, bar, scatter, candlestick charts
- Financial formatting: $394.00B, 23.76%, $150.25
- Exports: CSV, JSON, Excel

### Phase 5: Advanced Queries ✅
- 16 peer groups (FAANG, Semiconductors, Magnificent Seven, etc.)
- Natural language dates ("last 5 years", "YTD", "2023")
- Window functions (RANK, LAG, LEAD, NTILE)
- Portfolio tracking table

### Phase 6: Error Resilience ✅
- Graceful degradation (Ollama unavailable → direct SQL or templates)
- 10 query templates with parameter substitution
- Ticker validation + autocomplete
- Debug mode (system prompt, SQL, timing)

### Phase 7: Unified Workflow ✅
- Single CLI entry point (`finangpt.py`)
- YAML config + env var priority (env > config > defaults)
- Status monitoring (health checks, freshness stats, JSON output)
- Automated refresh script (`scripts/daily_refresh.sh` - cron-ready)

### Phase 8: Valuation Metrics & Earnings Intelligence ✅
- **Valuation metrics**: P/E, P/B, P/S, PEG ratios calculated from latest prices + financials
- **Dividend metrics**: Dividend yield, payout ratio (annual basis)
- **Market cap classification**: Large Cap (>$10B), Mid Cap ($2-10B), Small Cap (<$2B)
- **Earnings history**: EPS estimates vs actuals, earnings surprises ($ and %)
- **Earnings calendar**: Upcoming earnings dates with estimates
- **Auto-integration**: Earnings data fetched during standard ingestion flow
- **New tables**: `valuation.metrics`, `earnings.history`, `earnings.calendar`

### Phase 9: Analyst Intelligence & Sentiment ✅
- **Analyst recommendations**: Historical upgrades/downgrades with firm tracking and action scores
- **Price targets**: Consensus low/mean/high targets with upside/downside calculations
- **Analyst consensus**: Buy/hold/sell distributions with weighted rating scores (1-5 scale)
- **Growth estimates**: Quarterly, annual, and 5-year growth forecasts from analyst consensus
- **Forward PEG**: Automatically calculated using growth estimates and current valuations
- **Auto-integration**: Analyst data fetched during standard ingestion flow
- **New tables**: `analyst.recommendations`, `analyst.price_targets`, `analyst.consensus`, `analyst.growth_estimates`

---

## Critical Implementation Details

### Validation & Security

**Ingestion Filters** (fail-closed):
```python
is_etf()          # Rejects via quoteType, fundFamily, name patterns
is_us_listing()   # country="United States" OR market="us_market"
has_usd_financials() # financialCurrency or currency == "USD"
```

**SQL Guardrails**:
- SELECT-only (WITH/CTEs allowed if final statement is SELECT)
- Table allow-list: 18 tables enforced (Phase 9: +4 analyst tables)
- Column existence verified (including CTEs)
- LIMIT ≤ 100 (default: 25)

### Data Normalization

**Date Handling**:
- All dates normalized: `US/Eastern 16:00 → UTC`
- MongoDB: ISO 8601 strings
- DuckDB: DATE type
- Function: `normalise_reporting_date()` in `ingest.py:438`

**Field Mappings**:
- `FIELD_MAPPINGS` dict (lines 45-101 in `ingest.py`) handles yfinance variations
- Example: `"netIncome" → ["Net Income", "NetIncome", "Net Income Applicable To Common Shares"]`

**Numeric Filtering**:
- `is_numeric()` excludes booleans, NaN, Inf
- Columns sorted alphabetically after `ticker`, `date`

### Configuration Priority

```
1. Environment variables (highest)
   ↓
2. config.yaml file
   ↓
3. Hard-coded defaults (fallback)
```

### LLM System Prompt

**Dynamically generated** from DuckDB schema introspection:
- Runtime column discovery
- Auto-updates when schema changes
- Includes date context, peer groups, window functions
- Example queries and rules embedded

---

## Common Operations

### Data Refresh Workflows

**Daily incremental refresh** (recommended):
```bash
python finangpt.py refresh --tickers-file tickers.csv
# or via cron:
0 18 * * 1-5 /path/to/.venv/bin/python /path/to/finangpt.py refresh --tickers-file tickers.csv
```

**Force re-fetch** (when data quality issues suspected):
```bash
python ingest.py --force --tickers AAPL,MSFT
python transform.py
```

### Query Templates

**List available templates**:
```bash
python query.py --list-templates
```

**Execute template**:
```bash
python query.py --template top_revenue --template-params "year=2023,limit=10"
```

**10 built-in templates**:
- `top_revenue`, `ticker_comparison`, `revenue_trends`, `profit_margins`
- `peer_group_comparison`, `top_roe`, `dividend_history`
- `stock_price_range`, `growth_leaders`, `sector_analysis`

### Debug Mode

```bash
python query.py --debug "your question"
python chat.py --debug
```

**Output**: System prompt, LLM response, validated SQL, query time, row count

---

## Testing

**10 test suites** covering all phases:

```bash
pytest tests/                              # Run all tests
pytest tests/test_ingest_filters.py -v     # Ingestion validation
pytest tests/test_query_sql_guardrails.py -v # SQL security
pytest tests/test_visualizations.py -v     # Charts + formatting
pytest tests/test_error_resilience.py -v   # Templates + degradation
pytest tests/test_valuation.py -v          # Phase 8: Valuation & earnings
```

**Coverage**: ETF detection, SQL injection prevention, freshness tracking, chart detection, template execution, config loading, valuation calculations, earnings data transformation

---

## Example Queries

### Financial Analysis
```
"Show AAPL's profit margins over the last 5 years"
"Which companies have the highest ROE in the tech sector?"
"Compare MSFT and GOOGL revenue growth year-over-year"
```

### Peer Group Analysis
```
"Compare FAANG companies by market cap"
"Rank semiconductor companies by revenue growth"
"Show average profit margins for Cloud Computing companies"
```

### Advanced Analytics
```
"Show top 10 companies by revenue with their rank"
"Calculate 3-period moving average for AAPL revenue"
"Find companies in the top quartile by ROE"
```

### Portfolio Tracking
```
"What's my Tech Growth portfolio worth today?"
"Show unrealized gains for my holdings"
"Calculate portfolio allocation by sector"
```

### Valuation Analysis (Phase 8)
```
"Find undervalued tech stocks with P/E < 15 and P/B < 2"
"Show me high dividend yield stocks (>4%) with strong payout ratios"
"Compare valuation multiples for FAANG companies"
"Which large-cap stocks have PEG ratio < 1?" (growth at reasonable price)
"Show stocks with P/S ratio below their sector median"
```

### Earnings Intelligence (Phase 8)
```
"Which companies beat earnings estimates last quarter?"
"Show AAPL's earnings surprise history for last 2 years"
"Find stocks that consistently beat estimates (>75% of time)"
"When is the next earnings call for TSLA?"
"Show upcoming earnings this week"
"Find companies with improving earnings surprise trends"
```

### Analyst Intelligence (Phase 9)
```
"Show me stocks with recent analyst upgrades"
"Which stocks have the highest upside to price targets?"
"Find stocks rated 'Strong Buy' with upside > 15%"
"Show analyst consensus for FAANG companies"
"Which analysts are most bullish on tech stocks?"
"Companies with 5-year growth estimates > 20%"
"Stocks where analysts raised price targets this month"
"Find downgrades from major investment banks"
```

---

## Extension Points

### Adding Derived Metrics
```python
# transform.py:254 - create_ratios_table()
# Add new CASE WHEN calculations
# Re-run transform.py to rebuild
```

### Expanding Field Coverage
```python
# ingest.py:45-101 - FIELD_MAPPINGS dict
# Add new field name variations
# No schema migration needed
```

### Alternative LLM Providers
```python
# query.py:89 - call_ollama()
# Replace with OpenAI/Anthropic client
# Keep validate_sql() guardrails intact
```

### Custom Peer Groups
```python
# peer_groups.py
# Add new groups to PEER_GROUPS dict
# Re-run transform.py to populate company.peers
```

---

## Troubleshooting

### Common Issues

**MongoDB connection fails**:
- Ensure `MONGO_URI` includes database name: `mongodb://localhost:27017/financial_data`

**Ollama not reachable**:
- Check `ollama list` shows pulled model
- Verify `OLLAMA_URL` points to running instance

**Schema mismatch**:
- Delete `financial_data.duckdb` and re-run `transform.py`

**ETF/Non-USD rejection**:
- By design (data quality protection)
- Only US equity tickers with USD statements allowed

### Monitoring Data Freshness

**MongoDB query**:
```javascript
use financial_data
db.ingestion_metadata.find({"ticker": "AAPL"}).sort({"last_fetched": -1})
```

**System status**:
```bash
python finangpt.py status --json
```

### Debug Logging

All scripts log to `logs/` with UTC timestamps:
```json
{
  "ts": "2025-11-09T10:30:00Z",
  "phase": "ingest.annual",
  "ticker": "AAPL",
  "attempts": 1,
  "rows": 5,
  "duration_ms": 1234
}
```

---

## Performance Characteristics

- **Ingestion**: 50 tickers/batch, 1-4 sec retry backoff
- **Incremental refresh**: 10-100x faster than full fetch
- **Query latency**: <100ms (DuckDB), 1-5s (LLM generation)
- **Chart generation**: 100-200ms overhead
- **Formatting**: <10ms (negligible impact)

---

## Development Guidelines

### Code Patterns

**Error Handling**:
- Custom exceptions: `UnsupportedInstrument`, `StatementDownloadError`
- Exponential backoff: (1, 2, 4) seconds
- Failures logged but don't abort batch

**Validation Philosophy**:
- Fail closed (reject on missing metadata)
- No guessing/defaulting
- Strict schema enforcement

**Date Consistency**:
- Always use `normalise_reporting_date()`
- UTC everywhere
- ISO 8601 strings in MongoDB

### Testing Strategy

- **Unit tests**: Validation logic, SQL parsing
- **Integration tests**: End-to-end workflows
- **Coverage target**: >80% for critical paths
- **CI/CD**: pytest via GitHub Actions (optional)

---

## Configuration Reference

### config.yaml Structure

```yaml
database:
  mongo_uri: mongodb://localhost:27017/financial_data
  duckdb_path: financial_data.duckdb

ollama:
  url: http://localhost:11434
  model: gpt-oss:latest
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

### Environment Variables

```bash
MONGO_URI                # Override database.mongo_uri
OLLAMA_URL               # Override ollama.url
MODEL_NAME               # Override ollama.model
PRICE_LOOKBACK_DAYS      # Override ingestion.price_lookback_days
```

---

## Project Status

**Production-Ready**: Phase 9 implemented (Analyst Intelligence & Sentiment)

**Phase 9 Completed** (2025-11-09):
- ✅ Analyst recommendations (upgrades/downgrades with firm tracking)
- ✅ Price targets (consensus with upside/downside calculations)
- ✅ Analyst consensus (buy/hold/sell distributions, weighted scores)
- ✅ Growth estimates (quarterly, annual, 5-year forecasts)
- ✅ Forward PEG calculations (integrated with valuation metrics)
- ✅ Auto-integration with existing ingestion pipeline
- ✅ 4 new DuckDB tables/views, 4 new MongoDB collections

**Next Steps** (potential Phase 10+):
- Technical analysis (moving averages, RSI, MACD, momentum indicators)
- Query intelligence (decomposition, smart errors, autocomplete)
- Real-time data feeds (WebSocket integration)
- Web dashboard (React frontend + FastAPI backend)
- Multi-user authentication (user management)
- Advanced backtesting (portfolio simulation)
- API endpoints (REST/GraphQL)
- Cloud deployment (Docker + Kubernetes)

**Maintainability**:
- Modular architecture (11 independent modules)
- Clear separation of concerns (ingestion | transformation | query)
- Backward compatibility maintained (legacy scripts supported)

---

## Key Files Reference

### Core Modules
- `ingest.py` - Data fetching and validation
- `transform.py` - Analytics table generation
- `query.py` - One-shot NL queries
- `chat.py` - Conversational interface
- `visualize.py` - Charts and formatting
- `resilience.py` - Error handling and templates
- `finangpt.py` - Unified CLI
- `config_loader.py` - Configuration management
- `peer_groups.py` - Peer group definitions
- `valuation.py` - Phase 8: Valuation metrics & earnings tables

### Configuration
- `config.yaml` - Settings (env var override)
- `.env` - Secrets (MongoDB URI, Ollama URL)
- `requirements.txt` - Python dependencies

### Templates & Scripts
- `templates/queries.yaml` - Query templates
- `scripts/daily_refresh.sh` - Automated refresh

### Tests
- `tests/test_ingest_filters.py` - Validation logic
- `tests/test_transform_schema.py` - Data transformation
- `tests/test_query_sql_guardrails.py` - SQL security
- `tests/test_freshness_tracking.py` - Smart caching
- `tests/test_conversational_chat.py` - Chat interface
- `tests/test_visualizations.py` - Charts + formatting
- `tests/test_valuation.py` - Phase 8: Valuation & earnings
- `tests/test_advanced_queries.py` - Peer groups + window functions
- `tests/test_error_resilience.py` - Templates + degradation
- `tests/test_unified_cli.py` - Config + status

---

## Contact & Support

**Documentation**: This file (CLAUDE.md) + README.md
**Issues**: Check logs/ directory for error details
**Testing**: Run `pytest tests/ -v` for diagnostics

**Quick Health Check**:
```bash
python finangpt.py status
```

---

*Last Updated: 2025-11-09 | Version 2.2 | Phase 9 Complete: Analyst Intelligence & Sentiment*
