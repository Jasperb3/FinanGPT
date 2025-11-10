# CLAUDE.md

**Project**: FinanGPT - AI-Powered Financial Data Analysis Platform
**Status**: Production-ready (Phase 11 + Enhancement Plan 4 Phases 0-2)
**Version**: 2.6 (Updated 2025-11-10)

## Quick Reference

**Tech Stack**: Python 3.x | MongoDB | DuckDB | Ollama (LLM) | yfinance | SQLite
**Lines of Code**: ~8,700 Python | 20 core modules
**Data**: Global markets (12+ currencies, auto-normalized to USD) + US equities
**Latest**: Phase 11 + Enhancement Plan 4 (Production Hardening, Performance & Ollama Reliability)

### Enhancement Plan 4 - Phases 0-2 (NEW)

**Phase 0: Emergency Fixes** ✅
- Fixed duplicate hint augmentation bug in `query.py:844-845`
- Added Ollama health check (`check_ollama_health()`) to `query.py` and `chat.py`
- Implemented retry logic with exponential backoff (3 attempts: 1s, 2s, 4s delays)

**Phase 1: Performance Integration** ✅
- Integrated concurrent ingestion into `ingest.py` (10x speedup when enabled via config)
- Integrated query result caching into `query.py` (1000x speedup for repeated queries)
- Added `use_concurrent: true` flag in `config.yaml` for parallel ticker processing
- Added `cache_enabled: true` flag for LRU cache with TTL (300s default)

**Phase 2: Ollama Reliability Improvements** ✅
- **Exception hierarchy**: 5 custom exception types (`OllamaError`, `OllamaConnectionError`, `OllamaTimeoutError`, `OllamaResponseError`, `SQLExtractionError`)
- **Improved SQL extraction**: 4 fallback strategies (sql code block → generic code block → SELECT anywhere → WITH clause)
- **Semantic SQL validation**: 6 mismatch checks (aggregation, time range, comparison, ranking, count, growth/change)
- **Context window management**: Token estimation and smart trimming (4000 token limit, preserves recent 5 messages)
- **Rate limiting**: Token bucket algorithm with configurable limits (10 req/60s default)
- **Schema refresh detection**: Automatic cache invalidation on schema changes via SHA-256 hashing

**Configuration** (`config.yaml`):
```yaml
ingestion:
  use_concurrent: true      # Enable 10x faster parallel ingestion
  max_workers: 10
  worker_timeout: 120

query:
  cache_enabled: true       # Enable 1000x faster repeated queries
  cache_ttl_seconds: 300
  cache_max_entries: 100

ollama:
  rate_limit_requests: 10   # NEW Phase 2: Rate limiting
  rate_limit_window: 60
  max_context_tokens: 4000  # NEW Phase 2: Context window management
```

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

**DuckDB Tables** (18 tables/views):
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
technical.indicators                      # Phase 10: Technical analysis
```

### Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `ingest.py` | 1445 | Data fetching, validation, MongoDB storage |
| `transform.py` | 815 | MongoDB → DuckDB, derived metrics |
| `query.py` | 730 | One-shot NL queries, SQL generation |
| `chat.py` | 500 | Conversational interface with history commands |
| `visualize.py` | 463 | Charts (4 types), financial formatting, exports |
| `resilience.py` | 306 | Graceful degradation, templates, validation |
| `finangpt.py` | 420 | Unified CLI, status monitoring |
| `config_loader.py` | 204 | YAML config + env var fallback |
| `peer_groups.py` | 79 | 16 predefined peer groups |
| `valuation.py` | 230 | Phase 8: Valuation metrics & earnings tables |
| `analyst.py` | 250 | Phase 9: Analyst intelligence tables |
| `technical.py` | 175 | Phase 10: Technical indicators calculation |
| `query_history.py` | 380 | Phase 11: Query history & favorites |
| `error_handler.py` | 260 | Phase 11: Smart error messages |
| `autocomplete.py` | 320 | Phase 11: Suggestions engine |
| `date_parser.py` | 280 | Phase 11: Enhanced date parsing |
| `query_planner.py` | 360 | Phase 11: Query decomposition |
| **Enhancement Plan 3 - Phase 1** | | **Performance Optimizations** |
| `src/transform/streaming.py` | 245 | Memory-efficient chunked transformation |
| `src/ingest/concurrent.py` | 280 | Parallel ticker ingestion (10x speedup) |
| `src/query/cache.py` | 320 | LRU cache with TTL for queries |
| `src/query/validation.py` | 220 | Pre-compiled regex for SQL validation |
| `src/utils/progress.py` | 155 | Progress indicators with tqdm |
| **Enhancement Plan 3 - Phase 2** | | **Global Market Support** |
| `src/ingest/validators.py` | 360 | Flexible market configuration (global/us_only/custom) |
| `src/data/currency.py` | 390 | Currency conversion with FX rate caching |
| `src/data/valuation_multicurrency.py` | 280 | Multi-currency valuation metrics |

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

### Phase 10: Technical Analysis & Price Momentum ✅
- **Moving averages**: SMA (20/50/200-day) and EMA (12/26-day) for trend identification
- **RSI indicator**: 14-day RSI (0-100 scale) for overbought/oversold detection
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: 20-day bands with 2 standard deviations for volatility analysis
- **Volume metrics**: 20-day average volume and volume ratio calculations
- **Price momentum**: Percentage changes over 1d, 5d, 20d, 60d, and 252d periods
- **52-week analysis**: High/low tracking with percentage distance from current price
- **Zero data fetching**: All indicators calculated from existing prices.daily using window functions
- **New table**: `technical.indicators`

### Phase 11: Query Intelligence & UX Enhancement ✅
- **Query history**: Save all queries to SQLite with SQL, row counts, execution metadata
- **Favorites**: Star/unstar queries for quick access (`/favorite <id>` command)
- **Smart error handler**: Context-aware error messages with table/column suggestions
- **Enhanced date parsing**: Natural language dates (YTD, Q4 2024, last quarter, fiscal year)
- **Autocomplete engine**: Ticker suggestions with company names, query template suggestions
- **Query decomposition**: Break complex multi-part queries into sequential steps
- **New commands**: `/history`, `/favorites`, `/recall <id>`, `/favorite <id>`, `/search <term>`
- **New modules**: 5 (query_history, error_handler, autocomplete, date_parser, query_planner)

### Enhancement Plan 3 - Phase 1: Performance Optimizations ✅
- **Streaming transformation**: Chunked MongoDB → DuckDB processing (90% memory reduction)
- **Concurrent ingestion**: ThreadPoolExecutor-based parallel processing (10x speedup)
- **Query result caching**: LRU cache with TTL for 100-1000x faster repeated queries
- **Pre-compiled regex**: Module-level pattern compilation for validation performance
- **Progress indicators**: tqdm-based real-time feedback for long operations
- **Configuration updates**: Increased batch sizes, memory limits, cache settings
- **New modules**: 5 (streaming, concurrent, cache, validation, progress)
- **Test coverage**: 27/27 tests passing (100%)

### Enhancement Plan 3 - Phase 2: Global Market Support ✅
- **Global market support**: Accept stocks from any country (EU, Asia, Americas, emerging markets)
- **Multi-currency conversion**: 12+ major currencies (USD, EUR, GBP, JPY, CNY, CAD, AUD, CHF, HKD, SGD, KRW, INR)
- **FX rate caching**: Historical exchange rates cached in DuckDB (1000x+ speedup)
- **Currency normalization**: All metrics auto-converted to base currency (USD)
- **Flexible configuration**: 4 modes (global, us_only, eu_only, custom)
- **Market validators**: Configurable country/currency/exchange filtering
- **Multi-currency valuation**: Currency-neutral ratios (P/E, P/B, P/S) with FX transparency
- **New modules**: 3 (validators, currency, valuation_multicurrency)
- **Test coverage**: 18/18 tests passing (100%)
- **Backward compatible**: Existing US-only databases work without changes

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
- Table allow-list: 19 tables enforced (Phase 10: +1 technical table)
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

### Technical Analysis (Phase 10)
```
"Show AAPL's RSI and MACD indicators for the last month"
"Find stocks with RSI < 30 (oversold opportunities)"
"Which stocks are trading above their 200-day moving average?"
"Show me stocks where price crossed above SMA-50 recently"
"Find stocks with MACD bullish crossover (MACD > signal)"
"Show Bollinger Band squeeze patterns (low volatility)"
"Which stocks are near their 52-week high?"
"Compare moving averages for FAANG stocks"
"Find high-momentum stocks (20d % change > 10%)"
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

**Production-Ready**: Phase 11 implemented (Query Intelligence & UX Enhancement)

**Phase 11 Completed** (2025-11-09):
- ✅ Query history with SQLite persistence
- ✅ Favorites system with star/unstar functionality
- ✅ Smart error handler with contextual suggestions
- ✅ Enhanced date parser (YTD, quarters, fiscal years, relative dates)
- ✅ Autocomplete engine for tickers and query templates
- ✅ Query planner for complex query decomposition
- ✅ 5 new modules, 5 new chat commands
- ✅ Backward compatible (Phase 11 features optional)

**Next Steps** (potential Phase 12+):
- Web dashboard (React frontend + FastAPI backend)
- Real-time data feeds (WebSocket integration)
- Multi-user authentication (user management)
- Advanced backtesting (portfolio simulation)
- API endpoints (REST/GraphQL)
- Cloud deployment (Docker + Kubernetes)

**Maintainability**:
- Modular architecture (17 independent modules)
- Clear separation of concerns (ingestion | transformation | query | intelligence)
- Backward compatibility maintained (all Phase 11 features are opt-in)

---

## Key Files Reference

### Core Modules
- `ingest.py` - Data fetching and validation
- `transform.py` - Analytics table generation
- `query.py` - One-shot NL queries
- `chat.py` - Conversational interface with history commands
- `visualize.py` - Charts and formatting
- `resilience.py` - Error handling and templates
- `finangpt.py` - Unified CLI
- `config_loader.py` - Configuration management
- `peer_groups.py` - Peer group definitions
- `valuation.py` - Phase 8: Valuation metrics & earnings tables
- `analyst.py` - Phase 9: Analyst intelligence tables
- `technical.py` - Phase 10: Technical indicators calculation
- `query_history.py` - Phase 11: Query history & favorites
- `error_handler.py` - Phase 11: Smart error messages
- `autocomplete.py` - Phase 11: Suggestions engine
- `date_parser.py` - Phase 11: Enhanced date parsing
- `query_planner.py` - Phase 11: Query decomposition

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

*Last Updated: 2025-11-09 | Version 2.4 | Phase 11 Complete: Query Intelligence & UX Enhancement*
