# FinanGPT Enhancement Implementation Plan

**Date**: 2025-11-09
**Version**: 1.0
**Status**: Awaiting Sign-Off

---

## Executive Summary

This plan transforms FinanGPT from a basic LLM→SQL pipeline into a comprehensive, interactive financial analysis platform. The enhancements focus on:
- **Rich data ingestion** (price history, dividends, analyst estimates, company info)
- **Smart caching & incremental updates** for quick data refresh
- **Conversational query interface** with multi-turn context
- **Visual analytics** (charts, ratios, peer comparisons)
- **Robust error recovery** with intelligent fallbacks
- **Expanded query capabilities** without SQL guardrail violations

## Current Limitations Identified

### 1. Data Coverage Gaps
- Only ingests financial statements (Income, Balance, Cash Flow)
- Missing: stock prices, dividends, splits, analyst ratings, company metadata
- No derived financial ratios (P/E, ROE, ROA, Debt/Equity, margins)
- Cannot answer: "What's AAPL's current stock price?" or "Show dividend history"

### 2. User Experience Friction
- One-shot queries only (no conversation memory)
- Text-only output (no charts/graphs)
- Manual ingestion required for updates
- Generic error messages when LLM generates invalid SQL
- No query suggestions or autocomplete

### 3. Query Flexibility Constraints
- LIMIT 100 cap may be too restrictive for peer analysis (e.g., "Compare all tech stocks")
- No support for cross-ticker aggregations or rankings
- Cannot handle time-series calculations (YoY growth, moving averages)
- No multi-ticker portfolio analysis

### 4. Operational Pain Points
- Separate ingest → transform → query workflow
- No freshness tracking (can't tell if data is stale)
- Full table replace on transform (inefficient for incremental updates)
- No data validation after ingestion (corruption detection)

---

## Proposed Enhancements

## Phase 1: Rich Data Ingestion & Schema Expansion

### 1.1 Additional yfinance Data Sources

**New Collections/Tables**:
```python
# MongoDB Collections
- stock_prices_daily     # OHLCV + Volume
- stock_prices_intraday  # Optional: 1h/5m granularity
- dividends_history      # Ex-date, payment date, amount
- splits_history         # Split ratios and dates
- company_metadata       # Sector, industry, description, employees, etc.
- analyst_estimates      # Revenue/EPS forecasts
- insider_transactions   # Buys/sells by executives (if available)

# DuckDB Tables
- prices.daily           # ticker, date, open, high, low, close, adj_close, volume
- prices.intraday        # ticker, timestamp, ohlcv (optional)
- dividends.history      # ticker, ex_date, payment_date, amount
- splits.history         # ticker, date, ratio
- company.metadata       # ticker, sector, industry, market_cap, employees, description
- estimates.analyst      # ticker, date, metric, estimate, actual
```

**Implementation**:
- Extend `ingest.py` with new methods: `fetch_price_history()`, `fetch_dividends()`, etc.
- Add `PRICE_LOOKBACK_DAYS` env var (default 365 for 1 year)
- Incremental price updates: only fetch dates newer than last stored date
- Store metadata in separate collection with `last_updated` timestamp

### 1.2 Derived Financial Ratios Table

**New DuckDB Table**: `ratios.financial`
```sql
CREATE TABLE ratios.financial AS
SELECT
  ticker,
  date,
  netIncome / totalRevenue AS net_margin,
  netIncome / shareholderEquity AS roe,
  netIncome / totalAssets AS roa,
  totalLiabilities / totalAssets AS debt_ratio,
  operatingCashFlow / netIncome AS cash_conversion,
  freeCashFlow / totalRevenue AS fcf_margin,
  totalRevenue / totalAssets AS asset_turnover
FROM financials.annual
WHERE totalRevenue > 0 AND shareholderEquity > 0;
```

**Benefits**:
- Answer ratio queries without LLM computing divisions
- Pre-validated calculations (no division by zero)
- Faster query execution

### 1.3 Time-Series Views for Growth Calculations

**New DuckDB Views**:
```sql
-- Year-over-year growth rates
CREATE VIEW growth.annual AS
SELECT
  current.ticker,
  current.date,
  (current.totalRevenue - prior.totalRevenue) / NULLIF(prior.totalRevenue, 0) AS revenue_growth_yoy,
  (current.netIncome - prior.netIncome) / NULLIF(prior.netIncome, 0) AS income_growth_yoy
FROM financials.annual AS current
LEFT JOIN financials.annual AS prior
  ON current.ticker = prior.ticker
  AND current.date = prior.date + INTERVAL 1 YEAR;
```

---

## Phase 2: Smart Caching & Incremental Updates

### 2.1 Data Freshness Tracking

**New MongoDB Collection**: `ingestion_metadata`
```json
{
  "ticker": "AAPL",
  "data_type": "financials_annual",
  "last_fetched": "2025-11-09T10:30:00Z",
  "last_successful_date": "2024-09-30",
  "record_count": 5,
  "status": "success"
}
```

**Features**:
- Track per-ticker, per-data-type freshness
- `--refresh` flag: only re-ingest if older than N days (configurable)
- `--force` flag: ignore freshness, re-fetch everything
- Auto-detect stale data: warn if data >90 days old for active tickers

### 2.2 Incremental Price Updates

**Current Problem**: No price data at all

**Solution**:
```python
def fetch_incremental_prices(ticker: str, last_date: Optional[date]) -> pd.DataFrame:
    """Fetch only prices newer than last_date."""
    ticker_obj = yf.Ticker(ticker)
    if last_date:
        start = last_date + timedelta(days=1)
        history = ticker_obj.history(start=start, end=date.today())
    else:
        history = ticker_obj.history(period="1y")
    return history
```

**Benefits**:
- Ingest runs in seconds instead of minutes
- Can schedule hourly/daily updates via cron
- Reduces API load on yfinance

### 2.3 Auto-Refresh on Query

**Enhancement to `query.py`**:
```python
# Before executing query, check freshness
if is_data_stale(tickers_in_query, max_age_days=7):
    print("⚠️  Data is older than 7 days. Run 'python ingest.py --refresh' to update.")
    user_input = input("Continue with stale data? [y/N]: ")
    if user_input.lower() != 'y':
        sys.exit(0)
```

---

## Phase 3: Conversational Query Interface

### 3.1 Multi-Turn Conversation with Context

**New Script**: `chat.py` (replaces `query.py` for interactive use)

**Features**:
- Maintains conversation history with Ollama
- Remembers previous queries and results
- Supports follow-up questions:
  - User: "Show AAPL revenue for last 5 years"
  - System: [Shows table]
  - User: "Now compare to MSFT"
  - System: [Knows to use same time range, adds MSFT]

**Implementation**:
```python
conversation_history = [
    {"role": "system", "content": system_prompt}
]

while True:
    user_input = input("\nQuery> ")
    if user_input.lower() in {'exit', 'quit'}:
        break

    conversation_history.append({"role": "user", "content": user_input})

    # Call Ollama with full history
    response = call_ollama_chat(conversation_history)

    # Execute SQL, show results
    results = execute_query(response)

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": response})
    conversation_history.append({
        "role": "system",
        "content": f"Query returned {len(results)} rows"
    })
```

### 3.2 Intelligent Error Recovery

**Current Problem**: LLM generates invalid SQL → user sees error → manual retry

**Solution**: Automated retry with feedback loop
```python
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    try:
        sql = extract_sql(llm_response)
        validated_sql = validate_sql(sql, schema)
        results = conn.execute(validated_sql).fetchall()
        break
    except ValueError as e:
        if attempt == MAX_RETRIES - 1:
            raise
        # Feed error back to LLM
        error_msg = f"Previous SQL failed: {e}. Please revise."
        conversation_history.append({"role": "system", "content": error_msg})
        llm_response = call_ollama_chat(conversation_history)
```

**Benefits**:
- Self-correcting queries
- User doesn't see internal SQL errors
- LLM learns from mistakes within session

### 3.3 Query Suggestions & Examples

**Enhancement to System Prompt**:
```python
# Add recent successful queries as examples
recent_queries = get_recent_queries(limit=5)  # From logs
examples_text = "\n".join([f"- {q['nl_query']} → {q['sql']}" for q in recent_queries])

system_prompt += f"\n\nRecent successful queries:\n{examples_text}"
```

**Startup Message**:
```
Welcome to FinanGPT Interactive Query Interface

Available data: 150 tickers, 2019-2024
Try asking:
  • "Show top 10 companies by revenue in 2023"
  • "Compare AAPL and MSFT profit margins over time"
  • "Which tech stocks have highest dividend yield?"
  • "Plot TSLA stock price for last 6 months"
```

---

## Phase 4: Visual Analytics & Charting

### 4.1 Matplotlib Integration

**New Module**: `visualize.py`

**Chart Types**:
1. **Time-series line charts**: Revenue/income trends, stock prices
2. **Bar charts**: Compare metrics across tickers
3. **Scatter plots**: Correlation analysis (e.g., P/E vs growth rate)
4. **Heatmaps**: Sector performance matrix
5. **Candlestick charts**: OHLC stock price data (via `mplfinance`)

**Implementation**:
```python
def detect_visualization_intent(query: str, results: pd.DataFrame) -> Optional[str]:
    """Determine if query wants a chart."""
    chart_keywords = ['plot', 'chart', 'graph', 'visualize', 'show trend']
    if any(kw in query.lower() for kw in chart_keywords):
        return infer_chart_type(results)
    # Also auto-visualize time-series with >5 rows
    if 'date' in results.columns and len(results) >= 5:
        return 'line'
    return None

def create_chart(df: pd.DataFrame, chart_type: str, title: str) -> None:
    if chart_type == 'line':
        plt.figure(figsize=(12, 6))
        for ticker in df['ticker'].unique():
            subset = df[df['ticker'] == ticker]
            plt.plot(subset['date'], subset.iloc[:, 2], label=ticker, marker='o')
        plt.xlabel('Date')
        plt.ylabel(df.columns[2])
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'charts/{sanitize_filename(title)}.png', dpi=150)
        plt.show()
```

**Query Examples**:
- "Plot AAPL revenue over time" → Auto-generates line chart
- "Compare tech stocks by market cap" → Bar chart
- "Show TSLA price for last year" → Candlestick chart

### 4.2 Enhanced Output Formatting

**Current**: Plain text table via `pretty_print()`

**Enhancement**: Rich formatting with units
```python
def format_financial_value(value: float, column_name: str) -> str:
    """Format large numbers with K/M/B suffixes."""
    if 'revenue' in column_name.lower() or 'income' in column_name.lower():
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
    if 'ratio' in column_name.lower() or 'margin' in column_name.lower():
        return f"{value*100:.2f}%"
    return f"{value:.2f}"
```

**Output Example**:
```
ticker | date       | totalRevenue | netIncome  | net_margin
-------|------------|--------------|------------|------------
AAPL   | 2024-09-30 | $394.33B     | $93.74B    | 23.76%
MSFT   | 2024-06-30 | $245.12B     | $88.14B    | 35.96%
```

### 4.3 Export Options

**New Flags**:
```bash
python chat.py --export-csv results.csv
python chat.py --export-json results.json
python chat.py --export-excel report.xlsx  # Multi-sheet workbook
```

---

## Phase 5: Advanced Query Capabilities

### 5.1 Peer Group Analysis

**New Table**: `company.peers`
```sql
CREATE TABLE company.peers (
  ticker VARCHAR,
  sector VARCHAR,
  industry VARCHAR,
  peer_group VARCHAR  -- e.g., "FAANG", "Semiconductors"
);
```

**Seeded Data**:
```python
PEER_GROUPS = {
    "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "TSM"],
    "Megacap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
}
```

**Query Examples**:
- "Compare FAANG revenue growth" → Auto-expands to 5 tickers
- "Rank semiconductor companies by profit margin" → Uses peer group

### 5.2 Natural Language Date Parsing

**Enhancement**: Parse relative dates in queries
- "last quarter" → Most recent quarterly reporting period
- "last 5 years" → DATE >= CURRENT_DATE - INTERVAL 5 YEAR
- "2023" → DATE BETWEEN '2023-01-01' AND '2023-12-31'
- "YTD" → Year-to-date

**Implementation**: Add date parsing rules to system prompt
```python
date_context = f"""
Today is {date.today().isoformat()}.
When user says:
- "last N years" → filter to dates >= {(date.today() - timedelta(days=365*5)).isoformat()}
- "recent" or "latest" → ORDER BY date DESC LIMIT 1
- "2023" → extract YEAR(date) = 2023
"""
system_prompt = base_prompt + "\n" + date_context
```

### 5.3 Statistical Aggregations

**Allow Window Functions** (update SQL guardrails):
```python
# Currently blocks all non-SELECT keywords
# Whitelist window functions: RANK(), ROW_NUMBER(), LAG(), LEAD()
ALLOWED_FUNCTIONS = {
    'rank', 'row_number', 'lag', 'lead', 'dense_rank',
    'avg', 'sum', 'count', 'min', 'max', 'stddev'
}
```

**Query Examples Now Possible**:
- "Show 3-year moving average of AAPL revenue"
- "Rank companies by revenue growth rate"
- "Calculate median P/E ratio for tech sector"

### 5.4 Portfolio Analysis

**New Table**: `user.portfolios`
```sql
CREATE TABLE user.portfolios (
  portfolio_name VARCHAR,
  ticker VARCHAR,
  shares DOUBLE,
  purchase_date DATE,
  purchase_price DOUBLE
);
```

**Query Examples**:
- "What's my portfolio value today?" (joins prices.daily)
- "Show unrealized gains for my tech holdings"
- "Calculate portfolio allocation by sector"

---

## Phase 6: Error Resilience & UX Polish

### 6.1 Graceful Degradation

**Scenario**: Ollama is down

**Current Behavior**: ConnectionError, script exits

**Enhanced Behavior**:
```python
try:
    response = call_ollama(...)
except requests.ConnectionError:
    print("⚠️  Ollama is not reachable. Fallback options:")
    print("1. Enter SQL directly (expert mode)")
    print("2. Use saved query templates")
    print("3. Exit and fix connection")

    choice = input("Select [1/2/3]: ")
    if choice == '1':
        sql = input("SQL> ")
        results = execute_sql_directly(sql)
```

### 6.2 Query Templates Library

**New File**: `templates/queries.yaml`
```yaml
top_revenue:
  description: "Top N companies by revenue in specific year"
  sql: "SELECT ticker, date, totalRevenue FROM financials.annual WHERE YEAR(date) = {year} ORDER BY totalRevenue DESC LIMIT {limit}"
  params: [year, limit]

ticker_comparison:
  description: "Compare specific metric across multiple tickers"
  sql: "SELECT ticker, date, {metric} FROM financials.annual WHERE ticker IN ({tickers}) ORDER BY date DESC"
  params: [metric, tickers]
```

**Usage**:
```bash
python query.py --template top_revenue --year 2023 --limit 10
```

### 6.3 Input Validation & Sanitization

**Enhancement**: Validate ticker symbols before querying
```python
def validate_ticker(ticker: str) -> bool:
    """Check if ticker exists in database."""
    result = conn.execute(
        "SELECT COUNT(*) FROM company.metadata WHERE ticker = ?",
        [ticker]
    ).fetchone()
    return result[0] > 0

def suggest_tickers(partial: str) -> List[str]:
    """Autocomplete ticker symbols."""
    results = conn.execute(
        "SELECT ticker FROM company.metadata WHERE ticker LIKE ? LIMIT 10",
        [f"{partial}%"]
    ).fetchall()
    return [r[0] for r in results]
```

**User Experience**:
```
Query> Show APLE revenue
⚠️  Ticker 'APLE' not found. Did you mean: AAPL, APLE? [Select or type 'cancel']
```

### 6.4 Comprehensive Logging & Debugging

**Enhancement**: `--debug` flag shows full SQL and LLM prompts
```bash
python chat.py --debug

# Output:
[DEBUG] System Prompt (1245 chars):
You are FinanGPT, a financial SQL assistant...

[DEBUG] User Query: Show AAPL revenue for last 5 years

[DEBUG] LLM Response:
SELECT ticker, date, totalRevenue
FROM financials.annual
WHERE ticker = 'AAPL'
ORDER BY date DESC
LIMIT 5

[DEBUG] Validated SQL:
SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 5

[DEBUG] Query Time: 0.043s
[DEBUG] Rows Returned: 5
```

---

## Phase 7: Unified Workflow & Automation

### 7.1 Single Entry Point CLI

**New Script**: `finangpt.py` (replaces separate scripts)

```bash
# Ingest data
finangpt ingest --tickers AAPL,MSFT --include prices,dividends,fundamentals

# Transform (auto-runs if data is stale)
finangpt transform --auto-refresh

# Query (one-shot)
finangpt query "Show AAPL revenue trends"

# Interactive chat
finangpt chat

# Status check
finangpt status  # Shows data freshness, ticker count, last update time

# Full refresh
finangpt refresh --tickers-file tickers.csv --all-data-types
```

### 7.2 Configuration File

**New File**: `config.yaml`
```yaml
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
  max_limit: 500  # Increased from 100
  enable_visualizations: true
  chart_output_dir: charts/
  export_formats: [csv, json, excel]

features:
  conversational_mode: true
  auto_error_recovery: true
  query_suggestions: true
  portfolio_tracking: false  # Future feature
```

### 7.3 Scheduled Updates via Cron

**Script**: `scripts/daily_refresh.sh`
```bash
#!/bin/bash
source .venv/bin/activate
python finangpt.py refresh --incremental --quiet
python finangpt.py transform --auto
# Email notification on failure
```

**Cron Entry**:
```
0 18 * * 1-5  /path/to/scripts/daily_refresh.sh  # Weekdays at 6pm
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Extend ingestion to include prices, dividends, metadata
- [ ] Add `ratios.financial` and `growth.annual` tables
- [ ] Implement freshness tracking in MongoDB
- [ ] Test incremental price updates

### Phase 2: Query Intelligence (Week 3)
- [ ] Build conversational chat interface (`chat.py`)
- [ ] Add auto-retry with error feedback
- [ ] Implement query templates library
- [ ] Enhanced error messages with suggestions

### Phase 3: Visualizations (Week 4)
- [ ] Integrate matplotlib for time-series charts
- [ ] Add bar charts and scatter plots
- [ ] Implement auto-detection of visualization intent
- [ ] Format financial values with K/M/B units

### Phase 4: Advanced Features (Week 5)
- [ ] Peer group analysis tables and queries
- [ ] Natural language date parsing
- [ ] Whitelist window functions in SQL validator
- [ ] Ticker autocomplete and validation

### Phase 5: Polish & Automation (Week 6)
- [ ] Unified `finangpt.py` CLI
- [ ] Configuration file support (`config.yaml`)
- [ ] Automated daily refresh script
- [ ] Comprehensive test coverage for new features

### Phase 6: Documentation (Week 7)
- [ ] Update CLAUDE.md with new architecture
- [ ] Write user guide for interactive mode
- [ ] Create example query cookbook
- [ ] Record demo video

---

## Testing Strategy

### New Test Files

1. **`test_price_ingestion.py`**:
   - Verify incremental price fetch
   - Test OHLCV data validation
   - Check duplicate handling

2. **`test_conversational_query.py`**:
   - Multi-turn conversation context
   - Auto-retry on SQL errors
   - Conversation history limits

3. **`test_visualizations.py`**:
   - Chart generation for various data shapes
   - File output and cleanup
   - Format detection logic

4. **`test_freshness_tracking.py`**:
   - Metadata upserts
   - Staleness detection
   - Auto-refresh triggers

5. **`test_peer_analysis.py`**:
   - Peer group expansion
   - Cross-ticker aggregations
   - Ranking queries

### Integration Tests

**End-to-End Workflow**:
```python
def test_full_query_pipeline():
    # 1. Ingest fresh data
    ingest_tickers(['AAPL', 'MSFT'], include=['prices', 'fundamentals'])

    # 2. Transform to DuckDB
    transform_all()

    # 3. Query via LLM
    response = query_natural_language("Compare AAPL and MSFT revenue")

    # 4. Verify results
    assert len(response.results) > 0
    assert 'AAPL' in response.tickers
    assert response.chart_path is not None
```

---

## Risk Mitigation

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinates SQL | High | Multi-retry with error feedback; SQL validation |
| yfinance API rate limits | Medium | Exponential backoff; cache metadata |
| DuckDB schema drift | Medium | Schema versioning; migration scripts |
| Large result sets crash UI | Low | Enforce LIMIT 500; paginate large outputs |
| User confusion in chat mode | Medium | Rich help text; query suggestions; examples |

---

## Success Metrics

1. **Query Success Rate**: >90% of natural language queries execute without manual SQL editing
2. **Data Freshness**: <7 days average staleness for active tickers
3. **User Efficiency**: 3x faster to answer multi-ticker questions vs manual SQL
4. **Visualization Adoption**: >50% of time-series queries auto-generate charts
5. **Error Recovery**: 80% of failed queries succeed after auto-retry

---

## Dependencies to Add

```txt
# requirements.txt additions
matplotlib>=3.8.0
mplfinance>=0.12.10b0
pyyaml>=6.0
tabulate>=0.9.0
prompt-toolkit>=3.0.43  # For advanced CLI input
rich>=13.7.0  # For terminal formatting
```

---

## Breaking Changes & Migration

**Backwards Compatibility**:
- Existing `query.py` remains functional
- New `chat.py` is opt-in
- `config.yaml` defaults to current .env behavior
- Old DuckDB schema auto-migrates (adds new tables, keeps existing)

**Migration Path**:
1. Install new dependencies: `pip install -r requirements.txt`
2. Run schema migration: `python migrate_schema.py` (creates new tables)
3. Backfill price data: `python finangpt.py ingest --tickers-file tickers.csv --include prices --force`
4. Test new chat mode: `python chat.py`
5. Optional: Migrate to `config.yaml` from `.env`

---

## Open Questions for Sign-Off

1. **Max LIMIT increase**: Raise from 100 to 500? Or configurable per-user?
2. **Intraday price data**: Include 1h/5m granularity or only daily? (Storage impact)
3. **LLM model requirement**: Should we support OpenAI API as fallback if Ollama is down?
4. **Portfolio feature**: Include in Phase 1 or defer to Phase 2?
5. **Visualization defaults**: Auto-show charts in terminal (iTerm2/kitty) or always save to file?
6. **Data retention**: Keep raw MongoDB data indefinitely or purge >3 years?
7. **Multi-user support**: Single-user tool or add user accounts/portfolios?

---

**End of Implementation Plan**
