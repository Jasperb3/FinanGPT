# FinanGPT: AI-Powered Financial Analysis Platform

An intelligent financial data pipeline that combines comprehensive data ingestion, smart caching, and conversational natural language querying. Ask questions in plain English and get instant insights from US stock financials, prices, dividends, and more.

## 🌟 Key Features

### 💾 Comprehensive Data Coverage
- **Financial Statements**: Annual & quarterly income statements, balance sheets, cash flow
- **Market Data**: Daily stock prices (OHLCV), adjusted closes, trading volume
- **Corporate Actions**: Dividend payments, stock splits with historical tracking
- **Company Information**: Sector, industry, employees, market cap, descriptions
- **Derived Analytics**: 9 key financial ratios (ROE, ROA, margins, etc.), YoY growth metrics

### ⚡ Smart Caching & Performance
- **Incremental Updates**: 10-100x faster refresh by fetching only new data
- **Freshness Tracking**: MongoDB-backed metadata monitors data age per ticker
- **Three Ingestion Modes**: Normal (full), Refresh (smart), Force (rebuild)
- **Automated Workflows**: Cron-ready for daily updates with staleness detection

### 💬 Conversational Query Interface
- **Interactive Chat Mode**: Multi-turn conversations with context memory
- **Intelligent Error Recovery**: Auto-retries with LLM feedback (3 attempts)
- **One-Shot Queries**: Quick answers with `query.py`
- **Natural Language**: No SQL knowledge required - ask in plain English

### 📊 Visual Analytics & Charting
- **Automatic Chart Generation**: Intelligent detection and creation of charts
- **4 Chart Types**: Line (time-series), Bar (comparisons), Scatter (correlations), Candlestick (OHLC)
- **Financial Formatting**: Smart formatting ($1.50B, 25.00%, $150.25)
- **Multiple Exports**: CSV, JSON, Excel with preserved data types

### 🚀 Advanced Query Capabilities
- **Peer Group Analysis**: 16+ predefined groups (FAANG, Semiconductors, etc.)
- **Natural Language Dates**: "last year", "YTD", "2023" automatically parsed
- **Window Functions**: RANK(), LAG(), LEAD(), statistical aggregations
- **Portfolio Tracking**: Track holdings, calculate returns, analyze performance

### 🛡️ Error Resilience & UX Polish
- **Graceful Degradation**: Fallback options when Ollama is unavailable (direct SQL, templates)
- **Query Templates**: 10+ pre-built templates for common queries with parameter substitution
- **Ticker Validation**: Auto-complete and spell-check for stock ticker symbols
- **Debug Mode**: Comprehensive logging shows LLM prompts, SQL generation, and query timing

### 🔧 Unified Workflow & Automation
- **Single CLI Entry Point**: Unified `finangpt.py` command for all operations
- **Configuration Management**: YAML-based config with environment variable fallback
- **System Status Monitoring**: Real-time health checks and data freshness statistics
- **Automated Updates**: Cron-ready scripts for scheduled daily/weekly refreshes

### 🔒 Enterprise-Grade Safety
- **Data Validation**: US-only, non-ETF, USD-denominated instruments
- **SQL Guardrails**: Table allow-lists, column validation, read-only queries
- **LIMIT Enforcement**: Default 25 rows, max 100 to protect resources
- **Exponential Backoff**: Robust API retry logic for yfinance

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **MongoDB** (local or remote instance)
- **Ollama** with a model (e.g., `phi4:latest` or `gpt-oss:latest`)

### Installation

1. **Clone and setup virtual environment**:
   ```bash
   git clone <repository>
   cd FinanGPT
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (create `.env` file):
   ```bash
   MONGO_URI=mongodb://localhost:27017/financial_data
   OLLAMA_URL=http://localhost:11434
   MODEL_NAME=phi4:latest
   PRICE_LOOKBACK_DAYS=365
   ```

4. **Start services**:
   ```bash
   # Terminal 1: MongoDB
   mongod --dbpath /path/to/data

   # Terminal 2: Ollama
   ollama serve
   ollama pull phi4:latest  # or gpt-oss:latest
   ```

### Basic Usage

#### Option 1: Unified CLI (Phase 7 - Recommended)

```bash
# Single-command workflow
python finangpt.py refresh --tickers AAPL,MSFT,GOOGL

# Or query directly
python finangpt.py query "Show AAPL revenue for the last 5 years"

# Interactive chat
python finangpt.py chat

# Check system status
python finangpt.py status
```

#### Option 2: Individual Scripts (Compatible)

**1. Ingest financial data**:
```bash
# Ingest specific tickers
python ingest.py --tickers AAPL,MSFT,GOOGL

# Or from a CSV file (with 'ticker' column)
python ingest.py --tickers-file tickers.csv
```

**2. Transform to analytics database**:
```bash
python transform.py
```

**3. Query in natural language**:

**One-shot query**:
```bash
python query.py "Show AAPL revenue for the last 5 years"
```

**Interactive chat** (recommended):
```bash
python chat.py
```
```
💬 Query> Show AAPL revenue trends for last 5 years
✅ [Shows results]

💬 Query> Now compare to MSFT
✅ [Shows comparison with same context]

💬 Query> Which has higher profit margins?
✅ [Analyzes both companies]
```

## 📊 Advanced Features

### Smart Caching (Phase 2)

**Refresh Mode** - Only update stale data:
```bash
# Update tickers older than 7 days (default)
python ingest.py --refresh --tickers AAPL,MSFT

# Custom staleness threshold (3 days)
python ingest.py --refresh --refresh-days 3 --tickers-file tickers.csv
```

**Force Mode** - Full re-fetch:
```bash
python ingest.py --force --tickers AAPL,MSFT
```

**Benefits**:
- ⚡ **20x faster** for daily updates (60s → 3s)
- 📉 **Reduced API load** on yfinance
- 🤖 **Automation-ready** for scheduled jobs

### Conversational Interface (Phase 3)

**Chat Commands**:
- `/help` - Show usage tips and examples
- `/clear` - Reset conversation history
- `/exit` or `/quit` - Exit chat mode

**Skip freshness check**:
```bash
python chat.py --skip-freshness-check
```

**Example conversation**:
```
You: Show me tech stocks with ROE > 20%
AI:  [Shows filtered results]

You: Add their market caps
AI:  [Adds market cap column using context]

You: Sort by highest market cap
AI:  [Re-sorts maintaining all previous filters]
```

### Visual Analytics (Phase 4)

**Automatic Chart Generation**:
```bash
# Line chart for time-series data
python query.py "plot AAPL stock price over the last 6 months"

# Bar chart for comparisons
python query.py "compare revenue for AAPL, MSFT, GOOGL"

# Scatter plot for correlations
python query.py "show relationship between ROE and debt ratio"

# Candlestick chart for OHLC data
python query.py "show TSLA candlestick chart for October 2024"
```

**Financial Formatting** (automatic):
```
# Revenue/Income → $1.50B, $250.00M, $3.45K
# Margins/Ratios → 25.00%, 15.50%, -3.25%
# Prices → $150.25, $3,247.89
# Volume → 1,500,000 (comma-separated)
```

**Control Options**:
```bash
# Disable chart generation
python query.py --no-chart "show data"

# Disable financial formatting (raw numbers)
python query.py --no-formatting "show data"
```

**Chart Output**:
- All charts saved to `charts/` directory
- Timestamped filenames: `Query_Result_20251109_143022.png`
- High-resolution PNG (300 DPI)
- Professional styling with grid lines and legends

### Advanced Query Capabilities (Phase 5)

**Peer Group Analysis**:
```bash
# Query with predefined peer groups
python query.py "Compare FAANG companies by revenue"
python query.py "Rank semiconductor companies by profit margin"
python query.py "Show average ROE for FAANG vs Semiconductors"
```

**Available Peer Groups**:
- FAANG, Magnificent Seven, Semiconductors, Cloud Computing
- Social Media, Streaming, E-commerce, Payment Processors
- Electric Vehicles, Airlines, Banks, Oil & Gas
- Defense, Retail, Pharma, Telecom

**Natural Language Date Parsing**:
```bash
# Relative dates automatically understood
python query.py "Show AAPL revenue for the last 5 years"
python query.py "Compare Q4 2023 vs Q4 2024 performance"
python query.py "What's the YTD revenue for tech companies?"
```

**Window Functions & Rankings**:
```bash
# Advanced analytical queries
python query.py "Show top 10 companies by revenue with their rank"
python query.py "Calculate 3-period moving average for AAPL revenue"
python query.py "Find companies in top quartile by ROE"
```

**Portfolio Tracking**:
```bash
# After adding holdings to user.portfolios table
python query.py "What's my portfolio value today?"
python query.py "Show unrealized gains for my tech holdings"
python query.py "Calculate portfolio allocation by sector"
```

**Adding Portfolio Holdings** (SQL):
```sql
INSERT INTO user.portfolios
    (portfolio_name, ticker, shares, purchase_date, purchase_price, notes)
VALUES
    ('Tech Growth', 'AAPL', 100, '2023-01-15', 150.50, 'Long-term hold'),
    ('Tech Growth', 'MSFT', 50, '2023-02-20', 280.00, 'Cloud play');
```

### Error Resilience & UX Polish (Phase 6)

**Query Templates**:
```bash
# List all available templates
python query.py --list-templates

# Use a template with parameters
python query.py --template top_revenue --template-params "year=2023,limit=10"
python query.py --template ticker_comparison --template-params "metric=totalRevenue,tickers='AAPL','MSFT',limit=5"
python query.py --template peer_group_comparison --template-params "metric=netIncome,peer_group=FAANG,year=2024"
```

**Available Templates**:
- `top_revenue`: Top N companies by revenue in a specific year
- `ticker_comparison`: Compare a specific metric across multiple tickers
- `revenue_trends`: Show revenue trends for a ticker over N years
- `profit_margins`: Show profit margins for specific tickers
- `peer_group_comparison`: Compare metrics across a peer group
- `dividend_history`: Dividend history for a specific ticker
- `stock_price_range`: Stock prices within a date range
- `growth_leaders`: Companies with highest revenue growth
- `sector_analysis`: Average metrics by sector for a specific year

**Graceful Degradation** (when Ollama is down):
```bash
# If Ollama connection fails, you get options:
# 1. Enter SQL directly (expert mode)
# 2. Use saved query templates
# 3. Exit and fix connection
```

**Debug Mode**:
```bash
# Enable comprehensive debug logging
python query.py --debug "Show AAPL revenue"
# Shows: system prompt, LLM response, extracted SQL, validated SQL, query timing

# Debug mode in chat
python chat.py --debug
```

**Ticker Validation** (programmatic):
```python
from resilience import validate_ticker, suggest_tickers

# Check if ticker exists
if validate_ticker("AAPL", conn):
    print("Valid ticker")

# Get autocomplete suggestions
suggestions = suggest_tickers("A", conn, limit=5)
# Returns: ['AAPL', 'AMD', 'AMZN', ...]
```

## 🗃️ Data Schema

### DuckDB Tables

| Schema | Table | Description |
|--------|-------|-------------|
| `financials` | `annual` | Annual financial statements |
| `financials` | `quarterly` | Quarterly financial statements |
| `prices` | `daily` | Daily OHLCV price data |
| `dividends` | `history` | Dividend payment records |
| `splits` | `history` | Stock split events |
| `company` | `metadata` | Company information |
| `ratios` | `financial` | Derived financial ratios |
| `growth` | `annual` | YoY growth calculations (view) |
| `company` | `peers` | Peer group mappings (Phase 5) |
| `user` | `portfolios` | Portfolio holdings (Phase 5) |

### Key Financial Ratios

The `ratios.financial` table provides 9 pre-calculated metrics:
- **net_margin**: netIncome / totalRevenue
- **roe**: Return on Equity
- **roa**: Return on Assets
- **debt_ratio**: totalLiabilities / totalAssets
- **cash_conversion**: operatingCashFlow / netIncome
- **fcf_margin**: freeCashFlow / totalRevenue
- **asset_turnover**: totalRevenue / totalAssets
- **gross_margin**: grossProfit / totalRevenue
- **ebitda_margin**: ebitda / totalRevenue

## 🧪 Testing

Run the comprehensive test suite:
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_freshness_tracking.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

**Test Coverage**:
- ✅ Ingestion filters (ETF detection, currency validation)
- ✅ Transform schema guarantees (numeric columns, date parsing)
- ✅ SQL guardrails (injection prevention, table allow-lists)
- ✅ Freshness tracking (staleness detection, skip logic)
- ✅ Conversational interface (history management, error recovery)
- ✅ Visualization (chart detection, formatting, export functions)
- ✅ Advanced queries (peer groups, date parsing, window functions)
- ✅ Error resilience (query templates, ticker validation, graceful degradation)

## 📚 Example Queries

### Financial Analysis
```
"Show AAPL's profit margins over the last 5 years"
"Which companies have the highest ROE in the tech sector?"
"Compare MSFT and GOOGL revenue growth year-over-year"
"List all companies with debt ratio < 0.3"
```

### Price & Market Data
```
"What was TSLA's closing price on 2024-01-15?"
"Show AAPL's stock performance for the last quarter"
"Which stocks have paid dividends in 2024?"
"Compare stock prices for FAANG stocks"
```

### Company Information
```
"List all companies in the semiconductor industry"
"What is Microsoft's market cap?"
"Show me all technology sector stocks"
"Find companies with >50000 employees"
```

### Complex Analysis
```
"Show companies with ROE > 20% and positive revenue growth"
"Compare dividend yields for all stocks in my portfolio"
"Find stocks with price drops >10% in the last month"
```

### Visual Analysis (Phase 4)
```
"Plot AAPL revenue trend over the last 5 years"
"Compare profit margins for AAPL, MSFT, GOOGL as a bar chart"
"Show candlestick chart for TSLA in October 2024"
"Scatter plot of ROE vs revenue growth for all companies"
"Plot closing prices for FAANG stocks over the last quarter"
```

### Advanced Queries (Phase 5)
```
"Compare FAANG companies by revenue"
"Rank semiconductor companies by profit margin"
"Show average ROE for FAANG vs Semiconductors"
"Which Magnificent Seven stock has the highest revenue growth?"
"Show top 10 companies by market cap with their rank"
"Calculate 3-period moving average for AAPL revenue"
"What's my Tech Growth portfolio worth today?"
"Show unrealized gains for my holdings"
```

## 🔧 Troubleshooting

### Common Issues

**MongoDB connection fails**:
- Ensure `MONGO_URI` includes database name: `mongodb://localhost:27017/financial_data`
- Check MongoDB is running: `mongod --version`
- Verify connection: `mongo <uri>`

**Ollama not reachable**:
- Check service: `ollama list`
- Verify URL in `.env`: `OLLAMA_URL=http://localhost:11434`
- Pull model if missing: `ollama pull phi4:latest`

**Schema mismatch**:
- Run `transform.py` to rebuild DuckDB
- Delete `financial_data.duckdb` if columns changed
- Verify MongoDB has data: `db.raw_annual.count()`

**ETF/Non-USD rejection**:
- By design - only US equities with USD statements
- Check ticker is correct: AAPL ✅, VOO ❌ (ETF)

**Stale data warnings**:
- Run refresh: `python ingest.py --refresh --tickers <TICKER>`
- Or skip check: `python query.py --skip-freshness-check "query"`

## 🏗️ Architecture

```
┌─────────────────┐
│   yfinance API  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│   ingest.py     │─────▶│   MongoDB        │
│  (with retry)   │      │  • raw_annual    │
└─────────────────┘      │  • raw_quarterly │
                         │  • prices_daily  │
                         │  • dividends     │
                         │  • splits        │
                         │  • metadata      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │ transform.py   │
                         └────────┬───────┘
                                  │
                                  ▼
┌─────────────────┐      ┌──────────────────┐
│   Ollama LLM    │      │   DuckDB         │
│  • phi4:latest  │      │  • financials.*  │
│  • gpt-oss      │      │  • prices.*      │
└────────┬────────┘      │  • ratios.*      │
         │               │  • growth.*      │
         │               └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────┐
│  query.py / chat.py                 │
│  • LLM → SQL                        │
│  • Validation & guardrails          │
│  • Freshness checking               │
│  • Error recovery                   │
└─────────────────────────────────────┘
```

## 📈 Performance Metrics

| Operation | Before | After Phase 2 | Improvement |
|-----------|--------|---------------|-------------|
| Daily refresh (10 tickers) | ~60s | ~3s | **20x faster** |
| Weekly refresh (50 tickers) | ~300s | ~15s | **20x faster** |
| Query with context | N/A | Instant | **New feature** |

## 🔄 Automated Workflows

### Daily Refresh (Cron)

```bash
# Add to crontab: crontab -e
# Run at 6 PM weekdays after market close
0 18 * * 1-5 cd /path/to/FinanGPT && .venv/bin/python ingest.py --refresh --tickers-file tickers.csv
5 18 * * 1-5 cd /path/to/FinanGPT && .venv/bin/python transform.py
```

### Monitoring Freshness (MongoDB)

```javascript
// MongoDB shell
use financial_data

// Check freshness for specific ticker
db.ingestion_metadata.find({"ticker": "AAPL"}).sort({"last_fetched": -1})

// Find all stale data (>7 days)
var threshold = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
db.ingestion_metadata.find({"last_fetched": {$lt: threshold.toISOString()}})
```

## 🛣️ Roadmap

### Completed ✅
- **Phase 1**: Rich data sources (prices, dividends, splits, metadata, ratios)
- **Phase 2**: Smart caching & incremental updates
- **Phase 3**: Conversational query interface with error recovery
- **Phase 4**: Visual analytics & charting with financial formatting
- **Phase 5**: Advanced query capabilities (peer groups, window functions, portfolio tracking)
- **Phase 6**: Error resilience & UX polish (query templates, graceful degradation, debug mode)
- **Phase 7**: Unified workflow & automation (unified CLI, config management, status monitoring, scheduled updates)

### Future Enhancements 🚧
- **Phase 8**: Web dashboard (FastAPI + React frontend)
- **Phase 9**: Real-time data streaming and alerts
- **Phase 10**: ML-powered insights and predictions

## 📝 Development

### Adding New Fields

1. **Update field mappings** (`ingest.py`):
   ```python
   FIELD_MAPPINGS = {
       "newField": ["New Field", "NewField", "new_field"],
   }
   ```

2. **Re-run ingestion and transform**:
   ```bash
   python ingest.py --force --tickers AAPL
   python transform.py
   ```

3. **New field appears in DuckDB automatically**

### Extending Table Allow-List

Edit `query.py` and `chat.py`:
```python
ALLOWED_TABLES = (
    "financials.annual",
    "your.new_table",
    # ...
)
```

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Support

- **Documentation**: See `CLAUDE.md` for detailed architecture
- **Issues**: [GitHub Issues](https://github.com/yourusername/FinanGPT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FinanGPT/discussions)

---

**Built with**: Python 3.10+ • MongoDB • DuckDB • Ollama • yfinance • matplotlib

**Powered by**: LLM-driven natural language query translation with enterprise-grade safety guardrails
