# Phase 2 Quick Start: Global Markets & Multi-Currency

**Getting Started with International Stocks in 5 Minutes**

---

## What's New in Phase 2?

🌍 **Global Market Support**: Analyze stocks from ANY country (not just US)
💱 **Multi-Currency**: Automatic conversion between 12+ major currencies
📊 **Normalized Metrics**: Compare stocks across borders (all in USD or your chosen currency)
🔧 **Flexible Filtering**: Choose which markets to include (global, US-only, EU-only, or custom)

---

## Quick Examples

### 1. Enable Global Markets (1 line change)

```yaml
# config.yaml
ingestion:
  market_restrictions:
    mode: global  # Changed from "us_only" to "global"
```

That's it! You can now ingest international stocks.

### 2. Ingest International Stocks

```bash
# Create a ticker file with global stocks
cat > global_tickers.csv << EOF
AAPL,Apple Inc,United States
BMW.DE,BMW,Germany
7203.T,Toyota,Japan
HSBA.L,HSBC,United Kingdom
ASML.AS,ASML,Netherlands
SAP.DE,SAP,Germany
NESN.SW,Nestle,Switzerland
EOF

# Ingest (same command as before!)
python finangpt.py ingest --tickers-file global_tickers.csv

# Output:
# ✓ AAPL (United States, USD)
# ✓ BMW.DE (Germany, EUR)
# ✓ 7203.T (Japan, JPY)
# ✓ HSBA.L (United Kingdom, GBP)
# ✓ ASML.AS (Netherlands, EUR)
# ✓ SAP.DE (Germany, EUR)
# ✓ NESN.SW (Switzerland, CHF)
```

### 3. Query Multi-Currency Data

```bash
# Compare international companies by revenue (auto-converted to USD)
python query.py "Compare AAPL, BMW.DE, and 7203.T by revenue in USD"

# Output:
# Ticker    Revenue_Local  Currency  Revenue_USD   Company
# AAPL      394.00B       USD       394.00B       Apple Inc
# 7203.T    37.50T        JPY       268.20B       Toyota
# BMW.DE    155.50B       EUR       171.05B       BMW

# Find undervalued stocks across all markets
python query.py "Show me stocks with P/E ratio < 15 and market cap > 50B USD"

# Chat with global data
python chat.py
> "Which European car makers have the highest profit margins?"
> "Compare tech giants from US, Europe, and Asia"
```

---

## Configuration Options

### Mode 1: Global (Accept Everything)

```yaml
# config.yaml
ingestion:
  market_restrictions:
    mode: global  # No country/currency restrictions
    exclude_etfs: true  # Still exclude ETFs/funds (recommended)
```

**Use Case**: Maximum flexibility, analyze any stock worldwide

### Mode 2: US-Only (Legacy Behavior)

```yaml
ingestion:
  market_restrictions:
    mode: us_only  # Only US stocks with USD financials
```

**Use Case**: Keep existing behavior, no changes needed

### Mode 3: EU-Only (European Markets)

```yaml
ingestion:
  market_restrictions:
    mode: eu_only  # 13 European countries, 6 currencies
```

**Use Case**: Focus on European stocks only

**Included Countries**: United Kingdom, Germany, France, Netherlands, Spain, Italy, Ireland, Belgium, Austria, Sweden, Denmark, Finland, Norway

**Included Currencies**: EUR, GBP, SEK, DKK, NOK, CHF

### Mode 4: Custom (Your Rules)

```yaml
ingestion:
  market_restrictions:
    mode: custom
    custom:
      # Only accept these countries (empty = all allowed)
      allowed_countries:
        - United States
        - United Kingdom
        - Germany
        - Japan

      # Only accept these currencies (empty = all allowed)
      allowed_currencies:
        - USD
        - EUR
        - GBP
        - JPY

      # Only accept these exchanges (empty = all allowed)
      allowed_exchanges:
        - NYSE
        - NASDAQ
        - LSE  # London Stock Exchange
        - XETRA  # Frankfurt

    # Always exclude these (regardless of mode)
    exclude_etfs: true
    exclude_mutualfunds: true
    exclude_crypto: true
```

**Use Case**: Fine-grained control, specific market combinations

---

## Currency Settings

```yaml
currency:
  base_currency: USD  # Normalize all metrics to this currency
  auto_fetch_rates: true  # Automatically fetch FX rates during ingestion
  fx_cache_days: 365  # Days of historical FX rates to cache

  # Supported currencies (12 major currencies)
  supported_currencies:
    - USD  # US Dollar
    - EUR  # Euro
    - GBP  # British Pound
    - JPY  # Japanese Yen
    - CNY  # Chinese Yuan
    - CAD  # Canadian Dollar
    - AUD  # Australian Dollar
    - CHF  # Swiss Franc
    - HKD  # Hong Kong Dollar
    - SGD  # Singapore Dollar
    - KRW  # South Korean Won
    - INR  # Indian Rupee
```

**Key Settings**:

- **`base_currency`**: All metrics normalized to this currency (default: USD)
- **`auto_fetch_rates`**: Fetch FX rates automatically (recommended: true)
- **`fx_cache_days`**: How many days of historical rates to cache

---

## Common Workflows

### Workflow 1: Add International Stocks to Existing Database

```bash
# 1. Switch to global mode
# Edit config.yaml: mode: global

# 2. Ingest new tickers (doesn't affect existing US stocks)
python finangpt.py ingest --tickers BMW.DE,7203.T,HSBA.L

# 3. Transform (creates multi-currency valuation table)
python transform.py

# 4. Query!
python query.py "Show me all stocks with their currencies"
```

### Workflow 2: Start Fresh with Global Database

```bash
# 1. Enable global mode in config.yaml

# 2. Create ticker file with international stocks
cat > all_tickers.csv << EOF
# US Stocks
AAPL,Apple Inc,United States
MSFT,Microsoft,United States
GOOGL,Alphabet,United States

# European Stocks
BMW.DE,BMW,Germany
VOW3.DE,Volkswagen,Germany
HSBA.L,HSBC,United Kingdom
BP.L,BP,United Kingdom

# Asian Stocks
7203.T,Toyota,Japan
6758.T,Sony,Japan
005930.KS,Samsung,South Korea
EOF

# 3. Ingest all
python finangpt.py ingest --tickers-file all_tickers.csv

# 4. Transform
python transform.py

# 5. Query
python query.py "Compare auto makers by P/E ratio"
```

### Workflow 3: Compare Markets Side-by-Side

```bash
# Query with multi-currency valuation
python query.py "
  SELECT
    ticker,
    local_currency,
    market_cap_local,
    market_cap_usd,
    pe_ratio
  FROM valuation.metrics_multicurrency
  WHERE ticker IN ('AAPL', 'BMW.DE', '7203.T')
  ORDER BY market_cap_usd DESC
"

# Output:
# ticker  local_currency  market_cap_local  market_cap_usd  pe_ratio
# AAPL    USD            3000.0B           3000.0B         29.5
# 7203.T  JPY            40000.0B          285.7B          8.2
# BMW.DE  EUR            65.0B             71.5B           6.7
```

---

## Python Usage Examples

### Example 1: Fetch FX Rates Manually

```python
from src.data.currency import CurrencyConverter
import duckdb
from datetime import date

# Connect to database
conn = duckdb.connect("financial_data.duckdb")

# Initialize converter
converter = CurrencyConverter(conn)

# Fetch EUR/USD rates for 2024
count = converter.fetch_rates(
    "EUR",
    "USD",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(f"Fetched {count} EUR/USD rates")

# Convert €1000 to USD as of June 1, 2024
usd_amount = converter.convert(
    1000,
    "EUR",
    "USD",
    date(2024, 6, 1)
)

print(f"€1000 = ${usd_amount:.2f}")  # Output: €1000 = $1075.00
```

### Example 2: Prefetch Common Currency Pairs

```python
from src.data.currency import CurrencyConverter
import duckdb
from datetime import date

conn = duckdb.connect("financial_data.duckdb")
converter = CurrencyConverter(conn)

# Prefetch all major currencies vs USD for last year
results = converter.prefetch_common_pairs(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Print results
for pair, count in results.items():
    print(f"{pair}: {count} rates fetched")

# Output:
# EUR/USD: 365 rates fetched
# USD/EUR: 365 rates fetched
# GBP/USD: 365 rates fetched
# USD/GBP: 365 rates fetched
# JPY/USD: 365 rates fetched
# USD/JPY: 365 rates fetched
# ...
```

### Example 3: Create Multi-Currency Valuation Table

```python
from src.data.valuation_multicurrency import create_multicurrency_valuation_table
import duckdb

conn = duckdb.connect("financial_data.duckdb")

# Create table with USD normalization
rows = create_multicurrency_valuation_table(
    conn,
    base_currency="USD"
)

print(f"Created {rows} valuation entries")

# Query results
result = conn.execute("""
    SELECT
        ticker,
        local_currency,
        price_local,
        price_usd,
        pe_ratio,
        market_cap_category
    FROM valuation.metrics_multicurrency
    WHERE local_currency != 'USD'
    ORDER BY market_cap_usd DESC
    LIMIT 10
""").df()

print(result)
```

### Example 4: Check FX Cache Statistics

```python
from src.data.currency import CurrencyConverter
import duckdb

conn = duckdb.connect("financial_data.duckdb")
converter = CurrencyConverter(conn)

# Get cache stats
stats = converter.get_cache_stats()

print(f"Total cached rates: {stats['total_rates']}")
print(f"Currency pairs: {stats['pairs']}")
print(f"Date range: {stats['date_range_start']} to {stats['date_range_end']}")

# Output:
# Total cached rates: 2555
# Currency pairs: 7
# Date range: 2024-01-01 to 2024-12-31
```

---

## Ticker Symbol Format by Country

### United States
- Format: `AAPL`, `MSFT`, `GOOGL`
- No suffix needed

### Germany
- Format: `BMW.DE`, `SAP.DE`, `VOW3.DE`
- Suffix: `.DE` (Xetra exchange)

### Japan
- Format: `7203.T`, `6758.T`, `8306.T`
- Suffix: `.T` (Tokyo Stock Exchange)

### United Kingdom
- Format: `HSBA.L`, `BP.L`, `GSK.L`
- Suffix: `.L` (London Stock Exchange)

### Netherlands
- Format: `ASML.AS`, `HEIA.AS`
- Suffix: `.AS` (Amsterdam)

### Switzerland
- Format: `NESN.SW`, `NOVN.SW`
- Suffix: `.SW` (Swiss Exchange)

### South Korea
- Format: `005930.KS`, `000660.KS`
- Suffix: `.KS` (Korea Stock Exchange)

### Hong Kong
- Format: `0700.HK`, `9988.HK`
- Suffix: `.HK` (Hong Kong Stock Exchange)

### Canada
- Format: `SHOP.TO`, `RY.TO`
- Suffix: `.TO` (Toronto Stock Exchange)

### Australia
- Format: `BHP.AX`, `CBA.AX`
- Suffix: `.AX` (Australian Securities Exchange)

**Tip**: Use [Yahoo Finance](https://finance.yahoo.com) to find correct ticker symbols for international stocks.

---

## Troubleshooting

### Issue: "Country not in allowed list"

**Problem**: Ticker rejected during ingestion
```
UnsupportedInstrument: BMW.DE: Country 'Germany' not in allowed list
```

**Solution**: Switch to global mode or add country to custom list
```yaml
# Option 1: Global mode
ingestion:
  market_restrictions:
    mode: global

# Option 2: Add to custom list
ingestion:
  market_restrictions:
    mode: custom
    custom:
      allowed_countries: ["United States", "Germany"]
```

### Issue: "Exchange rate not available"

**Problem**: FX conversion fails
```python
ValueError: Exchange rate not available: EUR/USD on 2024-06-01
```

**Solution 1**: Enable auto-fetch (recommended)
```yaml
currency:
  auto_fetch_rates: true
```

**Solution 2**: Fetch rates manually
```python
from src.data.currency import CurrencyConverter
converter = CurrencyConverter(conn)
converter.fetch_rates("EUR", "USD", start_date, end_date)
```

**Solution 3**: Prefetch common pairs
```python
converter.prefetch_common_pairs(
    date(2024, 1, 1),
    date(2024, 12, 31)
)
```

### Issue: Empty Multi-Currency Valuation Table

**Problem**: `valuation.metrics_multicurrency` has no rows

**Causes**:
1. No international stocks ingested
2. Missing FX rates
3. Missing currency field in metadata

**Solution**: Re-run ingestion and transformation
```bash
python finangpt.py ingest --refresh --tickers-file tickers.csv
python transform.py
```

### Issue: Slow FX Rate Fetching

**Problem**: First-time FX fetch takes 2-3 minutes

**Explanation**: This is normal - fetching 365 days of rates for 12 currency pairs from yfinance API

**Solution**: Subsequent queries use cached rates (<1ms)

**Optimization**: Reduce `fx_cache_days` if you don't need full year
```yaml
currency:
  fx_cache_days: 90  # Only fetch 90 days (faster)
```

---

## Query Examples

### Find Undervalued Global Stocks

```python
python query.py "
  SELECT
    ticker,
    local_currency,
    pe_ratio,
    pb_ratio,
    market_cap_usd / 1e9 AS market_cap_billions
  FROM valuation.metrics_multicurrency
  WHERE pe_ratio < 15
    AND pb_ratio < 2
    AND market_cap_usd > 10e9
  ORDER BY pe_ratio
  LIMIT 10
"
```

### Compare Auto Makers Globally

```python
python query.py "
  SELECT
    ticker,
    local_currency,
    revenue_usd / 1e9 AS revenue_billions,
    pe_ratio,
    market_cap_category
  FROM valuation.metrics_multicurrency
  WHERE ticker IN ('F', 'GM', 'BMW.DE', 'VOW3.DE', '7203.T', '7267.T')
  ORDER BY revenue_usd DESC
"
```

### Find High-Dividend International Stocks

```python
python query.py "
  SELECT
    ticker,
    local_currency,
    dividend_yield_pct,
    pe_ratio,
    market_cap_usd / 1e9 AS market_cap_billions
  FROM valuation.metrics_multicurrency
  WHERE dividend_yield_pct > 4
    AND local_currency != 'USD'
  ORDER BY dividend_yield_pct DESC
  LIMIT 10
"
```

### Compare Tech Giants Across Regions

```python
python query.py "
  SELECT
    ticker,
    CASE
      WHEN local_currency = 'USD' THEN 'Americas'
      WHEN local_currency IN ('EUR', 'GBP', 'CHF') THEN 'Europe'
      WHEN local_currency IN ('JPY', 'KRW', 'HKD') THEN 'Asia'
      ELSE 'Other'
    END AS region,
    market_cap_usd / 1e9 AS market_cap_billions,
    pe_ratio,
    revenue_usd / 1e9 AS revenue_billions
  FROM valuation.metrics_multicurrency
  WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL', 'SAP.DE', 'ASML.AS', '6758.T', '005930.KS')
  ORDER BY market_cap_usd DESC
"
```

---

## Performance Tips

### 1. Prefetch FX Rates for Faster Queries

```python
from src.data.currency import CurrencyConverter
from datetime import date

converter = CurrencyConverter(conn)

# Prefetch all major pairs for last year
converter.prefetch_common_pairs(
    date(2024, 1, 1),
    date(2024, 12, 31)
)
```

**Result**: 1000x faster conversions (1.5s → 1.5ms)

### 2. Use Multi-Currency Valuation Table

```python
# Slow (computes FX on-the-fly)
python query.py "
  SELECT
    ticker,
    revenue_local * get_fx_rate(local_currency, 'USD', date) AS revenue_usd
  FROM financials.annual
"

# Fast (pre-computed)
python query.py "
  SELECT ticker, revenue_usd
  FROM valuation.metrics_multicurrency
"
```

**Result**: 5-10x faster (no runtime FX lookups)

### 3. Reduce FX Cache Days for Faster Ingestion

```yaml
currency:
  fx_cache_days: 90  # Instead of 365 (4x faster)
```

**Trade-off**: Historical queries beyond 90 days may require additional fetch

---

## Migration from Phase 1

### Minimal Migration (Keep US-Only)

**No changes needed!** Just set:

```yaml
ingestion:
  market_restrictions:
    mode: us_only  # Keep existing behavior
```

### Gradual Migration (Add International Stocks)

**Step 1**: Switch to global mode
```yaml
ingestion:
  market_restrictions:
    mode: global
```

**Step 2**: Ingest new international tickers
```bash
python finangpt.py ingest --tickers BMW.DE,7203.T,HSBA.L
```

**Step 3**: Re-run transformation
```bash
python transform.py
```

**Result**: Existing US stocks + new international stocks, all queryable

### Full Migration (Global-First)

**Step 1**: Backup database
```bash
cp financial_data.duckdb financial_data.duckdb.backup
```

**Step 2**: Switch to global mode
```yaml
ingestion:
  market_restrictions:
    mode: global

currency:
  auto_fetch_rates: true
```

**Step 3**: Re-ingest all tickers
```bash
python finangpt.py ingest --force --tickers-file all_tickers.csv
```

**Step 4**: Transform
```bash
python transform.py
```

---

## FAQ

**Q: Can I use both US and international stocks together?**
A: Yes! Global mode accepts all stocks. Queries can mix US and international tickers seamlessly.

**Q: Do I need to fetch FX rates manually?**
A: No, if `auto_fetch_rates: true` (default). Rates are fetched automatically during ingestion.

**Q: What happens if an FX rate is missing?**
A: System tries inverse rate, cross-rate via USD, and nearest date (±7 days). If still unavailable, raises error.

**Q: Can I change the base currency from USD to EUR?**
A: Yes! Change `base_currency: EUR` in config.yaml and re-run `transform.py`.

**Q: Are ETFs still excluded?**
A: Yes, by default. ETF detection works globally (not just US ETFs).

**Q: How much disk space do FX rates use?**
A: ~4KB per currency pair per year (365 rates). For 12 currencies × 1 year = ~48KB (negligible).

**Q: Can I query in local currency instead of USD?**
A: Yes! Use `price_local`, `revenue_local`, etc. columns in `valuation.metrics_multicurrency`.

**Q: What if yfinance doesn't have data for a currency pair?**
A: System will log warning and skip FX conversion for that pair. Valuation table will exclude those stocks.

---

## Next Steps

### Test with Real Data

```bash
# 1. Ingest sample international stocks
python finangpt.py ingest --tickers AAPL,BMW.DE,7203.T,HSBA.L

# 2. Check FX rates were fetched
python -c "
import duckdb
conn = duckdb.connect('financial_data.duckdb')
print(conn.execute('SELECT COUNT(*) FROM currency.exchange_rates').fetchone()[0], 'FX rates cached')
"

# 3. Query multi-currency valuation
python query.py "SELECT * FROM valuation.metrics_multicurrency LIMIT 5"
```

### Learn More

- **Full Technical Details**: See `reference/PHASE2_IMPLEMENTATION_SUMMARY.md`
- **Test Coverage**: See `tests/test_phase2_global_markets.py`
- **API Reference**: See `src/` module docstrings

### Get Help

If you encounter issues:

1. Check `logs/` directory for error details
2. Run `python finangpt.py status` for health check
3. Review troubleshooting section above
4. Check test suite: `pytest tests/test_phase2_global_markets.py -v`

---

**Happy global investing! 🌍📈**

*Last Updated: 2025-11-09 | Phase 2: Global Market Support*
