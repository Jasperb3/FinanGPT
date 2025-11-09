# FinanGPT Python Implementation Instructions

---

## 1. Overview

This document outlines the plan to build a Python-based version of **FinanGPT**, equivalent in function to the Node.js reference. The Python system must replicate its ingestion, processing, storage, and natural language query features.

It should:

* Ingest **US-only** stock financials (annual & quarterly) from `yfinance`.
* Filter **non-ETF** tickers and **non-USD** currency statements.
* Merge Balance Sheet, Cash Flow, and Income Statement data into one schema per reporting period.
* Store data in **MongoDB (raw)** and **DuckDB (analytical)**.
* Serve as a **LLM-to-SQL interface** via Ollama.

---

## 2. Environment Setup

### Environment Variables (`.env`)

```bash
MONGO_URI=mongodb://localhost:27017/financial_data
OLLAMA_URL=http://localhost:11434
MODEL_NAME=gpt-oss:latest
```

### Dependencies

```bash
pip install yfinance pandas duckdb pymongo requests python-dotenv
```

### Directory Layout

```
project/
  |-- data/
  |-- logs/
  |-- tickers.csv
  |-- ingest.py
  |-- transform.py
  |-- query.py
  |-- .env
  |-- README.md
  |-- instructions.md
```

---

## 3. Data Ingestion (`ingest.py`)

### 3.1. Input Source

* `tickers.csv` contains one column: `ticker`
* Each batch processes **up to 50 tickers**
* Retries with exponential backoff (3 attempts per ticker)

### 3.2. yFinance Extraction

* Pull `financials`, `balancesheet`, `cashflow`, and their quarterly variants.
* Keep only **USD** and **non-ETF** instruments.
* Map variable field names (e.g., `Net Income`, `Net Income Applicable To Common Shares`).

### 3.3. Transformation & Merge

* Merge statements on `date`.
* Add `ticker`, `statement_type` (`annual`/`quarterly`), and metadata.
* Select **numeric-only columns**, plus `ticker` and `date`.
* Convert `date` to UTC at **16:00 US/Eastern**.

### 3.4. Storage (Raw Layer)

* Store in MongoDB collections:

  * `raw_annual`
  * `raw_quarterly`
* Use compound index `{ticker: 1, date: 1}` with upsert logic.
* Schema example:

  ```json
  {
    "ticker": "AAPL",
    "date": "2024-12-31T21:00:00Z",
    "payload": { "Net Income": 123456000000, "Total Assets": 394000000000 }
  }
  ```

---

## 4. Processed Layer (`transform.py`)

### 4.1. Extract from MongoDB

* Read from `raw_annual` and `raw_quarterly`
* Flatten payloads and infer numeric dtypes

### 4.2. Load into DuckDB

* Create database: `financial_data.duckdb`
* Tables:

  * `financials.annual`
  * `financials.quarterly`

Example DuckDB schema:

```sql
CREATE TABLE financials.annual (
  ticker VARCHAR,
  date DATE,
  netIncome DOUBLE,
  totalAssets DOUBLE,
  totalLiabilities DOUBLE,
  operatingCashFlow DOUBLE,
  revenue DOUBLE
);
```

Use efficient insertion:

```python
con.register('df', processed_df)
con.execute("INSERT OR REPLACE INTO financials.annual SELECT * FROM df")
```

---

## 5. Query Interface (`query.py`)

### 5.1. Schema-driven Prompt

The system prompt dynamically reflects the current schema from DuckDB.

Example (auto-generated at runtime):

```text
You are FinanGPT, a financial SQL assistant.
Your task is to generate DuckDB-compatible SQL based on the schema below.

Current schema:

financials.annual(ticker, date, netIncome, totalAssets, totalLiabilities, operatingCashFlow, revenue)
financials.quarterly(ticker, date, netIncome, totalAssets, totalLiabilities, operatingCashFlow, revenue)

Rules:
- Always include `date` in results.
- Default LIMIT is 25; maximum is 100.
- Only SELECT statements are allowed.
- Prefer recent data.

Example queries:
1. "Show Apple's annual net income for the past 5 years"
→ `SELECT date, netIncome FROM financials.annual WHERE ticker='AAPL' ORDER BY date DESC LIMIT 5;`

2. "Compare Microsoft and Google revenue growth over time"
→ `SELECT date, ticker, revenue FROM financials.annual WHERE ticker IN ('MSFT', 'GOOGL') ORDER BY date;`
```

### 5.2. Ollama Query Example

```python
import requests, os, json

OLLAMA_URL = os.getenv('OLLAMA_URL')
MODEL_NAME = os.getenv('MODEL_NAME')

def query_llm(nl_query):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_query}
        ]
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 5.3. SQL Safety Checks

* Allow-list table names (`financials.annual`, `financials.quarterly`).
* Reject DDL/DML or multi-statement SQL.
* Validate column names before execution.

---

## 6. Testing & Validation

### 6.1. Local Test

```bash
python ingest.py --tickers AAPL,MSFT
python transform.py
python query.py
```

### 6.2. Validation Checklist

* [ ] `raw_annual` and `raw_quarterly` contain upserted data.
* [ ] DuckDB contains matching row counts.
* [ ] NL queries generate valid SQL with date column.

---

## 7. Logging & Monitoring

* Write logs as JSON: `{ticker, phase, duration_ms, rows}`
* Save under `logs/ingest_YYYYMMDD.log`
* Log levels: INFO (normal), ERROR (failures)

---

## 8. Summary of Key Rules

| Area           | Rule                                        |
| -------------- | ------------------------------------------- |
| Tickers        | US-only, non-ETF                            |
| Currency       | USD only                                    |
| Date           | Normalised to US/Eastern 16:00 → UTC        |
| Tables         | `financials.annual`, `financials.quarterly` |
| Schema         | Numeric columns + ticker + date             |
| Query          | Read-only, DuckDB SQL, include date         |
| Limit          | Default 25, Max 100                         |
| Error Handling | Retry 3× with backoff, skip failures        |
| Indexes        | `{ticker, date}` unique in Mongo            |
| Logs           | Structured JSON per phase                   |

---

## 9. Immediate Implementation Tasks

1. **Data Ingestion Pipeline**

   * Implement `ingest.py` to read tickers, fetch financial statements, and upsert raw data to MongoDB.
   * Ensure retries and error handling are implemented using exponential backoff.
   * Verify US-only and USD currency filtering.

2. **Transformation Layer**

   * Implement `transform.py` to read from MongoDB, flatten payloads, and load into DuckDB.
   * Validate schema inference and ensure numeric-only column selection.
   * Test schema consistency across tickers.

3. **Query Interface**

   * Build `query.py` to generate schema-aware prompts and call Ollama's `/api/chat`.
   * Add SQL safety checks and allow-list enforcement.

4. **Testing & Validation**

   * Run ingestion for 2–3 tickers (AAPL, MSFT, GOOGL).
   * Verify counts in `raw_annual`, `raw_quarterly`, and DuckDB tables.
   * Test multiple NL queries to validate SQL generation.

5. **Documentation & Logging**

   * Create README instructions mirroring these steps.
   * Add structured logging and verify error outputs.

---

## 10. Future Enhancements

1. **Financial Ratio Expansion**

   * Compute derived ratios (ROE, ROA, Profit Margin, Debt/Equity) dynamically at query-time.
   * Include ratio definitions in schema prompts for LLM interpretation.

2. **Performance & Caching**

   * Cache LLM-generated SQL for repeated queries.
   * Cache query results in DuckDB for offline access.

3. **Data Quality Assurance**

   * Implement duplicate detection and anomaly checks on numeric values.
   * Add schema versioning metadata to MongoDB documents.

4. **Visualisation Layer**

   * Integrate lightweight chart rendering (e.g., matplotlib) for time-series insights.
   * Offer CSV/JSON export of query results.

5. **Deployment & Automation**

   * Package Docker image with preconfigured MongoDB and DuckDB.
   * Schedule nightly data refresh via cron or Airflow DAG.

6. **Testing & QA**

   * Add unit tests with mock data for all components.
   * Create integration tests for the full NL → SQL → DuckDB flow.

---

**End of Document**
