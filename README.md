# FinanGPT Python (LLM→SQL)

A fully Pythonic recreation of the FinanGPT pipeline: fetch audited US financial statements via `yfinance`, keep a raw history in MongoDB, normalise it into DuckDB for analytics, and answer natural-language questions through a guarded LLM→SQL layer backed by Ollama.

## Quickstart

1. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the required packages**

   ```bash
   pip install yfinance pandas duckdb pymongo requests python-dotenv
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # edit MONGO_URI / OLLAMA_URL / MODEL_NAME as needed
   ```

4. **Start dependencies**

   - MongoDB running locally (`mongod --config ...`).
   - Ollama running with your chosen model pulled (`ollama pull gpt-oss:latest`).

5. **Load tickers**

   Edit `tickers.csv` (single `ticker` column). You can also pass tickers inline:

   ```bash
   python ingest.py --tickers AAPL,MSFT,GOOGL
   # or
   python ingest.py --tickers-file tickers.csv
   ```

6. **Transform to DuckDB**

   ```bash
   python transform.py
   ```

   This creates/updates `financial_data.duckdb` with `financials.annual` and `financials.quarterly` tables.

7. **Ask questions**

   ```bash
   python query.py "annual net income for AAPL over the last 5 years"
   ```

   The script builds a schema-aware prompt, validates the SQL returned by Ollama, executes it against DuckDB, and prints a neat table.

## Tests

```bash
python -m pytest tests
```

The suite covers ingestion filters, transform schema guarantees, and SQL guardrails.

## Troubleshooting

- **Mongo not running** – `ingest.py`/`transform.py` will abort with `MONGO_URI is not set/running`. Start MongoDB and ensure the URI points to a reachable database (the URI must include the database name so unique indexes live in the right place).

- **Ollama not reachable** – `query.py` raises `ConnectionError`. Confirm `OLLAMA_URL` and that the requested `MODEL_NAME` is available (`ollama list`). You can temporarily bypass the LLM by pasting SQL directly into DuckDB for debugging.

- **Schema mismatch** – If `query.py` says “No DuckDB tables found”, rerun `transform.py`. If columns moved, delete `financial_data.duckdb` and rebuild to make sure the schema snapshot matches Mongo.

- **ETF/Non-USD rejection** – ETFs, non-USD statements, and non-US listings are skipped by design. Use common US equity tickers; the filter fails closed to avoid polluting the dataset.

## Operational Notes

- Only **US-listed**, **non-ETF**, **USD** instruments are ingested. Missing metadata is treated as a rejection to protect downstream quality.
- Dates are normalised to **16:00 US/Eastern**, then stored as UTC ISO 8601 in Mongo and as `DATE` in DuckDB.
- Mongo raw collections (`raw_annual`, `raw_quarterly`) upsert on `{ticker, date}` to avoid duplication; DuckDB loads are idempotent via delete-then-insert.
- Logs land under `logs/` as JSON for easy ingestion into observers.
- The query layer enforces table allow-lists, single SELECT statements, valid columns, and `LIMIT 25` (max `LIMIT 100`) to keep DuckDB safe.

## Next Steps

- Extend `FIELD_MAPPINGS` in `ingest.py` as new statement fields appear.
- Layer on derived ratios during transform or query time.
- Automate nightly refreshes and wire Ollama responses into dashboards.
