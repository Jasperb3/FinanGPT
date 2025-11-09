#!/usr/bin/env python3
"""Download and persist raw financial statements for FinanGPT."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, time as dt_time
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

MAX_TICKERS_PER_RUN = 50
MAX_ATTEMPTS = 3
EST = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")
LOGS_DIR = Path("logs")
PRICE_LOOKBACK_DAYS = int(os.getenv("PRICE_LOOKBACK_DAYS", "365"))

ALLOWED_EQUITY_TYPES = {
    "EQUITY",
    "COMMON STOCK",
    "COMMONSTOCK",
    "STOCK",
    "PREFERRED STOCK",
    "ETF",  # kept so we can fail closed when quoteType is empty
}

FIELD_MAPPINGS: Dict[str, Sequence[str]] = {
    "netIncome": [
        "Net Income",
        "NetIncome",
        "Net Income Applicable To Common Shares",
        "Net Income Common Stockholders",
    ],
    "netIncomeApplicableToCommonShares": [
        "Net Income Applicable To Common Shares",
        "Net Income Common Stockholders",
    ],
    "totalRevenue": [
        "Total Revenue",
        "Revenue",
        "Revenues",
    ],
    "totalAssets": [
        "Total Assets",
        "TotalAssets",
    ],
    "totalLiabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
        "TotalLiab",
    ],
    "cashAndCashEquivalents": [
        "Cash And Cash Equivalents",
        "Cash",
        "Cash And Cash Equivalents At Carrying Value",
    ],
    "operatingCashFlow": [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
    ],
    "grossProfit": [
        "Gross Profit",
    ],
    "researchAndDevelopment": [
        "Research And Development",
        "Research Development",
    ],
    "ebit": [
        "EBIT",
        "Ebit",
    ],
    "ebitda": [
        "EBITDA",
        "Ebitda",
    ],
    "shareholderEquity": [
        "Total Stockholder Equity",
        "Total Equity Gross Minority Interest",
    ],
    "freeCashFlow": [
        "Free Cash Flow",
    ],
}

ESTIMATED_BACKOFF_SECONDS = (1, 2, 4)


class UnsupportedInstrument(RuntimeError):
    """Raised when a ticker fails ETF, currency, or domicile checks."""


class StatementDownloadError(RuntimeError):
    """Raised when financial statements cannot be retrieved."""


def configure_logger() -> logging.Logger:
    """Initialise a JSON logger that writes to logs/ and stdout."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("ingest")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_path = LOGS_DIR / f"ingest_{datetime.utcnow():%Y%m%d}.log"
    file_handler = logging.FileHandler(file_path)
    stream_handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_event(logger: logging.Logger, **payload: Any) -> None:
    """Emit a structured JSON record."""
    entry = {"ts": datetime.utcnow().isoformat(), **payload}
    logger.info(json.dumps(entry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw statements via yfinance.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tickers",
        help="Comma-separated tickers to ingest (max 50 per run).",
    )
    group.add_argument(
        "--tickers-file",
        help="Path to CSV containing a 'ticker' column.",
    )
    return parser.parse_args()


def read_tickers(args: argparse.Namespace, logger: logging.Logger) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(_split_tickers(args.tickers))
    if args.tickers_file:
        tickers.extend(_read_tickers_file(args.tickers_file))
    tickers = [t.upper() for t in tickers if t]
    # preserve order while deduplicating
    seen: set[str] = set()
    ordered = []
    for ticker in tickers:
        if ticker not in seen:
            ordered.append(ticker)
            seen.add(ticker)
    if not ordered:
        raise SystemExit("No tickers provided.")
    if len(ordered) > MAX_TICKERS_PER_RUN:
        log_event(
            logger,
            phase="ingest",
            ticker="*",
            level="warning",
            message=f"Limiting run to first {MAX_TICKERS_PER_RUN} tickers.",
        )
        ordered = ordered[:MAX_TICKERS_PER_RUN]
    return ordered


def _split_tickers(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _read_tickers_file(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"Tickers file not found: {path}")
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    first_col = rows[0][0].strip().lower() if rows[0] else ""
    start_idx = 1 if first_col == "ticker" else 0
    extracted = []
    for row in rows[start_idx:]:
        if not row:
            continue
        extracted.append(row[0].strip())
    return extracted


def load_database(client: MongoClient, mongo_uri: str) -> Database:
    try:
        db = client.get_default_database()
        if db:
            return db
    except Exception:
        pass
    parsed = urlparse(mongo_uri)
    db_name = parsed.path.strip("/")
    if not db_name:
        raise SystemExit("The Mongo URI must include a database name.")
    return client[db_name]


def ensure_indexes(collection: Collection) -> None:
    collection.create_index([("ticker", 1), ("date", 1)], unique=True)


def ensure_supported_instrument(symbol: str) -> Tuple[Any, Mapping[str, Any], str]:
    ticker_obj = yf.Ticker(symbol)
    info = getattr(ticker_obj, "info", {}) or {}
    if not info:
        raise UnsupportedInstrument("Missing instrument metadata.")
    if is_etf(info):
        raise UnsupportedInstrument("ETF instruments are not supported.")
    if not is_us_listing(info):
        raise UnsupportedInstrument("Only US-listed equities are supported.")
    if not has_usd_financials(info):
        raise UnsupportedInstrument("Financial statements must be denominated in USD.")
    currency = str(info.get("financialCurrency") or info.get("currency") or "").upper()
    return ticker_obj, info, currency


def is_etf(info: Mapping[str, Any]) -> bool:
    quote_type = _as_text(info.get("quoteType")).upper()
    if quote_type == "ETF":
        return True
    if not quote_type:
        return True  # fail closed when Yahoo omits the quoteType
    if quote_type not in ALLOWED_EQUITY_TYPES:
        return True
    boolean_flags = (
        info.get("isETF"),
        info.get("isEtf"),
        info.get("fundFamily"),
    )
    if any(str(flag).lower() == "true" for flag in boolean_flags):
        return True
    long_name = _as_text(info.get("longName")).upper()
    return " ETF" in f" {long_name} "


def is_us_listing(info: Mapping[str, Any]) -> bool:
    country = _as_text(info.get("country")).lower()
    if country in {"united states", "united states of america", "usa", "us"}:
        return True
    market = _as_text(info.get("market")).lower()
    return market in {"us_market", "us"}


def has_usd_financials(info: Mapping[str, Any]) -> bool:
    currency = _as_text(info.get("financialCurrency") or info.get("currency")).upper()
    return currency == "USD"


def ingest_symbol(
    symbol: str,
    collections: Mapping[str, Collection],
    logger: logging.Logger,
) -> None:
    start = time.time()
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            ticker_obj, info, currency = ensure_supported_instrument(symbol)
            payloads = fetch_period_payloads(symbol, ticker_obj)
            if not any(payloads.values()):
                log_event(
                    logger,
                    phase="ingest",
                    ticker=symbol,
                    attempts=attempts,
                    rows=0,
                    duration_ms=_duration_ms(start),
                    message="No statements returned.",
                )
                return
            for period, data in payloads.items():
                rows = upsert_payloads(collections[period], symbol, period, data, currency, info)
                log_event(
                    logger,
                    phase=f"ingest.{period}",
                    ticker=symbol,
                    attempts=attempts,
                    rows=rows,
                    duration_ms=_duration_ms(start),
                )
                # Track metadata for financials
                update_ingestion_metadata(
                    collections["metadata"],
                    symbol,
                    f"financials_{period}",
                    "success",
                    rows,
                )

            # Fetch and store additional data types
            try:
                # Price history
                price_df = fetch_price_history(ticker_obj, symbol)
                if price_df is not None:
                    price_rows = upsert_price_history(collections["prices"], symbol, price_df)
                    log_event(logger, phase="ingest.prices", ticker=symbol, rows=price_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "prices_daily", "success", price_rows)

                # Dividends
                div_df = fetch_dividends(ticker_obj)
                if div_df is not None:
                    div_rows = upsert_dividends(collections["dividends"], symbol, div_df)
                    log_event(logger, phase="ingest.dividends", ticker=symbol, rows=div_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "dividends_history", "success", div_rows)

                # Stock splits
                split_df = fetch_splits(ticker_obj)
                if split_df is not None:
                    split_rows = upsert_splits(collections["splits"], symbol, split_df)
                    log_event(logger, phase="ingest.splits", ticker=symbol, rows=split_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "splits_history", "success", split_rows)

                # Company metadata
                metadata = extract_company_metadata(info)
                meta_rows = upsert_company_metadata(collections["company_metadata"], symbol, metadata)
                log_event(logger, phase="ingest.metadata", ticker=symbol, rows=meta_rows, duration_ms=_duration_ms(start))
                update_ingestion_metadata(collections["metadata"], symbol, "company_metadata", "success", meta_rows)

            except Exception as data_err:
                log_event(logger, phase="ingest.additional_data", ticker=symbol, error=str(data_err))

            return
        except UnsupportedInstrument as exc:
            log_event(
                logger,
                phase="skip",
                ticker=symbol,
                attempts=attempts,
                rows=0,
                duration_ms=_duration_ms(start),
                error=str(exc),
            )
            return
        except StatementDownloadError as exc:
            if attempts >= MAX_ATTEMPTS:
                log_event(
                    logger,
                    phase="error",
                    ticker=symbol,
                    attempts=attempts,
                    rows=0,
                    duration_ms=_duration_ms(start),
                    error=str(exc),
                )
                return
            backoff = ESTIMATED_BACKOFF_SECONDS[min(attempts - 1, len(ESTIMATED_BACKOFF_SECONDS) - 1)]
            time.sleep(backoff)
        except PyMongoError as exc:
            log_event(
                logger,
                phase="error",
                ticker=symbol,
                attempts=attempts,
                rows=0,
                duration_ms=_duration_ms(start),
                error=f"Mongo error: {exc}",
            )
            return


def fetch_period_payloads(symbol: str, ticker_obj: Any) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    try:
        annual_income = dataframe_to_records(getattr(ticker_obj, "financials", None))
        annual_balance = dataframe_to_records(getattr(ticker_obj, "balance_sheet", None))
        annual_cashflow = dataframe_to_records(getattr(ticker_obj, "cashflow", None))

        quarterly_income = dataframe_to_records(getattr(ticker_obj, "quarterly_financials", None))
        quarterly_balance = dataframe_to_records(getattr(ticker_obj, "quarterly_balance_sheet", None))
        quarterly_cashflow = dataframe_to_records(getattr(ticker_obj, "quarterly_cashflow", None))
    except Exception as err:
        raise StatementDownloadError(f"yfinance error: {err}") from err

    return {
        "annual": combine_statements(annual_income, annual_balance, annual_cashflow),
        "quarterly": combine_statements(quarterly_income, quarterly_balance, quarterly_cashflow),
    }


def dataframe_to_records(frame: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    if frame is None or frame.empty:
        return {}
    records: Dict[str, Dict[str, float]] = {}
    transpose = frame.T
    for period_end, series in transpose.iterrows():
        try:
            date_key = normalise_reporting_date(period_end)
        except ValueError:
            continue
        payload: Dict[str, float] = {}
        for raw_key, value in series.items():
            if pd.isna(value) or not is_numeric(value):
                continue
            canonical = normalise_field_name(str(raw_key))
            payload[canonical] = float(value)
        if payload:
            records[date_key] = payload
    return records


def combine_statements(
    income: Mapping[str, Dict[str, float]],
    balance: Mapping[str, Dict[str, float]],
    cashflow: Mapping[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_dates = set(income) | set(balance) | set(cashflow)
    for date in sorted(all_dates):
        sections = {
            "income_statement": income.get(date, {}),
            "balance_sheet": balance.get(date, {}),
            "cash_flow": cashflow.get(date, {}),
        }
        if any(section for section in sections.values()):
            combined[date] = sections
    return combined


def upsert_payloads(
    collection: Collection,
    ticker: str,
    period: str,
    payloads: Mapping[str, Dict[str, Dict[str, float]]],
    currency: str,
    info: Mapping[str, Any],
) -> int:
    if not payloads:
        return 0
    operations: List[UpdateOne] = []
    fetched_at = datetime.utcnow().isoformat()
    for date_key, payload in payloads.items():
        document = {
            "ticker": ticker,
            "date": date_key,
            "period": period,
            "currency": currency,
            "payload": payload,
            "source": "yfinance",
            "fetched_at": fetched_at,
            "instrument": {
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "exchange": info.get("exchange"),
            },
        }
        operations.append(
            UpdateOne(
                {"ticker": ticker, "date": date_key},
                {"$set": document},
                upsert=True,
            )
        )
    if not operations:
        return 0
    result = collection.bulk_write(operations, ordered=False)
    return len(operations) if result else 0


def normalise_reporting_date(raw_date: Any) -> str:
    if raw_date is None:
        raise ValueError("Missing reporting date.")
    ts = pd.Timestamp(raw_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(UTC)
    date_only = ts.to_pydatetime().date()
    as_eastern = datetime.combine(date_only, dt_time(hour=16, tzinfo=EST))
    return as_eastern.astimezone(UTC).isoformat()


def normalise_field_name(raw_key: str) -> str:
    key = raw_key.strip()
    for canonical, variants in FIELD_MAPPINGS.items():
        if key in variants:
            return canonical
        lowered_variants = [variant.lower() for variant in variants]
        if key.lower() in lowered_variants:
            return canonical
    tokens = re.split(r"[^0-9a-zA-Z]+", key)
    tokens = [token for token in tokens if token]
    if not tokens:
        return key
    camel = tokens[0].lower() + "".join(token.title() for token in tokens[1:])
    return camel


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Number):
        return math.isfinite(float(value))
    return False


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _duration_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def fetch_price_history(ticker_obj: Any, symbol: str, lookback_days: int = PRICE_LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Fetch daily price history (OHLCV) for the ticker."""
    try:
        history = ticker_obj.history(period=f"{lookback_days}d")
        if history.empty:
            return None
        # Reset index to get date as a column
        history = history.reset_index()
        # Ensure date is normalized
        if 'Date' in history.columns:
            history['date'] = pd.to_datetime(history['Date']).dt.date
            history = history.drop(columns=['Date'])
        return history
    except Exception as err:
        return None


def fetch_dividends(ticker_obj: Any) -> Optional[pd.DataFrame]:
    """Fetch dividend history for the ticker."""
    try:
        divs = ticker_obj.dividends
        if divs.empty:
            return None
        df = divs.reset_index()
        df.columns = ['date', 'amount']
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception:
        return None


def fetch_splits(ticker_obj: Any) -> Optional[pd.DataFrame]:
    """Fetch stock split history for the ticker."""
    try:
        splits = ticker_obj.splits
        if splits.empty:
            return None
        df = splits.reset_index()
        df.columns = ['date', 'ratio']
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception:
        return None


def extract_company_metadata(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract relevant company metadata from yfinance info dict."""
    return {
        "longName": info.get("longName"),
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "country": info.get("country"),
        "exchange": info.get("exchange"),
        "quoteType": info.get("quoteType"),
        "marketCap": info.get("marketCap"),
        "employees": info.get("fullTimeEmployees"),
        "description": info.get("longBusinessSummary"),
        "currency": info.get("currency"),
        "financialCurrency": info.get("financialCurrency"),
    }


def upsert_price_history(collection: Collection, ticker: str, price_df: pd.DataFrame) -> int:
    """Upsert price history to MongoDB."""
    if price_df is None or price_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.utcnow().isoformat()

    for _, row in price_df.iterrows():
        date_val = row['date']
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        date_iso = datetime.combine(date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        document = {
            "ticker": ticker,
            "date": date_iso,
            "open": float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
            "high": float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
            "low": float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
            "close": float(row.get('Close', 0)) if pd.notna(row.get('Close')) else None,
            "adj_close": float(row.get('Adj Close', 0)) if pd.notna(row.get('Adj Close')) else None,
            "volume": int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
            "fetched_at": fetched_at,
        }

        operations.append(
            UpdateOne(
                {"ticker": ticker, "date": date_iso},
                {"$set": document},
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
        return len(operations)
    return 0


def upsert_dividends(collection: Collection, ticker: str, div_df: pd.DataFrame) -> int:
    """Upsert dividend history to MongoDB."""
    if div_df is None or div_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.utcnow().isoformat()

    for _, row in div_df.iterrows():
        date_val = row['date']
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        date_iso = datetime.combine(date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        document = {
            "ticker": ticker,
            "date": date_iso,
            "amount": float(row['amount']),
            "fetched_at": fetched_at,
        }

        operations.append(
            UpdateOne(
                {"ticker": ticker, "date": date_iso},
                {"$set": document},
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
        return len(operations)
    return 0


def upsert_splits(collection: Collection, ticker: str, split_df: pd.DataFrame) -> int:
    """Upsert stock split history to MongoDB."""
    if split_df is None or split_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.utcnow().isoformat()

    for _, row in split_df.iterrows():
        date_val = row['date']
        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        date_iso = datetime.combine(date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        document = {
            "ticker": ticker,
            "date": date_iso,
            "ratio": float(row['ratio']),
            "fetched_at": fetched_at,
        }

        operations.append(
            UpdateOne(
                {"ticker": ticker, "date": date_iso},
                {"$set": document},
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
        return len(operations)
    return 0


def upsert_company_metadata(collection: Collection, ticker: str, metadata: Dict[str, Any]) -> int:
    """Upsert company metadata to MongoDB."""
    if not metadata:
        return 0

    fetched_at = datetime.utcnow().isoformat()
    document = {
        "ticker": ticker,
        "metadata": metadata,
        "fetched_at": fetched_at,
        "last_updated": fetched_at,
    }

    collection.update_one(
        {"ticker": ticker},
        {"$set": document},
        upsert=True,
    )
    return 1


def update_ingestion_metadata(
    collection: Collection,
    ticker: str,
    data_type: str,
    status: str,
    record_count: int = 0,
    last_successful_date: Optional[str] = None,
) -> None:
    """Track ingestion metadata for freshness monitoring."""
    document = {
        "ticker": ticker,
        "data_type": data_type,
        "last_fetched": datetime.utcnow().isoformat(),
        "status": status,
        "record_count": record_count,
    }
    if last_successful_date:
        document["last_successful_date"] = last_successful_date

    collection.update_one(
        {"ticker": ticker, "data_type": data_type},
        {"$set": document},
        upsert=True,
    )


def main() -> None:
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise SystemExit("MONGO_URI is not set. Please provide a Mongo connection string.")
    logger = configure_logger()
    args = parse_args()
    tickers = read_tickers(args, logger)
    with MongoClient(mongo_uri) as client:
        database = load_database(client, mongo_uri)
        annual_collection = database["raw_annual"]
        quarterly_collection = database["raw_quarterly"]
        prices_collection = database["stock_prices_daily"]
        dividends_collection = database["dividends_history"]
        splits_collection = database["splits_history"]
        metadata_collection = database["company_metadata"]
        ingestion_metadata_collection = database["ingestion_metadata"]

        # Ensure indexes
        ensure_indexes(annual_collection)
        ensure_indexes(quarterly_collection)
        ensure_indexes(prices_collection)
        ensure_indexes(dividends_collection)
        ensure_indexes(splits_collection)
        metadata_collection.create_index("ticker", unique=True)
        ingestion_metadata_collection.create_index([("ticker", 1), ("data_type", 1)], unique=True)

        collections = {
            "annual": annual_collection,
            "quarterly": quarterly_collection,
            "prices": prices_collection,
            "dividends": dividends_collection,
            "splits": splits_collection,
            "company_metadata": metadata_collection,
            "metadata": ingestion_metadata_collection,
        }
        for ticker in tickers:
            ingest_symbol(ticker, collections, logger)


if __name__ == "__main__":
    main()
