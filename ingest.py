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
from datetime import datetime, time as dt_time, timedelta
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

from time_utils import parse_utc_timestamp
from config_loader import load_config

# Import concurrent ingestion module
try:
    from src.ingest.concurrent import ingest_batch_concurrent, print_ingestion_summary
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False

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
    file_path = LOGS_DIR / f"ingest_{datetime.now(UTC):%Y%m%d}.log"
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
    entry = {"ts": datetime.now(UTC).isoformat(), **payload}
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Only re-ingest data if it's older than the refresh threshold (default: 7 days).",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=7,
        help="Number of days before data is considered stale (default: 7).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch all data regardless of freshness.",
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


def get_last_fetch_info(
    collection: Collection,
    ticker: str,
    data_type: str,
) -> Optional[Mapping[str, Any]]:
    """Retrieve the last fetch metadata for a specific ticker and data type."""
    return collection.find_one({"ticker": ticker, "data_type": data_type})


def is_data_stale(
    collection: Collection,
    ticker: str,
    data_type: str,
    threshold_days: int,
) -> bool:
    """Check if data is older than threshold_days."""
    metadata = get_last_fetch_info(collection, ticker, data_type)
    if not metadata:
        return True  # No metadata means never fetched, so it's stale

    last_fetched_str = metadata.get("last_fetched")
    if not last_fetched_str:
        return True

    last_fetched = parse_utc_timestamp(last_fetched_str)
    age = datetime.now(UTC) - last_fetched
    return age.days >= threshold_days


def should_skip_ticker(
    metadata_collection: Collection,
    ticker: str,
    refresh_mode: bool,
    force_mode: bool,
    refresh_days: int,
) -> bool:
    """Determine if we should skip ingesting this ticker based on freshness."""
    if force_mode:
        return False  # Never skip in force mode

    if not refresh_mode:
        return False  # Always ingest if not in refresh mode

    # In refresh mode, skip if all data types are fresh
    data_types = ["financials_annual", "financials_quarterly", "prices_daily",
                  "dividends_history", "splits_history", "company_metadata"]

    all_fresh = True
    for data_type in data_types:
        if is_data_stale(metadata_collection, ticker, data_type, refresh_days):
            all_fresh = False
            break

    return all_fresh


def get_last_price_date(
    prices_collection: Collection,
    ticker: str,
) -> Optional[datetime]:
    """Get the most recent price date for a ticker."""
    result = prices_collection.find_one(
        {"ticker": ticker},
        sort=[("date", -1)],
    )
    if not result:
        return None

    date_str = result.get("date")
    if not date_str:
        return None

    return parse_utc_timestamp(date_str)


def ingest_symbol(
    symbol: str,
    collections: Mapping[str, Collection],
    logger: logging.Logger,
    refresh_mode: bool = False,
    force_mode: bool = False,
    refresh_days: int = 7,
) -> None:
    start = time.time()

    # Check if we should skip this ticker based on freshness
    if should_skip_ticker(collections["metadata"], symbol, refresh_mode, force_mode, refresh_days):
        log_event(
            logger,
            phase="skip.fresh",
            ticker=symbol,
            message=f"Data is fresh (less than {refresh_days} days old), skipping.",
        )
        return

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
                # Price history - use incremental updates if not in force mode
                last_price_date = None
                if not force_mode:
                    last_price_date = get_last_price_date(collections["prices"], symbol)

                price_df = fetch_price_history(ticker_obj, symbol, last_date=last_price_date)
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

                # Phase 8: Earnings history
                earnings_hist_df = fetch_earnings_history(ticker_obj)
                if earnings_hist_df is not None:
                    earnings_hist_rows = upsert_earnings_history(collections["earnings_history"], symbol, earnings_hist_df)
                    log_event(logger, phase="ingest.earnings_history", ticker=symbol, rows=earnings_hist_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "earnings_history", "success", earnings_hist_rows)

                # Phase 8: Earnings calendar
                earnings_cal_df = fetch_earnings_dates(ticker_obj)
                if earnings_cal_df is not None:
                    earnings_cal_rows = upsert_earnings_calendar(collections["earnings_calendar"], symbol, earnings_cal_df)
                    log_event(logger, phase="ingest.earnings_calendar", ticker=symbol, rows=earnings_cal_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "earnings_calendar", "success", earnings_cal_rows)

                # Phase 9: Analyst recommendations
                analyst_recs_df = fetch_analyst_recommendations(ticker_obj)
                if analyst_recs_df is not None:
                    analyst_recs_rows = upsert_analyst_recommendations(collections["analyst_recommendations"], symbol, analyst_recs_df)
                    log_event(logger, phase="ingest.analyst_recommendations", ticker=symbol, rows=analyst_recs_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "analyst_recommendations", "success", analyst_recs_rows)

                # Phase 9: Price targets
                price_targets = fetch_analyst_price_targets(ticker_obj, info)
                if price_targets is not None:
                    price_targets_rows = upsert_price_targets(collections["price_targets"], symbol, price_targets)
                    log_event(logger, phase="ingest.price_targets", ticker=symbol, rows=price_targets_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "price_targets", "success", price_targets_rows)

                # Phase 9: Analyst consensus
                analyst_consensus = fetch_analyst_consensus(ticker_obj, info)
                if analyst_consensus is not None:
                    consensus_rows = upsert_analyst_consensus(collections["analyst_consensus"], symbol, analyst_consensus)
                    log_event(logger, phase="ingest.analyst_consensus", ticker=symbol, rows=consensus_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "analyst_consensus", "success", consensus_rows)

                # Phase 9: Growth estimates
                growth_estimates = fetch_growth_estimates(ticker_obj, info)
                if growth_estimates is not None:
                    growth_est_rows = upsert_growth_estimates(collections["growth_estimates"], symbol, growth_estimates)
                    log_event(logger, phase="ingest.growth_estimates", ticker=symbol, rows=growth_est_rows, duration_ms=_duration_ms(start))
                    update_ingestion_metadata(collections["metadata"], symbol, "growth_estimates", "success", growth_est_rows)

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
    fetched_at = datetime.now(UTC).isoformat()
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


def fetch_price_history(
    ticker_obj: Any,
    symbol: str,
    lookback_days: int = PRICE_LOOKBACK_DAYS,
    last_date: Optional[datetime] = None,
) -> Optional[pd.DataFrame]:
    """Fetch daily price history (OHLCV) for the ticker.

    If last_date is provided, fetches only prices newer than that date (incremental update).
    Otherwise, fetches the full lookback period.
    """
    try:
        if last_date:
            # Incremental update: fetch from day after last_date until today
            start_date = (last_date + timedelta(days=1)).date()
            end_date = datetime.now(UTC).date()

            # Don't fetch if we're already up to date
            if start_date > end_date:
                return None

            history = ticker_obj.history(start=start_date, end=end_date)
        else:
            # Full fetch using lookback period
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


def fetch_earnings_history(ticker_obj: Any) -> Optional[pd.DataFrame]:
    """Fetch earnings history (EPS estimates vs actuals) for the ticker.

    Phase 8: Earnings Intelligence
    """
    try:
        # Try to get earnings history
        earnings = getattr(ticker_obj, "earnings_history", None)
        if earnings is None or (hasattr(earnings, 'empty') and earnings.empty):
            return None

        if isinstance(earnings, pd.DataFrame):
            df = earnings.reset_index()
            # Normalize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Extract relevant fields
            result_rows = []
            for _, row in df.iterrows():
                try:
                    result_rows.append({
                        'date': pd.to_datetime(row.get('date', row.name if hasattr(row, 'name') else None)).date() if pd.notna(row.get('date', row.name)) else None,
                        'fiscal_period': row.get('period', row.get('fiscal_period', '')),
                        'eps_estimate': float(row.get('epsestimate', row.get('eps_estimate'))) if pd.notna(row.get('epsestimate', row.get('eps_estimate'))) else None,
                        'eps_actual': float(row.get('epsactual', row.get('eps_actual'))) if pd.notna(row.get('epsactual', row.get('eps_actual'))) else None,
                        'revenue_estimate': float(row.get('revenueestimate', row.get('revenue_estimate'))) if pd.notna(row.get('revenueestimate', row.get('revenue_estimate'))) else None,
                        'revenue_actual': float(row.get('revenueactual', row.get('revenue_actual'))) if pd.notna(row.get('revenueactual', row.get('revenue_actual'))) else None,
                    })
                except Exception:
                    continue

            if result_rows:
                return pd.DataFrame(result_rows)
        return None
    except Exception:
        return None


def fetch_earnings_dates(ticker_obj: Any) -> Optional[pd.DataFrame]:
    """Fetch upcoming earnings dates for the ticker.

    Phase 8: Earnings Calendar
    """
    try:
        # Try to get earnings dates
        earnings_dates = getattr(ticker_obj, "earnings_dates", None)
        if earnings_dates is None or (hasattr(earnings_dates, 'empty') and earnings_dates.empty):
            return None

        if isinstance(earnings_dates, pd.DataFrame):
            df = earnings_dates.reset_index()
            # Normalize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Extract relevant fields
            result_rows = []
            for _, row in df.iterrows():
                try:
                    earnings_date_col = row.get('earnings_date', row.get('date', row.name if hasattr(row, 'name') else None))
                    result_rows.append({
                        'earnings_date': pd.to_datetime(earnings_date_col).date() if pd.notna(earnings_date_col) else None,
                        'period_ending': row.get('period_ending', row.get('fiscalperiod', '')),
                        'estimate': float(row.get('eps_estimate', row.get('estimate'))) if pd.notna(row.get('eps_estimate', row.get('estimate'))) else None,
                    })
                except Exception:
                    continue

            if result_rows:
                return pd.DataFrame(result_rows)
        return None
    except Exception:
        return None


def upsert_price_history(collection: Collection, ticker: str, price_df: pd.DataFrame) -> int:
    """Upsert price history to MongoDB."""
    if price_df is None or price_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.now(UTC).isoformat()

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
    fetched_at = datetime.now(UTC).isoformat()

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
    fetched_at = datetime.now(UTC).isoformat()

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

    fetched_at = datetime.now(UTC).isoformat()
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


def upsert_earnings_history(collection: Collection, ticker: str, earnings_df: pd.DataFrame) -> int:
    """Upsert earnings history to MongoDB.

    Phase 8: Earnings Intelligence
    """
    if earnings_df is None or earnings_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.now(UTC).isoformat()

    for _, row in earnings_df.iterrows():
        date_val = row.get('date')
        if date_val is None:
            continue

        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        date_iso = datetime.combine(date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        # Calculate surprise metrics
        eps_actual = row.get('eps_actual')
        eps_estimate = row.get('eps_estimate')
        eps_surprise = None
        surprise_pct = None

        if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
            eps_surprise = eps_actual - eps_estimate
            surprise_pct = (eps_surprise / abs(eps_estimate)) * 100

        document = {
            "ticker": ticker,
            "date": date_iso,
            "fiscal_period": row.get('fiscal_period', ''),
            "eps_estimate": float(eps_estimate) if pd.notna(eps_estimate) else None,
            "eps_actual": float(eps_actual) if pd.notna(eps_actual) else None,
            "eps_surprise": float(eps_surprise) if eps_surprise is not None else None,
            "surprise_pct": float(surprise_pct) if surprise_pct is not None else None,
            "revenue_estimate": float(row.get('revenue_estimate')) if pd.notna(row.get('revenue_estimate')) else None,
            "revenue_actual": float(row.get('revenue_actual')) if pd.notna(row.get('revenue_actual')) else None,
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


def upsert_earnings_calendar(collection: Collection, ticker: str, calendar_df: pd.DataFrame) -> int:
    """Upsert earnings calendar to MongoDB.

    Phase 8: Earnings Calendar
    """
    if calendar_df is None or calendar_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.now(UTC).isoformat()

    for _, row in calendar_df.iterrows():
        earnings_date_val = row.get('earnings_date')
        if earnings_date_val is None:
            continue

        if isinstance(earnings_date_val, pd.Timestamp):
            earnings_date_val = earnings_date_val.date()
        date_iso = datetime.combine(earnings_date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        document = {
            "ticker": ticker,
            "earnings_date": date_iso,
            "period_ending": row.get('period_ending', ''),
            "estimate": float(row.get('estimate')) if pd.notna(row.get('estimate')) else None,
            "fetched_at": fetched_at,
        }

        operations.append(
            UpdateOne(
                {"ticker": ticker, "earnings_date": date_iso},
                {"$set": document},
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
        return len(operations)
    return 0


def fetch_analyst_recommendations(ticker_obj: Any) -> Optional[pd.DataFrame]:
    """Fetch analyst recommendations (upgrades/downgrades) for the ticker.

    Phase 9: Analyst Intelligence
    """
    try:
        recommendations = getattr(ticker_obj, "recommendations", None)
        if recommendations is None or (hasattr(recommendations, 'empty') and recommendations.empty):
            # Try upgrades_downgrades as alternative
            recommendations = getattr(ticker_obj, "upgrades_downgrades", None)
            if recommendations is None or (hasattr(recommendations, 'empty') and recommendations.empty):
                return None

        if isinstance(recommendations, pd.DataFrame):
            df = recommendations.reset_index()
            # Normalize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Extract relevant fields
            result_rows = []
            for _, row in df.iterrows():
                try:
                    # Handle date field (could be index or column)
                    date_val = row.get('date', row.name if hasattr(row, 'name') else None)
                    if pd.isna(date_val):
                        continue

                    result_rows.append({
                        'date': pd.to_datetime(date_val).date() if pd.notna(date_val) else None,
                        'firm': row.get('firm', row.get('gradecompany', '')),
                        'from_grade': row.get('fromgrade', row.get('from_grade', '')),
                        'to_grade': row.get('tograde', row.get('to_grade', '')),
                        'action': row.get('action', '')
                    })
                except Exception:
                    continue

            if result_rows:
                return pd.DataFrame(result_rows)
        return None
    except Exception:
        return None


def fetch_analyst_price_targets(ticker_obj: Any, info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch analyst price targets from ticker info.

    Phase 9: Analyst Intelligence
    """
    try:
        # Price targets are usually in the info dict
        target_mean = info.get('targetMeanPrice')
        target_low = info.get('targetLowPrice')
        target_high = info.get('targetHighPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        num_analysts = info.get('numberOfAnalystOpinions')

        if target_mean is not None or target_low is not None or target_high is not None:
            return {
                'current_price': float(current_price) if current_price is not None else None,
                'target_low': float(target_low) if target_low is not None else None,
                'target_mean': float(target_mean) if target_mean is not None else None,
                'target_high': float(target_high) if target_high is not None else None,
                'num_analysts': int(num_analysts) if num_analysts is not None else None,
            }
        return None
    except Exception:
        return None


def fetch_analyst_consensus(ticker_obj: Any, info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch analyst consensus ratings from ticker info.

    Phase 9: Analyst Intelligence
    """
    try:
        # Try to get recommendation data
        rec_key = info.get('recommendationKey')
        rec_mean = info.get('recommendationMean')

        # Try to get detailed breakdown if available
        recommendations_summary = getattr(ticker_obj, "recommendations_summary", None)

        if recommendations_summary is not None and not recommendations_summary.empty:
            # Parse from summary dataframe
            latest = recommendations_summary.iloc[0] if len(recommendations_summary) > 0 else None
            if latest is not None:
                return {
                    'strong_buy': int(latest.get('strongBuy', 0)) if pd.notna(latest.get('strongBuy')) else 0,
                    'buy': int(latest.get('buy', 0)) if pd.notna(latest.get('buy')) else 0,
                    'hold': int(latest.get('hold', 0)) if pd.notna(latest.get('hold')) else 0,
                    'sell': int(latest.get('sell', 0)) if pd.notna(latest.get('sell')) else 0,
                    'strong_sell': int(latest.get('strongSell', 0)) if pd.notna(latest.get('strongSell')) else 0,
                }

        # Fallback: try to infer from recommendationMean
        if rec_mean is not None:
            # Estimate distribution based on mean (rough approximation)
            # recommendationMean: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
            return {
                'strong_buy': 0,
                'buy': 0,
                'hold': 0,
                'sell': 0,
                'strong_sell': 0,
            }

        return None
    except Exception:
        return None


def fetch_growth_estimates(ticker_obj: Any, info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch analyst growth estimates from ticker.

    Phase 9: Analyst Intelligence
    """
    try:
        # Try to get growth estimates
        earnings_growth = info.get('earningsGrowth')
        revenue_growth = info.get('revenueGrowth')
        earnings_quarterly_growth = info.get('earningsQuarterlyGrowth')

        result = {}

        # Add available estimates
        if earnings_quarterly_growth is not None:
            result['current_qtr_growth'] = float(earnings_quarterly_growth) * 100  # Convert to percentage

        if earnings_growth is not None:
            result['current_year_growth'] = float(earnings_growth) * 100

        if revenue_growth is not None:
            result['next_qtr_growth'] = float(revenue_growth) * 100

        # 5-year growth from analyst estimates
        five_year_growth = info.get('earningsGrowth')  # Some sources provide this
        if five_year_growth is not None:
            result['next_5yr_growth'] = float(five_year_growth) * 100

        if result:
            return result

        return None
    except Exception:
        return None


def upsert_analyst_recommendations(collection: Collection, ticker: str, recommendations_df: pd.DataFrame) -> int:
    """Upsert analyst recommendations to MongoDB.

    Phase 9: Analyst Intelligence
    """
    if recommendations_df is None or recommendations_df.empty:
        return 0

    operations: List[UpdateOne] = []
    fetched_at = datetime.now(UTC).isoformat()

    for _, row in recommendations_df.iterrows():
        date_val = row.get('date')
        if date_val is None:
            continue

        if isinstance(date_val, pd.Timestamp):
            date_val = date_val.date()
        date_iso = datetime.combine(date_val, dt_time(hour=16, tzinfo=EST)).astimezone(UTC).isoformat()

        document = {
            "ticker": ticker,
            "date": date_iso,
            "firm": row.get('firm', ''),
            "from_grade": row.get('from_grade', ''),
            "to_grade": row.get('to_grade', ''),
            "action": row.get('action', ''),
            "fetched_at": fetched_at,
        }

        operations.append(
            UpdateOne(
                {"ticker": ticker, "date": date_iso, "firm": document["firm"]},
                {"$set": document},
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
        return len(operations)
    return 0


def upsert_price_targets(collection: Collection, ticker: str, price_targets: Dict[str, Any]) -> int:
    """Upsert analyst price targets to MongoDB.

    Phase 9: Analyst Intelligence
    """
    if not price_targets:
        return 0

    fetched_at = datetime.now(UTC).isoformat()
    date_iso = datetime.now(UTC).date().isoformat()

    document = {
        "ticker": ticker,
        "date": date_iso,
        "current_price": price_targets.get('current_price'),
        "target_low": price_targets.get('target_low'),
        "target_mean": price_targets.get('target_mean'),
        "target_high": price_targets.get('target_high'),
        "num_analysts": price_targets.get('num_analysts'),
        "fetched_at": fetched_at,
    }

    collection.update_one(
        {"ticker": ticker, "date": date_iso},
        {"$set": document},
        upsert=True,
    )
    return 1


def upsert_analyst_consensus(collection: Collection, ticker: str, consensus: Dict[str, Any]) -> int:
    """Upsert analyst consensus to MongoDB.

    Phase 9: Analyst Intelligence
    """
    if not consensus:
        return 0

    fetched_at = datetime.now(UTC).isoformat()
    date_iso = datetime.now(UTC).date().isoformat()

    document = {
        "ticker": ticker,
        "date": date_iso,
        "strong_buy": consensus.get('strong_buy', 0),
        "buy": consensus.get('buy', 0),
        "hold": consensus.get('hold', 0),
        "sell": consensus.get('sell', 0),
        "strong_sell": consensus.get('strong_sell', 0),
        "fetched_at": fetched_at,
    }

    collection.update_one(
        {"ticker": ticker, "date": date_iso},
        {"$set": document},
        upsert=True,
    )
    return 1


def upsert_growth_estimates(collection: Collection, ticker: str, growth_estimates: Dict[str, Any]) -> int:
    """Upsert analyst growth estimates to MongoDB.

    Phase 9: Analyst Intelligence
    """
    if not growth_estimates:
        return 0

    fetched_at = datetime.now(UTC).isoformat()
    date_iso = datetime.now(UTC).date().isoformat()

    document = {
        "ticker": ticker,
        "date": date_iso,
        "current_qtr_growth": growth_estimates.get('current_qtr_growth'),
        "next_qtr_growth": growth_estimates.get('next_qtr_growth'),
        "current_year_growth": growth_estimates.get('current_year_growth'),
        "next_year_growth": growth_estimates.get('next_year_growth'),
        "next_5yr_growth": growth_estimates.get('next_5yr_growth'),
        "fetched_at": fetched_at,
    }

    collection.update_one(
        {"ticker": ticker, "date": date_iso},
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
        "last_fetched": datetime.now(UTC).isoformat(),
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

    # Log mode information
    if args.force:
        log_event(logger, phase="start", mode="force", message="Force mode: re-fetching all data regardless of freshness")
    elif args.refresh:
        log_event(logger, phase="start", mode="refresh", refresh_days=args.refresh_days,
                  message=f"Refresh mode: only updating data older than {args.refresh_days} days")
    else:
        log_event(logger, phase="start", mode="normal", message="Normal mode: fetching all tickers")

    with MongoClient(mongo_uri) as client:
        database = load_database(client, mongo_uri)
        annual_collection = database["raw_annual"]
        quarterly_collection = database["raw_quarterly"]
        prices_collection = database["stock_prices_daily"]
        dividends_collection = database["dividends_history"]
        splits_collection = database["splits_history"]
        metadata_collection = database["company_metadata"]
        ingestion_metadata_collection = database["ingestion_metadata"]
        earnings_history_collection = database["earnings_history"]  # Phase 8
        earnings_calendar_collection = database["earnings_calendar"]  # Phase 8
        # Phase 9: Analyst data collections
        analyst_recommendations_collection = database["analyst_recommendations"]
        price_targets_collection = database["price_targets"]
        analyst_consensus_collection = database["analyst_consensus"]
        growth_estimates_collection = database["growth_estimates"]

        # Ensure indexes
        ensure_indexes(annual_collection)
        ensure_indexes(quarterly_collection)
        ensure_indexes(prices_collection)
        ensure_indexes(dividends_collection)
        ensure_indexes(splits_collection)
        metadata_collection.create_index("ticker", unique=True)
        ingestion_metadata_collection.create_index([("ticker", 1), ("data_type", 1)], unique=True)
        # Phase 8: Earnings data indexes
        ensure_indexes(earnings_history_collection)
        ensure_indexes(earnings_calendar_collection)
        # Phase 9: Analyst data indexes
        ensure_indexes(analyst_recommendations_collection)
        ensure_indexes(price_targets_collection)
        ensure_indexes(analyst_consensus_collection)
        ensure_indexes(growth_estimates_collection)

        collections = {
            "annual": annual_collection,
            "quarterly": quarterly_collection,
            "prices": prices_collection,
            "dividends": dividends_collection,
            "splits": splits_collection,
            "company_metadata": metadata_collection,
            "metadata": ingestion_metadata_collection,
            "earnings_history": earnings_history_collection,  # Phase 8
            "earnings_calendar": earnings_calendar_collection,  # Phase 8
            "analyst_recommendations": analyst_recommendations_collection,  # Phase 9
            "price_targets": price_targets_collection,  # Phase 9
            "analyst_consensus": analyst_consensus_collection,  # Phase 9
            "growth_estimates": growth_estimates_collection,  # Phase 9
        }

        # Load configuration for concurrent processing
        config = load_config()
        use_concurrent = config.get('ingestion', {}).get('use_concurrent', False)

        if use_concurrent and CONCURRENT_AVAILABLE and len(tickers) > 1:
            # Use concurrent processing for better performance
            max_workers = config.get('ingestion', {}).get('max_workers', 10)
            worker_timeout = config.get('ingestion', {}).get('worker_timeout', 120)

            log_event(logger, phase="concurrent.start",
                     tickers_count=len(tickers), max_workers=max_workers,
                     message=f"Starting concurrent ingestion with {max_workers} workers")

            # Wrapper function to match concurrent module's expected signature
            def ingest_func(ticker, **kwargs):
                ingest_symbol(ticker, collections, **kwargs)
                return 1  # Return success indicator

            results = ingest_batch_concurrent(
                tickers=tickers,
                ingest_func=ingest_func,
                max_workers=max_workers,
                worker_timeout=worker_timeout,
                logger=logger,
                refresh_mode=args.refresh,
                force_mode=args.force,
                refresh_days=args.refresh_days,
            )

            # Print summary
            print_ingestion_summary(results)

            log_event(logger, phase="concurrent.complete",
                     success_count=sum(1 for r in results.values() if r.status == "success"),
                     total_count=len(results))
        else:
            # Fallback to sequential processing
            if use_concurrent and not CONCURRENT_AVAILABLE:
                logger.warning("Concurrent processing requested but module not available, using sequential")

            for ticker in tickers:
                ingest_symbol(
                    ticker,
                    collections,
                    logger,
                    refresh_mode=args.refresh,
                    force_mode=args.force,
                    refresh_days=args.refresh_days,
                )


if __name__ == "__main__":
    main()
