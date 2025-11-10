#!/usr/bin/env python3
"""Transform raw MongoDB snapshots into DuckDB analytical tables."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, date
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import duckdb
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.core.time_utils import parse_utc_timestamp
from src.core.config_loader import load_config

# Import streaming transformation module
try:
    from src.transform.streaming import transform_with_streaming
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

LOGS_DIR = Path("logs")
DUCKDB_PATH = "financial_data.duckdb"
ANNUAL_TABLE = "financials.annual"
QUARTERLY_TABLE = "financials.quarterly"


def configure_logger() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("transform")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(LOGS_DIR / f"transform_{datetime.now(UTC):%Y%m%d}.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_event(logger: logging.Logger, **payload: Any) -> None:
    entry = {"ts": datetime.now(UTC).isoformat(), **payload}
    logger.info(json.dumps(entry))


def load_database(client: MongoClient, mongo_uri: str) -> Database:
    try:
        db = client.get_default_database()
        if db:
            return db
    except Exception:
        pass
    path = mongo_uri.rsplit("/", 1)[-1]
    if not path:
        raise SystemExit("Mongo URI must contain a database name.")
    return client[path]


def fetch_documents(collection: Collection) -> List[Mapping[str, Any]]:
    return list(collection.find({}))


def parse_iso_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    iso_str = str(value or "")
    if not iso_str:
        raise ValueError("Missing date field.")
    parsed = parse_utc_timestamp(iso_str)
    return parsed.date()


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Number):
        return True
    return False


def prepare_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten Mongo payloads into a numeric-only DataFrame."""
    rows: List[Dict[str, Any]] = []
    for document in documents:
        payload = document.get("payload", {})
        ticker = document.get("ticker")
        date_value = document.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue
        combined: Dict[str, float] = {}
        for section_name in ("income_statement", "balance_sheet", "cash_flow"):
            section = payload.get(section_name, {})
            for field, value in section.items():
                if is_numeric(value):
                    combined[field] = float(value)
        if not combined:
            continue
        row = {"ticker": ticker, "date": parsed_date, **combined}
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date"])
    frame = pd.DataFrame(rows)
    numeric_columns = [col for col in frame.columns if col not in {"ticker", "date"}]
    ordered_numeric = sorted(numeric_columns)
    frame = frame[["ticker", "date", *ordered_numeric]]
    return frame


def upsert_dataframe(
    conn: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    table: str,
    schema: str | None = None,
    key_columns: Sequence[str] | None = None,
) -> int:
    if frame.empty:
        return 0
    view_name = f"staging_{table.replace('.', '_')}"
    if schema:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    conn.register(view_name, frame)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {view_name} LIMIT 0")
    ensure_columns(conn, table, frame)
    key_columns = key_columns or ["ticker", "date"]
    missing_keys = [col for col in key_columns if col not in frame.columns]
    if missing_keys:
        raise ValueError(f"{table} is missing key columns required for upsert: {missing_keys}")
    key_conditions = " AND ".join(
        f'{table}."{col}" = {view_name}."{col}"'
        for col in key_columns
    )
    conn.execute(
        f"""
        DELETE FROM {table}
        USING {view_name}
        WHERE {key_conditions}
        """
    )
    column_list = ", ".join(f'"{col}"' for col in frame.columns)
    conn.execute(f'INSERT INTO {table} ({column_list}) SELECT {column_list} FROM {view_name}')
    conn.unregister(view_name)
    return len(frame)


def ensure_columns(conn: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    """Ensure DuckDB table has all columns present in the incoming DataFrame."""
    info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    if not info:
        return
    existing = {row[1] for row in info}
    for column in frame.columns:
        if column in existing:
            continue
        duck_type = infer_duckdb_type(frame[column], column)
        conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" {duck_type}')
        existing.add(column)


def infer_duckdb_type(series: pd.Series, column: str) -> str:
    """Infer an appropriate DuckDB column type for a pandas Series."""
    if column == "ticker":
        return "VARCHAR"
    if column == "date":
        return "DATE"
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "VARCHAR"


def prepare_prices_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform price history documents into DataFrame."""
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "open": doc.get("open"),
            "high": doc.get("high"),
            "low": doc.get("low"),
            "close": doc.get("close"),
            "adj_close": doc.get("adj_close"),
            "volume": doc.get("volume"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])
    return pd.DataFrame(rows)


def prepare_dividends_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform dividend documents into DataFrame."""
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "amount": doc.get("amount"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "amount"])
    return pd.DataFrame(rows)


def prepare_splits_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform stock split documents into DataFrame."""
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "ratio": doc.get("ratio"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "ratio"])
    return pd.DataFrame(rows)


def prepare_metadata_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform company metadata documents into DataFrame."""
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        metadata = doc.get("metadata", {})
        if not ticker:
            continue

        row = {
            "ticker": ticker,
            "longName": metadata.get("longName"),
            "shortName": metadata.get("shortName"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "website": metadata.get("website"),
            "country": metadata.get("country"),
            "exchange": metadata.get("exchange"),
            "quoteType": metadata.get("quoteType"),
            "marketCap": metadata.get("marketCap"),
            "employees": metadata.get("employees"),
            "description": metadata.get("description"),
            "currency": metadata.get("currency"),
            "financialCurrency": metadata.get("financialCurrency"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "longName", "shortName", "sector", "industry", "website",
                                      "country", "exchange", "quoteType", "marketCap", "employees",
                                      "description", "currency", "financialCurrency"])
    return pd.DataFrame(rows)


def prepare_earnings_history_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform earnings history documents into DataFrame.

    Phase 8: Earnings Intelligence
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "report_date": parsed_date,
            "fiscal_period": doc.get("fiscal_period"),
            "eps_estimate": doc.get("eps_estimate"),
            "eps_actual": doc.get("eps_actual"),
            "eps_surprise": doc.get("eps_surprise"),
            "surprise_pct": doc.get("surprise_pct"),
            "revenue_estimate": doc.get("revenue_estimate"),
            "revenue_actual": doc.get("revenue_actual"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "report_date", "fiscal_period", "eps_estimate",
                                      "eps_actual", "eps_surprise", "surprise_pct",
                                      "revenue_estimate", "revenue_actual"])
    return pd.DataFrame(rows)


def prepare_earnings_calendar_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform earnings calendar documents into DataFrame.

    Phase 8: Earnings Calendar
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        earnings_date_value = doc.get("earnings_date")
        if not ticker or not earnings_date_value:
            continue
        try:
            parsed_date = parse_iso_date(earnings_date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "earnings_date": parsed_date,
            "period_ending": doc.get("period_ending"),
            "estimate": doc.get("estimate"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date", "period_ending", "estimate"])
    return pd.DataFrame(rows)


def prepare_analyst_recommendations_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform analyst recommendations documents into DataFrame.

    Phase 9: Analyst Intelligence
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "firm": doc.get("firm"),
            "from_grade": doc.get("from_grade"),
            "to_grade": doc.get("to_grade"),
            "action": doc.get("action"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "firm", "from_grade", "to_grade", "action"])
    return pd.DataFrame(rows)


def prepare_price_targets_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform price targets documents into DataFrame.

    Phase 9: Analyst Intelligence
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "current_price": doc.get("current_price"),
            "target_low": doc.get("target_low"),
            "target_mean": doc.get("target_mean"),
            "target_high": doc.get("target_high"),
            "num_analysts": doc.get("num_analysts"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "current_price", "target_low", "target_mean", "target_high", "num_analysts"])
    return pd.DataFrame(rows)


def prepare_analyst_consensus_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform analyst consensus documents into DataFrame.

    Phase 9: Analyst Intelligence
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "strong_buy": doc.get("strong_buy", 0),
            "buy": doc.get("buy", 0),
            "hold": doc.get("hold", 0),
            "sell": doc.get("sell", 0),
            "strong_sell": doc.get("strong_sell", 0),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "strong_buy", "buy", "hold", "sell", "strong_sell"])
    return pd.DataFrame(rows)


def prepare_growth_estimates_dataframe(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Transform growth estimates documents into DataFrame.

    Phase 9: Analyst Intelligence
    """
    rows: List[Dict[str, Any]] = []
    for doc in documents:
        ticker = doc.get("ticker")
        date_value = doc.get("date")
        if not ticker or not date_value:
            continue
        try:
            parsed_date = parse_iso_date(date_value)
        except ValueError:
            continue

        row = {
            "ticker": ticker,
            "date": parsed_date,
            "current_qtr_growth": doc.get("current_qtr_growth"),
            "next_qtr_growth": doc.get("next_qtr_growth"),
            "current_year_growth": doc.get("current_year_growth"),
            "next_year_growth": doc.get("next_year_growth"),
            "next_5yr_growth": doc.get("next_5yr_growth"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "current_qtr_growth", "next_qtr_growth", "current_year_growth", "next_year_growth", "next_5yr_growth"])
    return pd.DataFrame(rows)


def create_ratios_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create derived financial ratios table from annual financials."""
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS ratios")
        conn.execute("DROP TABLE IF EXISTS ratios.financial")
        result = conn.execute("""
            CREATE TABLE ratios.financial AS
            SELECT
                ticker,
                date,
                CASE WHEN totalRevenue > 0 THEN netIncome / totalRevenue ELSE NULL END AS net_margin,
                CASE WHEN shareholderEquity > 0 THEN netIncome / shareholderEquity ELSE NULL END AS roe,
                CASE WHEN totalAssets > 0 THEN netIncome / totalAssets ELSE NULL END AS roa,
                CASE WHEN totalAssets > 0 THEN totalLiabilities / totalAssets ELSE NULL END AS debt_ratio,
                CASE WHEN netIncome > 0 THEN operatingCashFlow / netIncome ELSE NULL END AS cash_conversion,
                CASE WHEN totalRevenue > 0 THEN freeCashFlow / totalRevenue ELSE NULL END AS fcf_margin,
                CASE WHEN totalAssets > 0 THEN totalRevenue / totalAssets ELSE NULL END AS asset_turnover,
                CASE WHEN totalRevenue > 0 THEN grossProfit / totalRevenue ELSE NULL END AS gross_margin,
                CASE WHEN totalRevenue > 0 THEN ebitda / totalRevenue ELSE NULL END AS ebitda_margin
            FROM financials.annual
            WHERE totalRevenue IS NOT NULL
        """)
        count = conn.execute("SELECT COUNT(*) FROM ratios.financial").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating ratios table: {e}")
        return 0


def create_growth_view(conn: duckdb.DuckDBPyConnection) -> int:
    """Create year-over-year growth view."""
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS growth")
        conn.execute("DROP VIEW IF EXISTS growth.annual")
        conn.execute("""
            CREATE VIEW growth.annual AS
            SELECT
                current.ticker,
                current.date,
                current.totalRevenue,
                current.netIncome,
                prior.totalRevenue AS prior_revenue,
                prior.netIncome AS prior_income,
                CASE
                    WHEN prior.totalRevenue > 0
                    THEN (current.totalRevenue - prior.totalRevenue) / prior.totalRevenue
                    ELSE NULL
                END AS revenue_growth_yoy,
                CASE
                    WHEN prior.netIncome > 0
                    THEN (current.netIncome - prior.netIncome) / prior.netIncome
                    ELSE NULL
                END AS income_growth_yoy
            FROM financials.annual AS current
            LEFT JOIN financials.annual AS prior
                ON current.ticker = prior.ticker
                AND YEAR(current.date) = YEAR(prior.date) + 1
        """)
        count = conn.execute("SELECT COUNT(*) FROM growth.annual").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating growth view: {e}")
        return 0


def create_peer_groups_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create peer groups reference table for comparative analysis.

    Returns:
        Number of peer group mappings created
    """
    try:
        from src.core.peer_groups import get_all_peer_data

        conn.execute("CREATE SCHEMA IF NOT EXISTS company")
        conn.execute("DROP TABLE IF EXISTS company.peers")

        # Create table
        conn.execute("""
            CREATE TABLE company.peers (
                ticker VARCHAR,
                peer_group VARCHAR
            )
        """)

        # Get peer group data
        peer_data = get_all_peer_data()

        # Insert data
        if peer_data:
            conn.executemany(
                "INSERT INTO company.peers (ticker, peer_group) VALUES (?, ?)",
                peer_data
            )

        count = conn.execute("SELECT COUNT(*) FROM company.peers").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating peer groups table: {e}")
        return 0


def create_portfolio_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create user portfolio tracking table (initially empty).

    Returns:
        Number of portfolio entries (0 initially)
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS user")
        conn.execute("DROP TABLE IF EXISTS user.portfolios")

        # Create table
        conn.execute("""
            CREATE TABLE user.portfolios (
                portfolio_name VARCHAR,
                ticker VARCHAR,
                shares DOUBLE,
                purchase_date DATE,
                purchase_price DOUBLE,
                notes VARCHAR
            )
        """)

        count = conn.execute("SELECT COUNT(*) FROM user.portfolios").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating portfolio table: {e}")
        return 0


def main() -> None:
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise SystemExit("MONGO_URI is not set.")
    logger = configure_logger()

    # Fetch all data from MongoDB
    with MongoClient(mongo_uri) as client:
        database = load_database(client, mongo_uri)
        annual_docs = fetch_documents(database["raw_annual"])
        quarterly_docs = fetch_documents(database["raw_quarterly"])
        prices_docs = fetch_documents(database["stock_prices_daily"])
        dividends_docs = fetch_documents(database["dividends_history"])
        splits_docs = fetch_documents(database["splits_history"])
        metadata_docs = fetch_documents(database["company_metadata"])
        # Phase 8: Earnings data
        earnings_history_docs = fetch_documents(database["earnings_history"])
        earnings_calendar_docs = fetch_documents(database["earnings_calendar"])
        # Phase 9: Analyst data
        analyst_recommendations_docs = fetch_documents(database["analyst_recommendations"])
        price_targets_docs = fetch_documents(database["price_targets"])
        analyst_consensus_docs = fetch_documents(database["analyst_consensus"])
        growth_estimates_docs = fetch_documents(database["growth_estimates"])

    # Prepare DataFrames
    annual_frame = prepare_dataframe(annual_docs)
    quarterly_frame = prepare_dataframe(quarterly_docs)
    prices_frame = prepare_prices_dataframe(prices_docs)
    dividends_frame = prepare_dividends_dataframe(dividends_docs)
    splits_frame = prepare_splits_dataframe(splits_docs)
    metadata_frame = prepare_metadata_dataframe(metadata_docs)
    # Phase 8: Earnings data
    earnings_history_frame = prepare_earnings_history_dataframe(earnings_history_docs)
    earnings_calendar_frame = prepare_earnings_calendar_dataframe(earnings_calendar_docs)
    # Phase 9: Analyst data - prepare frames from MongoDB documents
    # These will be processed into raw tables, then views will be created by analyst.py functions
    analyst_recs_frame = prepare_analyst_recommendations_dataframe(analyst_recommendations_docs)
    price_targets_frame = prepare_price_targets_dataframe(price_targets_docs)
    analyst_consensus_frame = prepare_analyst_consensus_dataframe(analyst_consensus_docs)
    growth_estimates_frame = prepare_growth_estimates_dataframe(growth_estimates_docs)

    # Transform to DuckDB
    conn = duckdb.connect(DUCKDB_PATH)
    try:
        # Financial statements
        annual_rows = upsert_dataframe(conn, annual_frame, ANNUAL_TABLE, "financials")
        log_event(logger, phase="transform.annual", rows=annual_rows)

        quarterly_rows = upsert_dataframe(conn, quarterly_frame, QUARTERLY_TABLE, "financials")
        log_event(logger, phase="transform.quarterly", rows=quarterly_rows)

        # Prices
        conn.execute("CREATE SCHEMA IF NOT EXISTS prices")
        prices_rows = upsert_dataframe(conn, prices_frame, "prices.daily", "prices")
        log_event(logger, phase="transform.prices", rows=prices_rows)

        # Dividends
        conn.execute("CREATE SCHEMA IF NOT EXISTS dividends")
        div_rows = upsert_dataframe(conn, dividends_frame, "dividends.history", "dividends")
        log_event(logger, phase="transform.dividends", rows=div_rows)

        # Splits
        conn.execute("CREATE SCHEMA IF NOT EXISTS splits")
        split_rows = upsert_dataframe(conn, splits_frame, "splits.history", "splits")
        log_event(logger, phase="transform.splits", rows=split_rows)

        # Company metadata (simple replace, no date-based upsert)
        conn.execute("CREATE SCHEMA IF NOT EXISTS company")
        if not metadata_frame.empty:
            conn.register("metadata_frame", metadata_frame)
            conn.execute("DROP TABLE IF EXISTS company.metadata")
            conn.execute("CREATE TABLE company.metadata AS SELECT * FROM metadata_frame")
            conn.unregister("metadata_frame")
            meta_rows = len(metadata_frame)
        else:
            meta_rows = 0
        log_event(logger, phase="transform.metadata", rows=meta_rows)

        # Phase 8: Earnings history
        conn.execute("CREATE SCHEMA IF NOT EXISTS earnings")
        if not earnings_history_frame.empty:
            # Use date-based upsert for earnings history
            earnings_hist_rows = upsert_dataframe(
                conn,
                earnings_history_frame,
                "earnings.history_raw",
                "earnings",
                key_columns=["ticker", "report_date", "fiscal_period"],
            )
        else:
            conn.execute("CREATE TABLE IF NOT EXISTS earnings.history_raw (ticker VARCHAR, report_date DATE, fiscal_period VARCHAR, eps_estimate DOUBLE, eps_actual DOUBLE, eps_surprise DOUBLE, surprise_pct DOUBLE, revenue_estimate DOUBLE, revenue_actual DOUBLE)")
            earnings_hist_rows = 0
        log_event(logger, phase="transform.earnings_history", rows=earnings_hist_rows)

        # Phase 8: Earnings calendar
        if not earnings_calendar_frame.empty:
            # Simple replace for earnings calendar (upcoming dates)
            conn.register("earnings_calendar_frame", earnings_calendar_frame)
            conn.execute("DROP TABLE IF EXISTS earnings.calendar")
            conn.execute("CREATE TABLE earnings.calendar AS SELECT * FROM earnings_calendar_frame")
            conn.unregister("earnings_calendar_frame")
            earnings_cal_rows = len(earnings_calendar_frame)
        else:
            conn.execute("CREATE TABLE IF NOT EXISTS earnings.calendar (ticker VARCHAR, earnings_date DATE, period_ending VARCHAR, estimate DOUBLE)")
            earnings_cal_rows = 0
        log_event(logger, phase="transform.earnings_calendar", rows=earnings_cal_rows)

        # Phase 9: Analyst recommendations
        if not analyst_recs_frame.empty:
            analyst_recs_rows = upsert_dataframe(conn, analyst_recs_frame, "analyst.recommendations_raw", "analyst")
        else:
            conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")
            conn.execute("CREATE TABLE IF NOT EXISTS analyst.recommendations_raw (ticker VARCHAR, date DATE, firm VARCHAR, from_grade VARCHAR, to_grade VARCHAR, action VARCHAR)")
            analyst_recs_rows = 0
        log_event(logger, phase="transform.analyst_recommendations", rows=analyst_recs_rows)

        # Phase 9: Price targets
        if not price_targets_frame.empty:
            conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")
            conn.register("price_targets_frame", price_targets_frame)
            conn.execute("DROP TABLE IF EXISTS analyst.price_targets_raw")
            conn.execute("CREATE TABLE analyst.price_targets_raw AS SELECT * FROM price_targets_frame")
            conn.unregister("price_targets_frame")
            price_targets_rows = len(price_targets_frame)
        else:
            conn.execute("CREATE TABLE IF NOT EXISTS analyst.price_targets_raw (ticker VARCHAR, date DATE, current_price DOUBLE, target_low DOUBLE, target_mean DOUBLE, target_high DOUBLE, num_analysts INTEGER)")
            price_targets_rows = 0
        log_event(logger, phase="transform.price_targets", rows=price_targets_rows)

        # Phase 9: Analyst consensus
        if not analyst_consensus_frame.empty:
            conn.register("analyst_consensus_frame", analyst_consensus_frame)
            conn.execute("DROP TABLE IF EXISTS analyst.consensus_raw")
            conn.execute("CREATE TABLE analyst.consensus_raw AS SELECT * FROM analyst_consensus_frame")
            conn.unregister("analyst_consensus_frame")
            consensus_rows = len(analyst_consensus_frame)
        else:
            conn.execute("CREATE TABLE IF NOT EXISTS analyst.consensus_raw (ticker VARCHAR, date DATE, strong_buy INTEGER, buy INTEGER, hold INTEGER, sell INTEGER, strong_sell INTEGER)")
            consensus_rows = 0
        log_event(logger, phase="transform.analyst_consensus", rows=consensus_rows)

        # Phase 9: Growth estimates
        if not growth_estimates_frame.empty:
            conn.register("growth_estimates_frame", growth_estimates_frame)
            conn.execute("DROP TABLE IF EXISTS analyst.growth_estimates_raw")
            conn.execute("CREATE TABLE analyst.growth_estimates_raw AS SELECT * FROM growth_estimates_frame")
            conn.unregister("growth_estimates_frame")
            growth_estimates_rows = len(growth_estimates_frame)
        else:
            conn.execute("CREATE TABLE IF NOT EXISTS analyst.growth_estimates_raw (ticker VARCHAR, date DATE, current_qtr_growth DOUBLE, next_qtr_growth DOUBLE, current_year_growth DOUBLE, next_year_growth DOUBLE, next_5yr_growth DOUBLE)")
            growth_estimates_rows = 0
        log_event(logger, phase="transform.growth_estimates", rows=growth_estimates_rows)

        # Create derived ratios table
        ratio_rows = create_ratios_table(conn)
        log_event(logger, phase="transform.ratios", rows=ratio_rows)

        # Create growth view
        growth_rows = create_growth_view(conn)
        log_event(logger, phase="transform.growth_view", rows=growth_rows)

        # Create peer groups table (Phase 5)
        peer_rows = create_peer_groups_table(conn)
        log_event(logger, phase="transform.peer_groups", rows=peer_rows)

        # Create portfolio table (Phase 5)
        portfolio_rows = create_portfolio_table(conn)
        log_event(logger, phase="transform.portfolios", rows=portfolio_rows)

        # Create valuation metrics table (Phase 8)
        from src.analysis.valuation import create_valuation_metrics_table, create_earnings_history_table, create_earnings_calendar_view

        valuation_rows = create_valuation_metrics_table(conn)
        log_event(logger, phase="transform.valuation_metrics", rows=valuation_rows)

        # Create earnings history view (Phase 8)
        earnings_history_rows = create_earnings_history_table(conn)
        log_event(logger, phase="transform.earnings_history_view", rows=earnings_history_rows)

        # Create earnings calendar view (Phase 8)
        earnings_calendar_view_rows = create_earnings_calendar_view(conn)
        log_event(logger, phase="transform.earnings_calendar_view", rows=earnings_calendar_view_rows)

        # Create analyst views (Phase 9)
        from src.analysis.analyst import (
            create_analyst_recommendations_table,
            create_price_targets_table,
            create_analyst_consensus_table,
            create_growth_estimates_table
        )

        analyst_recs_view_rows = create_analyst_recommendations_table(conn)
        log_event(logger, phase="transform.analyst_recommendations_view", rows=analyst_recs_view_rows)

        price_targets_view_rows = create_price_targets_table(conn)
        log_event(logger, phase="transform.price_targets_view", rows=price_targets_view_rows)

        analyst_consensus_view_rows = create_analyst_consensus_table(conn)
        log_event(logger, phase="transform.analyst_consensus_view", rows=analyst_consensus_view_rows)

        growth_estimates_view_rows = create_growth_estimates_table(conn)
        log_event(logger, phase="transform.growth_estimates_view", rows=growth_estimates_view_rows)

        # Create technical indicators table (Phase 10)
        from src.analysis.technical import create_technical_indicators_table

        technical_rows = create_technical_indicators_table(conn)
        log_event(logger, phase="transform.technical_indicators", rows=technical_rows)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
