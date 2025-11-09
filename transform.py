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
    iso_str = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
    parsed = datetime.fromisoformat(iso_str)
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


def upsert_dataframe(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str, schema: str | None = None) -> int:
    if frame.empty:
        return 0
    view_name = f"staging_{table.replace('.', '_')}"
    if schema:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    conn.register(view_name, frame)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {view_name} LIMIT 0")
    ensure_columns(conn, table, frame)
    conn.execute(
        f"""
        DELETE FROM {table}
        USING {view_name}
        WHERE {table}.ticker = {view_name}.ticker
          AND {table}.date = {view_name}.date
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
        from peer_groups import get_all_peer_data

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

    # Prepare DataFrames
    annual_frame = prepare_dataframe(annual_docs)
    quarterly_frame = prepare_dataframe(quarterly_docs)
    prices_frame = prepare_prices_dataframe(prices_docs)
    dividends_frame = prepare_dividends_dataframe(dividends_docs)
    splits_frame = prepare_splits_dataframe(splits_docs)
    metadata_frame = prepare_metadata_dataframe(metadata_docs)

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

    finally:
        conn.close()


if __name__ == "__main__":
    main()
