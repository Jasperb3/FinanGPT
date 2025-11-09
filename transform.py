#!/usr/bin/env python3
"""Transform raw MongoDB snapshots into DuckDB analytical tables."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
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
    file_handler = logging.FileHandler(LOGS_DIR / f"transform_{datetime.utcnow():%Y%m%d}.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_event(logger: logging.Logger, **payload: Any) -> None:
    entry = {"ts": datetime.utcnow().isoformat(), **payload}
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


def upsert_dataframe(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str) -> int:
    if frame.empty:
        return 0
    view_name = f"staging_{table.replace('.', '_')}"
    conn.execute("CREATE SCHEMA IF NOT EXISTS financials")
    conn.register(view_name, frame)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {view_name} LIMIT 0")
    conn.execute(
        f"""
        DELETE FROM {table}
        USING {view_name}
        WHERE {table}.ticker = {view_name}.ticker
          AND {table}.date = {view_name}.date
        """
    )
    conn.execute(f"INSERT INTO {table} SELECT * FROM {view_name}")
    conn.unregister(view_name)
    return len(frame)


def main() -> None:
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise SystemExit("MONGO_URI is not set.")
    logger = configure_logger()
    with MongoClient(mongo_uri) as client:
        database = load_database(client, mongo_uri)
        annual_docs = fetch_documents(database["raw_annual"])
        quarterly_docs = fetch_documents(database["raw_quarterly"])
    annual_frame = prepare_dataframe(annual_docs)
    quarterly_frame = prepare_dataframe(quarterly_docs)
    conn = duckdb.connect(DUCKDB_PATH)
    try:
        annual_rows = upsert_dataframe(conn, annual_frame, ANNUAL_TABLE)
        log_event(logger, phase="transform.annual", rows=annual_rows)
        quarterly_rows = upsert_dataframe(conn, quarterly_frame, QUARTERLY_TABLE)
        log_event(logger, phase="transform.quarterly", rows=quarterly_rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
