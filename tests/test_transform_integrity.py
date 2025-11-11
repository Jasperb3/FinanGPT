from __future__ import annotations

from unittest.mock import MagicMock

import duckdb
import pytest

from src.transformation.transform import run_integrity_check


def _make_conn(count: int, table: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    schema, name = table.split(".") if "." in table else ("main", table)
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    conn.execute(f"CREATE TABLE {table} (id INTEGER)")
    if count:
        conn.executemany(f"INSERT INTO {table} VALUES (?)", [(i,) for i in range(count)])
    return conn


def test_run_integrity_check_failure(tmp_path):
    conn = _make_conn(2, "financials.annual")
    logger = MagicMock()
    with pytest.raises(SystemExit):
        run_integrity_check(conn, "financials.annual", source_count=10, tolerance_pct=5.0, dataset_label="annual", logger=logger)


def test_run_integrity_check_passes_within_tolerance():
    conn = _make_conn(99, "financials.quarterly")
    logger = MagicMock()
    # Source 100 vs dest 99 => 1% diff should pass for tolerance 5%
    run_integrity_check(conn, "financials.quarterly", source_count=100, tolerance_pct=5.0, dataset_label="quarterly", logger=logger)
