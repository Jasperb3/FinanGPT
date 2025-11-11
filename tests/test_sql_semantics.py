from __future__ import annotations

import pytest

from src.query_engine import query
from src.query_engine.query import (
    SemanticValidationError,
    validate_sql,
    validate_sql_semantics,
)

SCHEMA = {
    "financials.annual": ["ticker", "totalRevenue", "netIncome", "date"],
    "prices.daily": ["ticker", "date", "close"],
}


def enable_semantic_validation(monkeypatch, enabled: bool = True) -> None:
    monkeypatch.setattr(query, "_SEMANTIC_VALIDATION_TOGGLE", enabled)


def test_missing_group_by_detected(monkeypatch):
    enable_semantic_validation(monkeypatch, True)
    sql = "SELECT ticker, SUM(totalRevenue) FROM financials.annual"
    with pytest.raises(SemanticValidationError):
        validate_sql(sql, SCHEMA, question="Show total revenue per ticker")


def test_join_requires_qualified_columns():
    sql = (
        "SELECT close, ticker FROM prices.daily JOIN financials.annual ON prices.daily.ticker = financials.annual.ticker"
    )
    ok, reason = validate_sql_semantics(sql, "Compare prices vs revenue")
    assert not ok
    assert "qualified" in reason.lower()


def test_disallows_unsafe_cast():
    sql = "SELECT CAST(ticker AS blob) FROM financials.annual"
    ok, reason = validate_sql_semantics(sql, "Show blob cast")
    assert not ok
    assert "blob" in reason.lower()


def test_top_query_requires_order(monkeypatch):
    enable_semantic_validation(monkeypatch, True)
    sql = "SELECT ticker, totalRevenue FROM financials.annual"
    with pytest.raises(SemanticValidationError):
        validate_sql(sql, SCHEMA, question="Top companies by revenue")


def test_count_query_requires_count(monkeypatch):
    enable_semantic_validation(monkeypatch, True)
    sql = "SELECT ticker FROM financials.annual"
    with pytest.raises(SemanticValidationError):
        validate_sql(sql, SCHEMA, question="How many tickers are there")
