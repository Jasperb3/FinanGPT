"""Expose query guardrails for compatibility with legacy imports."""

from query_engine.query import (  # noqa: F401
    extract_tickers_from_sql,
    check_data_freshness,
    build_system_prompt,
    ALLOWED_TABLES,
)

__all__ = [
    "extract_tickers_from_sql",
    "check_data_freshness",
    "build_system_prompt",
    "ALLOWED_TABLES",
]
