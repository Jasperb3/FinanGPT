"""Expose ingestion helpers for backwards compatibility."""

from ingestion.ingest import (  # noqa: F401
    get_last_fetch_info,
    is_data_stale,
    should_skip_ticker,
    get_last_price_date,
)

__all__ = [
    "get_last_fetch_info",
    "is_data_stale",
    "should_skip_ticker",
    "get_last_price_date",
]
