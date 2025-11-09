#!/usr/bin/env python3
"""Tests for data freshness tracking and incremental updates (Phase 2)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from pymongo.collection import Collection
from pymongo.database import Database

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import (
    get_last_fetch_info,
    is_data_stale,
    should_skip_ticker,
    get_last_price_date,
)
from query import (
    extract_tickers_from_sql,
    check_data_freshness,
)


class TestFreshnessTracking:
    """Test freshness tracking functionality."""

    def test_get_last_fetch_info_found(self):
        """Test retrieving existing metadata."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "data_type": "prices_daily",
            "last_fetched": "2025-11-09T10:00:00Z",
            "status": "success",
            "record_count": 365,
        }

        result = get_last_fetch_info(mock_collection, "AAPL", "prices_daily")

        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["data_type"] == "prices_daily"
        mock_collection.find_one.assert_called_once_with({
            "ticker": "AAPL",
            "data_type": "prices_daily"
        })

    def test_get_last_fetch_info_not_found(self):
        """Test retrieving non-existent metadata."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.find_one.return_value = None

        result = get_last_fetch_info(mock_collection, "NVDA", "prices_daily")

        assert result is None

    def test_is_data_stale_no_metadata(self):
        """Test that missing metadata is considered stale."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.find_one.return_value = None

        result = is_data_stale(mock_collection, "AAPL", "prices_daily", threshold_days=7)

        assert result is True

    def test_is_data_stale_fresh_data(self):
        """Test that recent data is not considered stale."""
        mock_collection = MagicMock(spec=Collection)
        # Data fetched 2 days ago
        recent_time = datetime.utcnow() - timedelta(days=2)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "data_type": "prices_daily",
            "last_fetched": recent_time.isoformat() + "Z",
        }

        result = is_data_stale(mock_collection, "AAPL", "prices_daily", threshold_days=7)

        assert result is False

    def test_is_data_stale_old_data(self):
        """Test that old data is considered stale."""
        mock_collection = MagicMock(spec=Collection)
        # Data fetched 10 days ago
        old_time = datetime.utcnow() - timedelta(days=10)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "data_type": "prices_daily",
            "last_fetched": old_time.isoformat() + "Z",
        }

        result = is_data_stale(mock_collection, "AAPL", "prices_daily", threshold_days=7)

        assert result is True

    def test_should_skip_ticker_force_mode(self):
        """Test that force mode never skips tickers."""
        mock_collection = MagicMock(spec=Collection)

        result = should_skip_ticker(
            mock_collection,
            "AAPL",
            refresh_mode=True,
            force_mode=True,
            refresh_days=7,
        )

        assert result is False

    def test_should_skip_ticker_normal_mode(self):
        """Test that normal mode never skips tickers."""
        mock_collection = MagicMock(spec=Collection)

        result = should_skip_ticker(
            mock_collection,
            "AAPL",
            refresh_mode=False,
            force_mode=False,
            refresh_days=7,
        )

        assert result is False

    def test_should_skip_ticker_refresh_mode_fresh_data(self):
        """Test that refresh mode skips tickers with fresh data."""
        mock_collection = MagicMock(spec=Collection)

        # Mock all data types as fresh
        def mock_find_one(query):
            recent_time = datetime.utcnow() - timedelta(days=2)
            return {
                "ticker": query["ticker"],
                "data_type": query["data_type"],
                "last_fetched": recent_time.isoformat() + "Z",
            }

        mock_collection.find_one.side_effect = mock_find_one

        result = should_skip_ticker(
            mock_collection,
            "AAPL",
            refresh_mode=True,
            force_mode=False,
            refresh_days=7,
        )

        assert result is True

    def test_should_skip_ticker_refresh_mode_stale_data(self):
        """Test that refresh mode doesn't skip tickers with stale data."""
        mock_collection = MagicMock(spec=Collection)

        # Mock some data types as stale
        def mock_find_one(query):
            if query["data_type"] == "prices_daily":
                old_time = datetime.utcnow() - timedelta(days=10)
                return {
                    "ticker": query["ticker"],
                    "data_type": query["data_type"],
                    "last_fetched": old_time.isoformat() + "Z",
                }
            return None

        mock_collection.find_one.side_effect = mock_find_one

        result = should_skip_ticker(
            mock_collection,
            "AAPL",
            refresh_mode=True,
            force_mode=False,
            refresh_days=7,
        )

        assert result is False

    def test_get_last_price_date_found(self):
        """Test retrieving the last price date."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "date": "2025-11-08T21:00:00Z",
            "close": 150.25,
        }

        result = get_last_price_date(mock_collection, "AAPL")

        assert result is not None
        assert isinstance(result, datetime)
        mock_collection.find_one.assert_called_once()

    def test_get_last_price_date_not_found(self):
        """Test retrieving last price date when none exists."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.find_one.return_value = None

        result = get_last_price_date(mock_collection, "NVDA")

        assert result is None


class TestQueryFreshnessChecking:
    """Test freshness checking in query.py."""

    def test_extract_tickers_from_sql_single_ticker(self):
        """Test extracting a single ticker from SQL."""
        sql = "SELECT date, close FROM prices.daily WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 10"
        tickers = extract_tickers_from_sql(sql)
        assert tickers == ["AAPL"]

    def test_extract_tickers_from_sql_multiple_tickers(self):
        """Test extracting multiple tickers from IN clause."""
        sql = "SELECT ticker, date, totalRevenue FROM financials.annual WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')"
        tickers = extract_tickers_from_sql(sql)
        assert set(tickers) == {"AAPL", "MSFT", "GOOGL"}

    def test_extract_tickers_from_sql_no_tickers(self):
        """Test SQL without ticker filters."""
        sql = "SELECT * FROM company.metadata LIMIT 10"
        tickers = extract_tickers_from_sql(sql)
        assert tickers == []

    def test_check_data_freshness_no_db(self):
        """Test freshness check with no database connection."""
        result = check_data_freshness(None, ["AAPL"])
        assert result["is_stale"] is False
        assert result["stale_tickers"] == []

    def test_check_data_freshness_no_tickers(self):
        """Test freshness check with no tickers."""
        mock_db = MagicMock(spec=Database)
        result = check_data_freshness(mock_db, [])
        assert result["is_stale"] is False
        assert result["stale_tickers"] == []

    def test_check_data_freshness_fresh_data(self):
        """Test freshness check with fresh data."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_db.__getitem__.return_value = mock_collection

        recent_time = datetime.utcnow() - timedelta(days=2)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "data_type": "prices_daily",
            "last_fetched": recent_time.isoformat() + "Z",
        }

        result = check_data_freshness(mock_db, ["AAPL"], threshold_days=7)

        assert result["is_stale"] is False
        assert result["stale_tickers"] == []
        assert "AAPL" in result["freshness_info"]

    def test_check_data_freshness_stale_data(self):
        """Test freshness check with stale data."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_db.__getitem__.return_value = mock_collection

        old_time = datetime.utcnow() - timedelta(days=10)
        mock_collection.find_one.return_value = {
            "ticker": "AAPL",
            "data_type": "prices_daily",
            "last_fetched": old_time.isoformat() + "Z",
        }

        result = check_data_freshness(mock_db, ["AAPL"], threshold_days=7)

        assert result["is_stale"] is True
        assert "AAPL" in result["stale_tickers"]
        assert "AAPL" in result["freshness_info"]

    def test_check_data_freshness_never_fetched(self):
        """Test freshness check with never-fetched ticker."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_db.__getitem__.return_value = mock_collection

        mock_collection.find_one.return_value = None

        result = check_data_freshness(mock_db, ["NVDA"], threshold_days=7)

        assert result["is_stale"] is True
        assert "NVDA" in result["stale_tickers"]
        assert result["freshness_info"]["NVDA"] == "never fetched"


class TestIncrementalPriceUpdates:
    """Test incremental price update logic."""

    def test_incremental_update_skips_current_data(self):
        """Test that incremental updates skip when already up-to-date."""
        # This would be tested with actual yfinance integration
        # For now, we test the logic that determines if a fetch should occur
        last_date = datetime.utcnow()
        start_date = (last_date + timedelta(days=1)).date()
        end_date = datetime.utcnow().date()

        # If start_date > end_date, we should skip the fetch
        should_skip = start_date > end_date
        assert should_skip is True

    def test_incremental_update_fetches_new_data(self):
        """Test that incremental updates fetch when data is missing."""
        last_date = datetime.utcnow() - timedelta(days=5)
        start_date = (last_date + timedelta(days=1)).date()
        end_date = datetime.utcnow().date()

        # If start_date <= end_date, we should fetch
        should_fetch = start_date <= end_date
        assert should_fetch is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
