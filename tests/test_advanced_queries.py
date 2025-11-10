#!/usr/bin/env python3
"""Tests for Phase 5: Advanced Query Capabilities."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.peer_groups import (
    PEER_GROUPS,
    get_peer_group,
    list_peer_groups,
    get_all_peer_data,
    is_valid_peer_group,
)
from query import build_system_prompt, ALLOWED_TABLES


class TestPeerGroups:
    """Test peer group functionality."""

    def test_peer_groups_defined(self):
        """Test that peer groups are properly defined."""
        assert len(PEER_GROUPS) > 0
        assert "FAANG" in PEER_GROUPS
        assert "Semiconductors" in PEER_GROUPS
        assert "Magnificent Seven" in PEER_GROUPS

    def test_faang_group(self):
        """Test FAANG peer group contents."""
        faang = get_peer_group("FAANG")
        assert "AAPL" in faang
        assert "META" in faang
        assert "AMZN" in faang
        assert "NFLX" in faang
        assert "GOOGL" in faang
        assert len(faang) == 5

    def test_case_insensitive_lookup(self):
        """Test case-insensitive peer group lookup."""
        assert get_peer_group("faang") == get_peer_group("FAANG")
        assert get_peer_group("Faang") == get_peer_group("FAANG")

    def test_invalid_peer_group(self):
        """Test error handling for invalid peer group."""
        with pytest.raises(KeyError, match="not found"):
            get_peer_group("INVALID_GROUP")

    def test_list_peer_groups(self):
        """Test listing all peer groups."""
        groups = list_peer_groups()
        assert len(groups) > 0
        assert "FAANG" in groups
        assert "Semiconductors" in groups

    def test_get_all_peer_data(self):
        """Test getting all peer data for database insertion."""
        all_data = get_all_peer_data()
        assert len(all_data) > 0

        # Check format
        for ticker, group_name in all_data:
            assert isinstance(ticker, str)
            assert isinstance(group_name, str)
            assert len(ticker) > 0

        # Check that AAPL appears in multiple groups
        aapl_groups = [group for ticker, group in all_data if ticker == "AAPL"]
        assert len(aapl_groups) >= 2  # AAPL is in FAANG and Magnificent Seven

    def test_is_valid_peer_group(self):
        """Test peer group name validation."""
        assert is_valid_peer_group("FAANG")
        assert is_valid_peer_group("faang")
        assert is_valid_peer_group("Semiconductors")
        assert not is_valid_peer_group("INVALID")
        assert not is_valid_peer_group("")


class TestSystemPromptEnhancements:
    """Test enhancements to system prompt for Phase 5."""

    def test_system_prompt_includes_date_context(self):
        """Test that system prompt includes date parsing context."""
        schema = {
            "financials.annual": ["ticker", "date", "totalRevenue"],
        }
        prompt = build_system_prompt(schema)

        # Check for date context
        assert "Date Context" in prompt
        assert str(date.today().year) in prompt
        assert "last year" in prompt or "past year" in prompt
        assert "YTD" in prompt or "year to date" in prompt

    def test_system_prompt_includes_peer_groups(self):
        """Test that system prompt includes peer groups information."""
        schema = {
            "company.peers": ["ticker", "peer_group"],
        }
        prompt = build_system_prompt(schema)

        # Check for peer groups info
        assert "Peer Groups" in prompt
        assert "FAANG" in prompt
        assert "Semiconductors" in prompt

    def test_system_prompt_includes_window_functions(self):
        """Test that system prompt includes window functions documentation."""
        schema = {
            "financials.annual": ["ticker", "date", "totalRevenue"],
        }
        prompt = build_system_prompt(schema)

        # Check for window functions
        assert "Window functions" in prompt or "RANK()" in prompt
        assert "ROW_NUMBER" in prompt
        assert "PARTITION BY" in prompt

    def test_system_prompt_includes_new_tables(self):
        """Test that system prompt references new tables."""
        schema = {
            "company.peers": ["ticker", "peer_group"],
            "user.portfolios": ["portfolio_name", "ticker", "shares"],
        }
        prompt = build_system_prompt(schema)

        # Check that new tables are in the schema
        assert "company.peers" in prompt
        assert "user.portfolios" in prompt


class TestAllowedTables:
    """Test that new tables are in the allow-list."""

    def test_peer_groups_table_allowed(self):
        """Test that company.peers is in allowed tables."""
        assert "company.peers" in ALLOWED_TABLES

    def test_portfolios_table_allowed(self):
        """Test that user.portfolios is in allowed tables."""
        assert "user.portfolios" in ALLOWED_TABLES

    def test_all_tables_still_allowed(self):
        """Test that all original tables are still allowed."""
        assert "financials.annual" in ALLOWED_TABLES
        assert "financials.quarterly" in ALLOWED_TABLES
        assert "prices.daily" in ALLOWED_TABLES
        assert "dividends.history" in ALLOWED_TABLES
        assert "splits.history" in ALLOWED_TABLES
        assert "company.metadata" in ALLOWED_TABLES
        assert "ratios.financial" in ALLOWED_TABLES
        assert "growth.annual" in ALLOWED_TABLES


class TestDateParsing:
    """Test date parsing context in system prompt."""

    def test_date_context_accuracy(self):
        """Test that date calculations are accurate."""
        today = date.today()
        one_year_ago = today - timedelta(days=365)
        five_years_ago = today - timedelta(days=365*5)

        schema = {"financials.annual": ["ticker", "date"]}
        prompt = build_system_prompt(schema)

        # Check that dates are approximately correct (within a few days tolerance)
        assert str(today.year) in prompt
        # The prompt should mention years that are close to the calculated values
        assert str(one_year_ago.year) in prompt or str(one_year_ago.year + 1) in prompt


class TestWindowFunctionSupport:
    """Test that window functions are mentioned and encouraged."""

    def test_ranking_functions_documented(self):
        """Test that ranking functions are documented."""
        schema = {"financials.annual": ["ticker", "date", "totalRevenue"]}
        prompt = build_system_prompt(schema)

        # Check for ranking functions
        assert "RANK" in prompt
        assert "ROW_NUMBER" in prompt
        assert "DENSE_RANK" in prompt

    def test_analytical_functions_documented(self):
        """Test that analytical functions are documented."""
        schema = {"financials.annual": ["ticker", "date", "totalRevenue"]}
        prompt = build_system_prompt(schema)

        # Check for analytical functions
        assert "LAG" in prompt or "LEAD" in prompt
        assert "PARTITION BY" in prompt

    def test_statistical_functions_documented(self):
        """Test that statistical functions are documented."""
        schema = {"financials.annual": ["ticker", "date", "totalRevenue"]}
        prompt = build_system_prompt(schema)

        # Check for statistical functions
        assert "AVG" in prompt or "STDDEV" in prompt or "MEDIAN" in prompt


class TestPeerGroupQueries:
    """Test example peer group query patterns."""

    def test_faang_comparison_pattern(self):
        """Test that FAANG comparison is well-documented."""
        schema = {"company.peers": ["ticker", "peer_group"]}
        prompt = build_system_prompt(schema)

        # Should mention FAANG in examples
        assert "FAANG" in prompt

    def test_peer_group_join_pattern(self):
        """Test that JOIN pattern is suggested."""
        schema = {"company.peers": ["ticker", "peer_group"]}
        prompt = build_system_prompt(schema)

        # Should suggest using JOIN for peer groups
        assert "JOIN" in prompt or "join" in prompt


class TestPortfolioFeatures:
    """Test portfolio tracking capabilities."""

    def test_portfolio_table_in_schema(self):
        """Test that portfolio table is properly defined."""
        assert "user.portfolios" in ALLOWED_TABLES

    def test_portfolio_schema_documented(self):
        """Test that portfolio schema is visible in prompt."""
        schema = {
            "user.portfolios": [
                "portfolio_name",
                "ticker",
                "shares",
                "purchase_date",
                "purchase_price",
                "notes"
            ]
        }
        prompt = build_system_prompt(schema)

        assert "user.portfolios" in prompt
        assert "shares" in prompt


class TestIntegration:
    """Integration tests for Phase 5 features."""

    def test_system_prompt_completeness(self):
        """Test that system prompt includes all Phase 5 features."""
        schema = {
            "financials.annual": ["ticker", "date", "totalRevenue"],
            "company.peers": ["ticker", "peer_group"],
            "user.portfolios": ["portfolio_name", "ticker", "shares"],
        }
        prompt = build_system_prompt(schema)

        # Should include all major Phase 5 components
        assert "Date Context" in prompt
        assert "Peer Groups" in prompt
        assert "Window functions" in prompt or "RANK" in prompt
        assert len(prompt) > 500  # Prompt should be substantial

    def test_all_peer_groups_have_tickers(self):
        """Test that all peer groups have at least one ticker."""
        for group_name, tickers in PEER_GROUPS.items():
            assert len(tickers) > 0, f"Peer group '{group_name}' has no tickers"
            for ticker in tickers:
                assert isinstance(ticker, str), f"Invalid ticker type in '{group_name}'"
                assert len(ticker) > 0, f"Empty ticker in '{group_name}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
