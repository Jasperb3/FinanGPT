#!/usr/bin/env python3
"""Tests for Phase 6: Error Resilience & UX Polish."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.resilience import (
    execute_template,
    get_all_tickers,
    list_templates,
    load_query_templates,
    suggest_tickers,
    validate_ticker,
)


class TestQueryTemplates:
    """Test query template functionality."""

    def test_load_templates(self):
        """Test loading templates from YAML file."""
        templates = load_query_templates()
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert "top_revenue" in templates
        assert "ticker_comparison" in templates

    def test_template_structure(self):
        """Test that templates have required fields."""
        templates = load_query_templates()

        for name, tpl in templates.items():
            assert "description" in tpl, f"Template '{name}' missing description"
            assert "sql" in tpl, f"Template '{name}' missing SQL"
            assert "params" in tpl, f"Template '{name}' missing params"

    def test_list_templates(self):
        """Test listing all templates."""
        templates_list = list_templates()
        assert isinstance(templates_list, list)
        assert len(templates_list) > 0

        for tpl in templates_list:
            assert "name" in tpl
            assert "description" in tpl

    def test_execute_template_top_revenue(self):
        """Test executing top_revenue template."""
        # Create in-memory DuckDB with test data
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA financials;
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                totalRevenue DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO financials.annual VALUES
                ('AAPL', '2023-09-30', 394000000000),
                ('MSFT', '2023-06-30', 245000000000),
                ('GOOGL', '2023-12-31', 307000000000)
        """)

        # Execute template
        params = {"year": 2023, "limit": 3}
        columns, rows, sql = execute_template("top_revenue", params, conn)

        assert len(columns) == 3
        assert "ticker" in columns
        assert "totalRevenue" in columns
        assert len(rows) == 3
        assert rows[0][0] == "AAPL"  # Top revenue

        conn.close()

    def test_execute_template_missing_params(self):
        """Test that missing required parameters raise error."""
        conn = duckdb.connect(":memory:")

        with pytest.raises(KeyError, match="Missing required parameters"):
            execute_template("top_revenue", {}, conn)

        conn.close()

    def test_execute_template_with_defaults(self):
        """Test that default parameters are applied."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA financials;
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                totalRevenue DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO financials.annual VALUES
                ('AAPL', '2023-09-30', 394000000000)
        """)

        # Execute with only required params (limit should use default)
        params = {"year": 2023}
        columns, rows, sql = execute_template("top_revenue", params, conn)

        # Default limit is 10
        assert "LIMIT 10" in sql

        conn.close()

    def test_execute_template_invalid_name(self):
        """Test that invalid template name raises error."""
        conn = duckdb.connect(":memory:")

        with pytest.raises(KeyError, match="Template .* not found"):
            execute_template("invalid_template", {}, conn)

        conn.close()


class TestTickerValidation:
    """Test ticker validation and autocomplete."""

    def test_validate_ticker_exists(self):
        """Test validating an existing ticker."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA company;
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                longName VARCHAR
            )
        """)
        conn.execute("INSERT INTO company.metadata VALUES ('AAPL', 'Apple Inc.')")

        assert validate_ticker("AAPL", conn) is True
        assert validate_ticker("aapl", conn) is True  # Case insensitive

        conn.close()

    def test_validate_ticker_not_exists(self):
        """Test validating a non-existent ticker."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA company;
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                longName VARCHAR
            )
        """)

        assert validate_ticker("INVALID", conn) is False

        conn.close()

    def test_suggest_tickers(self):
        """Test ticker autocomplete suggestions."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA company;
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                longName VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO company.metadata VALUES
                ('AAPL', 'Apple Inc.'),
                ('AMD', 'Advanced Micro Devices'),
                ('AMZN', 'Amazon.com'),
                ('MSFT', 'Microsoft')
        """)

        # Suggest tickers starting with 'A'
        suggestions = suggest_tickers("A", conn)
        assert len(suggestions) == 3
        assert "AAPL" in suggestions
        assert "AMD" in suggestions
        assert "AMZN" in suggestions
        assert "MSFT" not in suggestions

        # Suggest tickers starting with 'AM'
        suggestions = suggest_tickers("AM", conn, limit=2)
        assert len(suggestions) == 2
        assert all(s.startswith("AM") for s in suggestions)

        conn.close()

    def test_get_all_tickers(self):
        """Test getting all available tickers."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA company;
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                longName VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO company.metadata VALUES
                ('AAPL', 'Apple Inc.'),
                ('MSFT', 'Microsoft'),
                ('GOOGL', 'Google')
        """)

        tickers = get_all_tickers(conn)
        assert len(tickers) == 3
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

        conn.close()


class TestTemplateIntegration:
    """Integration tests for template system."""

    def test_ticker_comparison_template(self):
        """Test ticker_comparison template."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA financials;
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                totalRevenue DOUBLE,
                netIncome DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO financials.annual VALUES
                ('AAPL', '2023-09-30', 394000000000, 97000000000),
                ('MSFT', '2023-06-30', 245000000000, 88000000000)
        """)

        params = {
            "metric": "totalRevenue",
            "tickers": "'AAPL', 'MSFT'",
            "limit": 10
        }

        columns, rows, sql = execute_template("ticker_comparison", params, conn)

        assert len(rows) == 2
        assert "totalRevenue" in columns

        conn.close()

    def test_peer_group_comparison_template(self):
        """Test peer_group_comparison template."""
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE SCHEMA financials;
            CREATE SCHEMA company;

            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                totalRevenue DOUBLE
            );

            CREATE TABLE company.peers (
                ticker VARCHAR,
                peer_group VARCHAR
            );
        """)
        conn.execute("""
            INSERT INTO financials.annual VALUES
                ('AAPL', '2023-09-30', 394000000000),
                ('MSFT', '2023-06-30', 245000000000);

            INSERT INTO company.peers VALUES
                ('AAPL', 'Tech Giants'),
                ('MSFT', 'Tech Giants');
        """)

        params = {
            "metric": "totalRevenue",
            "peer_group": "Tech Giants",
            "year": 2023,
            "limit": 10
        }

        columns, rows, sql = execute_template("peer_group_comparison", params, conn)

        assert len(rows) == 2
        assert "ticker" in columns
        assert "totalRevenue" in columns

        conn.close()


class TestGracefulDegradation:
    """Test graceful degradation when services are unavailable."""

    @patch("resilience.input")
    def test_handle_ollama_failure_exit(self, mock_input):
        """Test handling Ollama failure with exit choice."""
        from resilience import handle_ollama_failure

        mock_input.return_value = "3"  # Choose to exit

        error = ConnectionError("Ollama not reachable")
        result = handle_ollama_failure(error)

        assert result is None
        assert mock_input.called

    @patch("resilience.input")
    def test_handle_ollama_failure_direct_sql(self, mock_input):
        """Test handling Ollama failure with direct SQL input."""
        from resilience import handle_ollama_failure

        mock_input.side_effect = ["1", "SELECT * FROM financials.annual LIMIT 10"]

        error = ConnectionError("Ollama not reachable")
        result = handle_ollama_failure(error)

        assert result == "SELECT * FROM financials.annual LIMIT 10"


class TestDebugLogging:
    """Test debug logging functionality."""

    def test_debug_log_enabled(self, capsys):
        """Test that debug log prints when enabled."""
        from resilience import debug_log

        debug_log("Test message", enabled=True)

        captured = capsys.readouterr()
        assert "[DEBUG] Test message" in captured.out

    def test_debug_log_disabled(self, capsys):
        """Test that debug log doesn't print when disabled."""
        from resilience import debug_log

        debug_log("Test message", enabled=False)

        captured = capsys.readouterr()
        assert "[DEBUG]" not in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
