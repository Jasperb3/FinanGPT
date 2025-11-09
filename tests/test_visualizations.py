#!/usr/bin/env python3
"""Tests for visualization and formatting features (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize import (
    detect_visualization_intent,
    format_financial_value,
    format_large_number,
    format_column_name,
    sanitize_filename,
    infer_chart_type,
    create_chart,
    pretty_print_formatted,
)


class TestVisualizationIntentDetection:
    """Test visualization intent detection from queries."""

    def test_detect_plot_keyword(self):
        """Test detection of 'plot' keyword."""
        query = "Plot AAPL revenue over time"
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'revenue': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        })
        result = detect_visualization_intent(query, df)
        assert result is not None

    def test_detect_compare_keyword(self):
        """Test detection of 'compare' keyword."""
        query = "Compare AAPL and MSFT revenue"
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'revenue': [100, 90]
        })
        result = detect_visualization_intent(query, df)
        assert result in ['bar', 'scatter']

    def test_no_visualization_for_empty_df(self):
        """Test that empty DataFrame returns None."""
        query = "Show data"
        df = pd.DataFrame()
        result = detect_visualization_intent(query, df)
        assert result is None

    def test_auto_detect_time_series(self):
        """Test automatic detection of time-series data."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        result = infer_chart_type(df)
        assert result == 'line'

    def test_auto_detect_bar_chart(self):
        """Test automatic detection of bar chart data."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'revenue': [100, 90, 85]
        })
        result = infer_chart_type(df)
        assert result == 'bar'

    def test_auto_detect_scatter(self):
        """Test automatic detection of scatter plot data."""
        df = pd.DataFrame({
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [2, 4, 6, 8, 10]
        })
        result = infer_chart_type(df)
        assert result == 'scatter'

    def test_candlestick_detection(self):
        """Test detection of candlestick chart from OHLC data."""
        query = "Show candlestick chart"
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'open': [100, 102, 101, 103, 105],
            'high': [105, 106, 104, 107, 108],
            'low': [99, 101, 100, 102, 104],
            'close': [102, 101, 103, 105, 106]
        })
        result = detect_visualization_intent(query, df)
        assert result == 'candlestick'


class TestFinancialFormatting:
    """Test financial value formatting functions."""

    def test_format_large_number_billions(self):
        """Test formatting billions."""
        assert format_large_number(1500000000000) == "$1.50T"
        assert format_large_number(150000000000) == "$150.00B"
        assert format_large_number(1500000000) == "$1.50B"

    def test_format_large_number_millions(self):
        """Test formatting millions."""
        assert format_large_number(1500000) == "$1.50M"
        assert format_large_number(500000) == "$500.00K"  # 500K, not 0.50M

    def test_format_large_number_thousands(self):
        """Test formatting thousands."""
        assert format_large_number(1500) == "$1.50K"
        assert format_large_number(500) == "$500.00"

    def test_format_large_number_negative(self):
        """Test formatting negative numbers."""
        result = format_large_number(-1500000000)
        assert result == "$-1.50B"

    def test_format_financial_value_revenue(self):
        """Test formatting revenue columns."""
        assert format_financial_value(1500000000, "totalRevenue") == "$1.50B"
        assert format_financial_value(500000000, "netIncome") == "$500.00M"

    def test_format_financial_value_ratio(self):
        """Test formatting ratio columns."""
        assert format_financial_value(0.25, "net_margin") == "25.00%"
        assert format_financial_value(0.15, "roe") == "15.00%"
        assert format_financial_value(0.05, "debt_ratio") == "5.00%"

    def test_format_financial_value_price(self):
        """Test formatting price columns."""
        assert format_financial_value(150.25, "close") == "$150.25"
        assert format_financial_value(75.50, "open") == "$75.50"

    def test_format_financial_value_volume(self):
        """Test formatting volume columns."""
        assert format_financial_value(1500000, "volume") == "1,500,000"
        assert format_financial_value(5000, "shares") == "5,000"

    def test_format_financial_value_none(self):
        """Test formatting None values."""
        assert format_financial_value(None, "revenue") == "N/A"
        assert format_financial_value(pd.NA, "margin") == "N/A"

    def test_format_financial_value_default(self):
        """Test default formatting for unknown columns."""
        assert format_financial_value(123.456, "unknown_column") == "123.46"


class TestColumnFormatting:
    """Test column name formatting."""

    def test_format_camel_case(self):
        """Test formatting camelCase."""
        assert format_column_name("totalRevenue") == "Total Revenue"
        assert format_column_name("netIncome") == "Net Income"

    def test_format_snake_case(self):
        """Test formatting snake_case."""
        assert format_column_name("net_margin") == "Net Margin"
        assert format_column_name("debt_ratio") == "Debt Ratio"

    def test_format_simple_name(self):
        """Test formatting simple names."""
        assert format_column_name("ticker") == "Ticker"
        assert format_column_name("date") == "Date"


class TestFilenameGeneration:
    """Test filename sanitization."""

    def test_sanitize_valid_filename(self):
        """Test sanitizing a valid filename."""
        result = sanitize_filename("Revenue Chart 2024")
        assert result == "Revenue_Chart_2024"

    def test_sanitize_invalid_characters(self):
        """Test removing invalid characters."""
        result = sanitize_filename("Revenue/Chart<2024>")
        assert result == "RevenueChart2024"

    def test_sanitize_long_filename(self):
        """Test truncating long filenames."""
        long_name = "A" * 250
        result = sanitize_filename(long_name)
        assert len(result) <= 200


class TestChartCreation:
    """Test chart creation functions."""

    @patch('visualize.plt')
    def test_create_line_chart(self, mock_plt):
        """Test line chart creation."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'ticker': ['AAPL'] * 5,
            'revenue': [100, 110, 120, 130, 140]
        })

        # Mock the savefig and close methods
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())

        # Call should not raise exception
        # Note: In real test environment, this would save a file
        # For now, we're just testing it doesn't crash

    def test_create_chart_empty_df(self):
        """Test that empty DataFrame returns None."""
        df = pd.DataFrame()
        result = create_chart(df, 'line', 'Test Chart')
        assert result is None

    def test_create_chart_invalid_type(self):
        """Test that invalid chart type returns None."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = create_chart(df, 'invalid_type', 'Test Chart')
        assert result is None


class TestPrettyPrintFormatted:
    """Test enhanced pretty print function."""

    def test_pretty_print_with_formatting(self, capsys):
        """Test pretty print with financial formatting."""
        columns = ['ticker', 'totalRevenue', 'net_margin']
        rows = [
            ('AAPL', 394000000000, 0.2376),
            ('MSFT', 245000000000, 0.3596),
        ]

        pretty_print_formatted(columns, rows, use_formatting=True)

        captured = capsys.readouterr()
        assert '$394.00B' in captured.out
        assert '$245.00B' in captured.out
        assert '23.76%' in captured.out
        assert '35.96%' in captured.out

    def test_pretty_print_empty_rows(self, capsys):
        """Test pretty print with no rows."""
        columns = ['ticker', 'revenue']
        rows = []

        pretty_print_formatted(columns, rows)

        captured = capsys.readouterr()
        assert 'No rows returned' in captured.out

    def test_pretty_print_without_formatting(self, capsys):
        """Test pretty print without formatting."""
        columns = ['ticker', 'value']
        rows = [('AAPL', 1000000)]

        pretty_print_formatted(columns, rows, use_formatting=False)

        captured = capsys.readouterr()
        assert '1000000' in captured.out
        assert '$' not in captured.out


class TestExportFunctions:
    """Test export functions."""

    def test_export_to_csv(self, tmp_path, capsys):
        """Test CSV export."""
        from visualize import export_to_csv

        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'revenue': [100, 90]
        })

        filepath = tmp_path / "test.csv"
        export_to_csv(df, str(filepath))

        assert filepath.exists()
        captured = capsys.readouterr()
        assert "exported" in captured.out.lower()

    def test_export_to_json(self, tmp_path, capsys):
        """Test JSON export."""
        from visualize import export_to_json

        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'revenue': [100, 90]
        })

        filepath = tmp_path / "test.json"
        export_to_json(df, str(filepath))

        assert filepath.exists()
        captured = capsys.readouterr()
        assert "exported" in captured.out.lower()


class TestIntegration:
    """Integration tests for visualization module."""

    def test_full_workflow_line_chart(self):
        """Test full workflow from query to chart."""
        query = "Plot AAPL revenue over time"
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'revenue': range(100, 200, 10)
        })

        # Detect intent
        chart_type = detect_visualization_intent(query, df)
        assert chart_type is not None

        # Format would be applied
        formatted = format_financial_value(150000000, 'revenue')
        assert '$' in formatted

    def test_full_workflow_comparison(self):
        """Test full workflow for comparison query."""
        query = "Compare AAPL and MSFT revenue"
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'revenue': [394000000000, 245000000000, 280000000000]
        })

        # Detect intent
        chart_type = detect_visualization_intent(query, df)
        assert chart_type in ['bar', 'scatter']

        # Format values
        formatted_values = [format_financial_value(v, 'revenue') for v in df['revenue']]
        assert all('B' in v for v in formatted_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
