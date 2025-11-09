"""Tests for Phase 8: Valuation Metrics & Earnings Intelligence"""

import datetime
import unittest
from typing import Any, Dict, List, Mapping

import duckdb
import pandas as pd

from transform import (
    prepare_earnings_calendar_dataframe,
    prepare_earnings_history_dataframe,
)
from valuation import (
    create_earnings_calendar_view,
    create_earnings_history_table,
    create_valuation_metrics_table,
)


class TestEarningsDataFrames(unittest.TestCase):
    """Test earnings data transformation from MongoDB to pandas DataFrames."""

    def test_earnings_history_dataframe(self) -> None:
        """Test earnings history DataFrame preparation."""
        documents = [
            {
                "ticker": "AAPL",
                "date": "2024-01-31T21:00:00+00:00",
                "fiscal_period": "2024-Q1",
                "eps_estimate": 2.10,
                "eps_actual": 2.18,
                "eps_surprise": 0.08,
                "surprise_pct": 3.81,
                "revenue_estimate": 118000000000,
                "revenue_actual": 119600000000,
            },
            {
                "ticker": "MSFT",
                "date": "2024-01-30T21:00:00+00:00",
                "fiscal_period": "2024-Q2",
                "eps_estimate": 2.75,
                "eps_actual": 2.93,
                "eps_surprise": 0.18,
                "surprise_pct": 6.55,
                "revenue_estimate": None,
                "revenue_actual": None,
            },
        ]
        frame = prepare_earnings_history_dataframe(documents)

        # Check columns
        expected_columns = [
            "ticker",
            "report_date",
            "fiscal_period",
            "eps_estimate",
            "eps_actual",
            "eps_surprise",
            "surprise_pct",
            "revenue_estimate",
            "revenue_actual",
        ]
        self.assertEqual(frame.columns.tolist(), expected_columns)

        # Check data types
        self.assertIsInstance(frame["report_date"].iloc[0], datetime.date)
        self.assertEqual(len(frame), 2)

        # Check values
        aapl_row = frame[frame["ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl_row["eps_actual"], 2.18)
        self.assertEqual(aapl_row["eps_surprise"], 0.08)
        self.assertEqual(aapl_row["fiscal_period"], "2024-Q1")

    def test_earnings_calendar_dataframe(self) -> None:
        """Test earnings calendar DataFrame preparation."""
        documents = [
            {
                "ticker": "AAPL",
                "earnings_date": "2025-01-30T21:00:00+00:00",
                "period_ending": "2024-Q4",
                "estimate": 2.50,
            },
            {
                "ticker": "MSFT",
                "earnings_date": "2025-02-01T21:00:00+00:00",
                "period_ending": "2024-Q4",
                "estimate": 3.00,
            },
        ]
        frame = prepare_earnings_calendar_dataframe(documents)

        # Check columns
        expected_columns = ["ticker", "earnings_date", "period_ending", "estimate"]
        self.assertEqual(frame.columns.tolist(), expected_columns)

        # Check data types
        self.assertIsInstance(frame["earnings_date"].iloc[0], datetime.date)
        self.assertEqual(len(frame), 2)

        # Check values
        aapl_row = frame[frame["ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl_row["estimate"], 2.50)
        self.assertEqual(aapl_row["period_ending"], "2024-Q4")

    def test_empty_documents(self) -> None:
        """Test handling of empty document lists."""
        frame_hist = prepare_earnings_history_dataframe([])
        frame_cal = prepare_earnings_calendar_dataframe([])

        self.assertTrue(frame_hist.empty)
        self.assertTrue(frame_cal.empty)


class TestValuationTables(unittest.TestCase):
    """Test valuation metrics table creation in DuckDB."""

    def setUp(self) -> None:
        """Create an in-memory DuckDB connection with sample data."""
        self.conn = duckdb.connect(":memory:")

        # Create necessary schemas
        self.conn.execute("CREATE SCHEMA financials")
        self.conn.execute("CREATE SCHEMA prices")
        self.conn.execute("CREATE SCHEMA dividends")
        self.conn.execute("CREATE SCHEMA company")
        self.conn.execute("CREATE SCHEMA growth")

        # Create sample annual financials table
        self.conn.execute("""
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                netIncome DOUBLE,
                totalRevenue DOUBLE,
                shareholderEquity DOUBLE,
                totalAssets DOUBLE
            )
        """)
        self.conn.execute("""
            INSERT INTO financials.annual VALUES
            ('AAPL', '2023-09-30', 100000000000, 380000000000, 60000000000, 350000000000),
            ('MSFT', '2023-06-30', 75000000000, 210000000000, 200000000000, 400000000000),
            ('GOOGL', '2023-12-31', 60000000000, 280000000000, 250000000000, 400000000000)
        """)

        # Create sample prices table
        self.conn.execute("""
            CREATE TABLE prices.daily (
                ticker VARCHAR,
                date DATE,
                close DOUBLE
            )
        """)
        self.conn.execute("""
            INSERT INTO prices.daily VALUES
            ('AAPL', '2024-01-15', 185.50),
            ('AAPL', '2024-01-14', 183.20),
            ('MSFT', '2024-01-15', 395.00),
            ('MSFT', '2024-01-14', 392.00),
            ('GOOGL', '2024-01-15', 142.00),
            ('GOOGL', '2024-01-14', 140.50)
        """)

        # Create sample company metadata table
        self.conn.execute("""
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                marketCap BIGINT
            )
        """)
        self.conn.execute("""
            INSERT INTO company.metadata VALUES
            ('AAPL', 2900000000000),
            ('MSFT', 2950000000000),
            ('GOOGL', 1800000000000)
        """)

        # Create sample dividends table
        self.conn.execute("""
            CREATE TABLE dividends.history (
                ticker VARCHAR,
                date DATE,
                amount DOUBLE
            )
        """)
        self.conn.execute("""
            INSERT INTO dividends.history VALUES
            ('AAPL', '2023-11-10', 0.24),
            ('AAPL', '2023-08-11', 0.24),
            ('AAPL', '2023-05-12', 0.24),
            ('AAPL', '2023-02-10', 0.23),
            ('MSFT', '2023-11-15', 0.75),
            ('MSFT', '2023-08-16', 0.68),
            ('MSFT', '2023-05-17', 0.68),
            ('MSFT', '2023-02-15', 0.62)
        """)

        # Create sample growth table
        self.conn.execute("""
            CREATE VIEW growth.annual AS
            SELECT
                ticker,
                date,
                totalRevenue,
                netIncome,
                totalRevenue AS prior_revenue,
                netIncome AS prior_income,
                0.15 AS revenue_growth_yoy,
                0.12 AS income_growth_yoy
            FROM financials.annual
        """)

    def tearDown(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def test_valuation_metrics_table_creation(self) -> None:
        """Test that valuation metrics table is created correctly."""
        row_count = create_valuation_metrics_table(self.conn)

        # Should have metrics for all tickers with complete data
        self.assertGreater(row_count, 0)

        # Check that the table was created
        tables = self.conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'valuation' AND table_name = 'metrics'
        """).fetchall()
        self.assertEqual(len(tables), 1)

        # Check columns exist
        columns = self.conn.execute("""
            PRAGMA table_info('valuation.metrics')
        """).fetchall()
        column_names = [col[1] for col in columns]

        expected_columns = [
            "ticker",
            "date",
            "price",
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
            "peg_ratio",
            "dividend_yield",
            "payout_ratio",
            "cap_class",
        ]
        for col in expected_columns:
            self.assertIn(col, column_names)

    def test_valuation_metrics_calculations(self) -> None:
        """Test that valuation ratios are calculated correctly."""
        create_valuation_metrics_table(self.conn)

        # Get AAPL metrics
        result = self.conn.execute("""
            SELECT
                ticker,
                pe_ratio,
                pb_ratio,
                ps_ratio,
                dividend_yield,
                cap_class
            FROM valuation.metrics
            WHERE ticker = 'AAPL'
        """).fetchone()

        self.assertIsNotNone(result)
        ticker, pe_ratio, pb_ratio, ps_ratio, dividend_yield, cap_class = result

        # Verify calculations
        self.assertEqual(ticker, "AAPL")
        self.assertIsNotNone(pe_ratio)  # Should have P/E ratio
        self.assertIsNotNone(pb_ratio)  # Should have P/B ratio
        self.assertIsNotNone(ps_ratio)  # Should have P/S ratio
        self.assertIsNotNone(dividend_yield)  # Should have dividend yield
        self.assertEqual(cap_class, "Large Cap")  # Market cap > $10B

    def test_cap_classification(self) -> None:
        """Test market cap classification logic."""
        create_valuation_metrics_table(self.conn)

        results = self.conn.execute("""
            SELECT ticker, cap_class, market_cap
            FROM valuation.metrics
            ORDER BY ticker
        """).fetchall()

        # All test companies should be Large Cap (> $10B)
        for ticker, cap_class, market_cap in results:
            self.assertEqual(cap_class, "Large Cap")
            self.assertGreater(market_cap, 10000000000)


class TestEarningsTables(unittest.TestCase):
    """Test earnings tables/views creation in DuckDB."""

    def setUp(self) -> None:
        """Create an in-memory DuckDB connection with sample data."""
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("CREATE SCHEMA earnings")

        # Create sample earnings history raw table
        self.conn.execute("""
            CREATE TABLE earnings.history_raw (
                ticker VARCHAR,
                report_date DATE,
                fiscal_period VARCHAR,
                eps_estimate DOUBLE,
                eps_actual DOUBLE,
                eps_surprise DOUBLE,
                surprise_pct DOUBLE,
                revenue_estimate DOUBLE,
                revenue_actual DOUBLE
            )
        """)
        self.conn.execute("""
            INSERT INTO earnings.history_raw VALUES
            ('AAPL', '2024-01-31', '2024-Q1', 2.10, 2.18, 0.08, 3.81, 118000000000, 119600000000),
            ('AAPL', '2023-10-31', '2023-Q4', 1.39, 1.46, 0.07, 5.04, 89000000000, 89500000000),
            ('MSFT', '2024-01-30', '2024-Q2', 2.75, 2.93, 0.18, 6.55, NULL, NULL)
        """)

    def tearDown(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def test_earnings_history_view_creation(self) -> None:
        """Test that earnings history view is created correctly."""
        row_count = create_earnings_history_table(self.conn)

        # Should have all records
        self.assertEqual(row_count, 3)

        # Check that the view was created
        views = self.conn.execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'earnings' AND table_name = 'history'
        """).fetchall()
        self.assertEqual(len(views), 1)

        # Test querying the view
        results = self.conn.execute("""
            SELECT ticker, eps_surprise
            FROM earnings.history
            WHERE ticker = 'AAPL'
            ORDER BY report_date DESC
        """).fetchall()

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], 0.08)  # Most recent surprise

    def test_earnings_calendar_view_creation(self) -> None:
        """Test that earnings calendar view is created correctly."""
        # Create sample calendar table with future dates
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        past_date = datetime.date.today() - datetime.timedelta(days=30)

        self.conn.execute("""
            CREATE TABLE earnings.calendar (
                ticker VARCHAR,
                earnings_date DATE,
                period_ending VARCHAR,
                estimate DOUBLE
            )
        """)
        self.conn.execute(f"""
            INSERT INTO earnings.calendar VALUES
            ('AAPL', '{future_date.isoformat()}', '2025-Q1', 2.50),
            ('MSFT', '{past_date.isoformat()}', '2024-Q4', 2.80)
        """)

        create_earnings_calendar_view(self.conn)

        # Check that the view was created
        views = self.conn.execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'earnings' AND table_name = 'calendar_upcoming'
        """).fetchall()
        self.assertEqual(len(views), 1)

        # Test that upcoming view only shows future earnings
        results = self.conn.execute("""
            SELECT ticker
            FROM earnings.calendar_upcoming
        """).fetchall()

        # Should only have future earnings (AAPL)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "AAPL")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and NULL handling."""

    def setUp(self) -> None:
        """Create an in-memory DuckDB connection."""
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("CREATE SCHEMA earnings")

    def tearDown(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def test_empty_earnings_history_table(self) -> None:
        """Test creating earnings view with empty raw table."""
        self.conn.execute("""
            CREATE TABLE earnings.history_raw (
                ticker VARCHAR,
                report_date DATE,
                fiscal_period VARCHAR,
                eps_estimate DOUBLE,
                eps_actual DOUBLE,
                eps_surprise DOUBLE,
                surprise_pct DOUBLE,
                revenue_estimate DOUBLE,
                revenue_actual DOUBLE
            )
        """)

        row_count = create_earnings_history_table(self.conn)
        self.assertEqual(row_count, 0)

    def test_null_dates_filtered_out(self) -> None:
        """Test that NULL dates are filtered from earnings history view."""
        self.conn.execute("""
            CREATE TABLE earnings.history_raw (
                ticker VARCHAR,
                report_date DATE,
                fiscal_period VARCHAR,
                eps_estimate DOUBLE,
                eps_actual DOUBLE,
                eps_surprise DOUBLE,
                surprise_pct DOUBLE,
                revenue_estimate DOUBLE,
                revenue_actual DOUBLE
            )
        """)
        self.conn.execute("""
            INSERT INTO earnings.history_raw VALUES
            ('AAPL', '2024-01-31', '2024-Q1', 2.10, 2.18, 0.08, 3.81, NULL, NULL),
            ('MSFT', NULL, '2024-Q2', 2.75, 2.93, 0.18, 6.55, NULL, NULL)
        """)

        create_earnings_history_table(self.conn)

        results = self.conn.execute("""
            SELECT COUNT(*)
            FROM earnings.history
        """).fetchone()

        # Should only have 1 row (MSFT filtered out due to NULL date)
        self.assertEqual(results[0], 1)


if __name__ == "__main__":
    unittest.main()
