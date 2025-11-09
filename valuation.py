#!/usr/bin/env python3
"""Calculate valuation metrics and earnings data for FinanGPT (Phase 8)."""

from __future__ import annotations

import duckdb
from typing import Any


def create_valuation_metrics_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create valuation metrics table with P/E, P/B, P/S, PEG, dividend yield, etc.

    This function calculates market-based valuation ratios by combining:
    - Latest stock prices (from prices.daily)
    - Financial data (from financials.annual)
    - Dividend data (from dividends.history)
    - Company metadata (from company.metadata)
    - Growth metrics (from growth.annual)

    Returns:
        Number of rows created in valuation.metrics table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS valuation")
        conn.execute("DROP TABLE IF EXISTS valuation.metrics")

        # Create valuation metrics table
        result = conn.execute("""
            CREATE TABLE valuation.metrics AS
            WITH latest_prices AS (
                -- Get the most recent price for each ticker
                SELECT
                    ticker,
                    close AS price,
                    date AS price_date,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM prices.daily
            ),
            latest_financials AS (
                -- Get the most recent annual financials
                SELECT
                    ticker,
                    date AS fiscal_date,
                    netIncome,
                    totalRevenue,
                    shareholderEquity,
                    totalAssets,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM financials.annual
                WHERE totalRevenue IS NOT NULL
            ),
            annual_dividends AS (
                -- Calculate annual dividend per ticker
                SELECT
                    ticker,
                    SUM(amount) AS annual_dividend,
                    MAX(date) AS last_dividend_date
                FROM dividends.history
                WHERE date >= CURRENT_DATE - INTERVAL '365 days'
                GROUP BY ticker
            ),
            shares_outstanding AS (
                -- Calculate shares outstanding from market cap and price
                SELECT
                    m.ticker,
                    m.marketCap,
                    CASE
                        WHEN p.price > 0 THEN m.marketCap / p.price
                        ELSE NULL
                    END AS shares
                FROM company.metadata m
                LEFT JOIN latest_prices p ON m.ticker = p.ticker AND p.rn = 1
            )
            SELECT
                p.ticker,
                p.price_date AS date,
                p.price,
                m.marketCap AS market_cap,

                -- P/E Ratio: Price / (Earnings per Share)
                CASE
                    WHEN f.netIncome > 0 AND s.shares > 0
                    THEN p.price / (f.netIncome / s.shares)
                    ELSE NULL
                END AS pe_ratio,

                -- P/B Ratio: Price / (Book Value per Share)
                CASE
                    WHEN f.shareholderEquity > 0 AND s.shares > 0
                    THEN p.price / (f.shareholderEquity / s.shares)
                    ELSE NULL
                END AS pb_ratio,

                -- P/S Ratio: Market Cap / Total Revenue
                CASE
                    WHEN f.totalRevenue > 0 AND m.marketCap > 0
                    THEN m.marketCap / f.totalRevenue
                    ELSE NULL
                END AS ps_ratio,

                -- PEG Ratio: P/E / (Revenue Growth Rate * 100)
                CASE
                    WHEN f.netIncome > 0 AND s.shares > 0 AND g.revenue_growth_yoy > 0
                    THEN (p.price / (f.netIncome / s.shares)) / (g.revenue_growth_yoy * 100)
                    ELSE NULL
                END AS peg_ratio,

                -- Dividend Yield: Annual Dividend / Price
                CASE
                    WHEN p.price > 0 AND d.annual_dividend > 0
                    THEN (d.annual_dividend / p.price)
                    ELSE NULL
                END AS dividend_yield,

                -- Payout Ratio: Dividends / Net Income (annual basis)
                CASE
                    WHEN f.netIncome > 0 AND d.annual_dividend > 0
                    THEN (d.annual_dividend * s.shares) / f.netIncome
                    ELSE NULL
                END AS payout_ratio,

                -- Market Cap Classification
                CASE
                    WHEN m.marketCap >= 10000000000 THEN 'Large Cap'
                    WHEN m.marketCap >= 2000000000 THEN 'Mid Cap'
                    WHEN m.marketCap > 0 THEN 'Small Cap'
                    ELSE 'Unknown'
                END AS cap_class

            FROM latest_prices p
            INNER JOIN shares_outstanding s ON p.ticker = s.ticker
            LEFT JOIN company.metadata m ON p.ticker = m.ticker
            LEFT JOIN latest_financials f ON p.ticker = f.ticker AND f.rn = 1
            LEFT JOIN annual_dividends d ON p.ticker = d.ticker
            LEFT JOIN growth.annual g ON p.ticker = g.ticker AND f.fiscal_date = g.date
            WHERE p.rn = 1
        """)

        count = conn.execute("SELECT COUNT(*) FROM valuation.metrics").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating valuation metrics table: {e}")
        return 0


def create_earnings_history_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create earnings history view that provides a clean interface to earnings data.

    This view is based on the earnings.history_raw table populated during transform.

    Returns:
        Number of rows in earnings.history view
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS earnings")
        conn.execute("DROP VIEW IF EXISTS earnings.history")

        # Create earnings history view (consolidating data from raw table)
        conn.execute("""
            CREATE VIEW earnings.history AS
            SELECT
                ticker,
                fiscal_period,
                report_date,
                eps_estimate,
                eps_actual,
                eps_surprise,
                surprise_pct,
                revenue_estimate,
                revenue_actual
            FROM earnings.history_raw
            WHERE report_date IS NOT NULL
            ORDER BY ticker, report_date DESC
        """)

        count = conn.execute("SELECT COUNT(*) FROM earnings.history").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating earnings history view: {e}")
        return 0


def create_earnings_calendar_view(conn: duckdb.DuckDBPyConnection) -> int:
    """Create earnings calendar view for upcoming earnings dates.

    This view filters to show only future earnings dates.

    Returns:
        Number of rows in earnings.calendar_upcoming view
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS earnings")

        # Check if base table exists, if not create it
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'earnings' AND table_name = 'calendar'
        """).fetchone()[0]

        if table_exists == 0:
            conn.execute("""
                CREATE TABLE earnings.calendar (
                    ticker VARCHAR,
                    earnings_date DATE,
                    period_ending VARCHAR,
                    estimate DOUBLE
                )
            """)

        # Create view for upcoming earnings
        conn.execute("DROP VIEW IF EXISTS earnings.calendar_upcoming")
        conn.execute("""
            CREATE VIEW earnings.calendar_upcoming AS
            SELECT
                ticker,
                earnings_date,
                period_ending,
                estimate
            FROM earnings.calendar
            WHERE earnings_date >= CURRENT_DATE
            ORDER BY earnings_date ASC
        """)

        count = conn.execute("SELECT COUNT(*) FROM earnings.calendar").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating earnings calendar view: {e}")
        return 0
