#!/usr/bin/env python3
"""Analyst intelligence and sentiment data for FinanGPT (Phase 9)."""

from __future__ import annotations

import duckdb
from typing import Any


def create_analyst_recommendations_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create analyst recommendations table with upgrades/downgrades.

    This table tracks analyst rating changes over time, showing when analysts
    upgrade, downgrade, or maintain their ratings for stocks.

    Returns:
        Number of rows in analyst.recommendations table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")

        # Check if base table exists from transform
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'analyst' AND table_name = 'recommendations_raw'
        """).fetchone()[0]

        if table_exists == 0:
            conn.execute("""
                CREATE TABLE analyst.recommendations_raw (
                    ticker VARCHAR,
                    date DATE,
                    firm VARCHAR,
                    from_grade VARCHAR,
                    to_grade VARCHAR,
                    action VARCHAR
                )
            """)

        # Create view for easy querying
        conn.execute("DROP VIEW IF EXISTS analyst.recommendations")
        conn.execute("""
            CREATE VIEW analyst.recommendations AS
            SELECT
                ticker,
                date,
                firm,
                from_grade,
                to_grade,
                action,
                CASE
                    WHEN action = 'up' THEN 1
                    WHEN action = 'down' THEN -1
                    ELSE 0
                END AS action_score
            FROM analyst.recommendations_raw
            WHERE date IS NOT NULL
            ORDER BY ticker, date DESC
        """)

        count = conn.execute("SELECT COUNT(*) FROM analyst.recommendations").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating analyst recommendations table: {e}")
        return 0


def create_price_targets_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create analyst price targets table.

    This table contains consensus price targets from analysts, including
    low, mean, and high targets, plus calculated upside potential.

    Returns:
        Number of rows in analyst.price_targets table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")

        # Check if base table exists from transform
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'analyst' AND table_name = 'price_targets_raw'
        """).fetchone()[0]

        if table_exists == 0:
            conn.execute("""
                CREATE TABLE analyst.price_targets_raw (
                    ticker VARCHAR,
                    date DATE,
                    current_price DOUBLE,
                    target_low DOUBLE,
                    target_mean DOUBLE,
                    target_high DOUBLE,
                    num_analysts INTEGER
                )
            """)

        # Create view with calculated upside
        conn.execute("DROP VIEW IF EXISTS analyst.price_targets")
        conn.execute("""
            CREATE VIEW analyst.price_targets AS
            SELECT
                ticker,
                date,
                current_price,
                target_low,
                target_mean,
                target_high,
                num_analysts,
                CASE
                    WHEN current_price > 0 AND target_mean IS NOT NULL
                    THEN ((target_mean - current_price) / current_price) * 100
                    ELSE NULL
                END AS upside_pct,
                CASE
                    WHEN current_price > 0 AND target_low IS NOT NULL
                    THEN ((target_low - current_price) / current_price) * 100
                    ELSE NULL
                END AS downside_pct,
                CASE
                    WHEN current_price > 0 AND target_high IS NOT NULL
                    THEN ((target_high - current_price) / current_price) * 100
                    ELSE NULL
                END AS max_upside_pct
            FROM analyst.price_targets_raw
            WHERE date IS NOT NULL
            ORDER BY ticker, date DESC
        """)

        count = conn.execute("SELECT COUNT(*) FROM analyst.price_targets").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating price targets table: {e}")
        return 0


def create_analyst_consensus_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create analyst consensus ratings table.

    This table aggregates analyst ratings into a consensus view, showing
    the distribution of buy/hold/sell recommendations and a consensus score.

    Returns:
        Number of rows in analyst.consensus table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")

        # Check if base table exists from transform
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'analyst' AND table_name = 'consensus_raw'
        """).fetchone()[0]

        if table_exists == 0:
            conn.execute("""
                CREATE TABLE analyst.consensus_raw (
                    ticker VARCHAR,
                    date DATE,
                    strong_buy INTEGER,
                    buy INTEGER,
                    hold INTEGER,
                    sell INTEGER,
                    strong_sell INTEGER
                )
            """)

        # Create view with calculated consensus rating
        conn.execute("DROP VIEW IF EXISTS analyst.consensus")
        conn.execute("""
            CREATE VIEW analyst.consensus AS
            SELECT
                ticker,
                date,
                strong_buy,
                buy,
                hold,
                sell,
                strong_sell,
                (strong_buy + buy + hold + sell + strong_sell) AS total_analysts,
                CASE
                    WHEN (strong_buy + buy + hold + sell + strong_sell) > 0
                    THEN (strong_buy * 1.0 + buy * 2.0 + hold * 3.0 + sell * 4.0 + strong_sell * 5.0) /
                         (strong_buy + buy + hold + sell + strong_sell)
                    ELSE NULL
                END AS consensus_rating,
                CASE
                    WHEN (strong_buy + buy + hold + sell + strong_sell) > 0
                    THEN
                        CASE
                            WHEN (strong_buy * 1.0 + buy * 2.0 + hold * 3.0 + sell * 4.0 + strong_sell * 5.0) /
                                 (strong_buy + buy + hold + sell + strong_sell) <= 1.5 THEN 'Strong Buy'
                            WHEN (strong_buy * 1.0 + buy * 2.0 + hold * 3.0 + sell * 4.0 + strong_sell * 5.0) /
                                 (strong_buy + buy + hold + sell + strong_sell) <= 2.5 THEN 'Buy'
                            WHEN (strong_buy * 1.0 + buy * 2.0 + hold * 3.0 + sell * 4.0 + strong_sell * 5.0) /
                                 (strong_buy + buy + hold + sell + strong_sell) <= 3.5 THEN 'Hold'
                            WHEN (strong_buy * 1.0 + buy * 2.0 + hold * 3.0 + sell * 4.0 + strong_sell * 5.0) /
                                 (strong_buy + buy + hold + sell + strong_sell) <= 4.5 THEN 'Sell'
                            ELSE 'Strong Sell'
                        END
                    ELSE NULL
                END AS consensus_label
            FROM analyst.consensus_raw
            WHERE date IS NOT NULL
            ORDER BY ticker, date DESC
        """)

        count = conn.execute("SELECT COUNT(*) FROM analyst.consensus").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating analyst consensus table: {e}")
        return 0


def create_growth_estimates_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create analyst growth estimates table.

    This table contains analyst estimates for growth rates across different
    time periods (quarterly, annual, 5-year).

    Returns:
        Number of rows in analyst.growth_estimates table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")

        # Check if base table exists from transform
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'analyst' AND table_name = 'growth_estimates_raw'
        """).fetchone()[0]

        if table_exists == 0:
            conn.execute("""
                CREATE TABLE analyst.growth_estimates_raw (
                    ticker VARCHAR,
                    date DATE,
                    current_qtr_growth DOUBLE,
                    next_qtr_growth DOUBLE,
                    current_year_growth DOUBLE,
                    next_year_growth DOUBLE,
                    next_5yr_growth DOUBLE
                )
            """)

        # Create view with calculated forward PEG ratio
        conn.execute("DROP VIEW IF EXISTS analyst.growth_estimates")
        conn.execute("""
            CREATE VIEW analyst.growth_estimates AS
            SELECT
                g.ticker,
                g.date,
                g.current_qtr_growth,
                g.next_qtr_growth,
                g.current_year_growth,
                g.next_year_growth,
                g.next_5yr_growth,
                CASE
                    WHEN g.next_5yr_growth > 0 AND v.pe_ratio IS NOT NULL
                    THEN v.pe_ratio / g.next_5yr_growth
                    ELSE NULL
                END AS peg_forward
            FROM analyst.growth_estimates_raw g
            LEFT JOIN valuation.metrics v ON g.ticker = v.ticker
            WHERE g.date IS NOT NULL
            ORDER BY g.ticker, g.date DESC
        """)

        count = conn.execute("SELECT COUNT(*) FROM analyst.growth_estimates").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating growth estimates table: {e}")
        return 0
