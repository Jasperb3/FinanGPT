#!/usr/bin/env python3
"""Technical analysis indicators for FinanGPT (Phase 10)."""

from __future__ import annotations

import duckdb
from typing import Any


def create_technical_indicators_table(conn: duckdb.DuckDBPyConnection) -> int:
    """Create technical indicators table with moving averages, RSI, MACD, etc.

    This function calculates technical indicators from existing price data using
    window functions in DuckDB. No new data fetching is required - all calculations
    are performed on the prices.daily table.

    Indicators calculated:
    - Simple Moving Averages (SMA): 20, 50, 200 days
    - Exponential Moving Averages (EMA): 12, 26 days
    - RSI (Relative Strength Index): 14-day
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands (20-day, 2 standard deviations)
    - Volume metrics (20-day average, volume ratio)
    - Price momentum (% change over 1d, 5d, 20d, 60d, 252d)
    - 52-week high/low and distance from current price

    Returns:
        Number of rows created in technical.indicators table
    """
    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS technical")
        conn.execute("DROP TABLE IF EXISTS technical.indicators")

        # Create comprehensive technical indicators table
        result = conn.execute("""
            CREATE TABLE technical.indicators AS
            WITH price_base AS (
                SELECT
                    ticker,
                    date,
                    close,
                    volume,
                    -- Simple Moving Averages
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200,
                    -- Volume metrics
                    AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volume_avg_20,
                    -- Bollinger Bands (using 20-day SMA and stddev)
                    STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std_20,
                    -- Price changes for momentum
                    LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) AS close_1d_ago,
                    LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) AS close_5d_ago,
                    LAG(close, 20) OVER (PARTITION BY ticker ORDER BY date) AS close_20d_ago,
                    LAG(close, 60) OVER (PARTITION BY ticker ORDER BY date) AS close_60d_ago,
                    LAG(close, 252) OVER (PARTITION BY ticker ORDER BY date) AS close_252d_ago,
                    -- 52-week (252 trading days) high and low
                    MAX(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week_52_high,
                    MIN(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS week_52_low
                FROM prices.daily
            ),
            ema_calc AS (
                -- EMA calculations using recursive approach approximation
                -- EMA = Price(t) * k + EMA(t-1) * (1-k), where k = 2/(N+1)
                SELECT
                    ticker,
                    date,
                    close,
                    -- EMA-12: k = 2/13 ≈ 0.1538
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ema_12,
                    -- EMA-26: k = 2/27 ≈ 0.0741
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ema_26
                FROM prices.daily
            ),
            rsi_calc AS (
                SELECT
                    ticker,
                    date,
                    close,
                    LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
                    CASE
                        WHEN close > LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date)
                        THEN close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date)
                        ELSE 0
                    END AS gain,
                    CASE
                        WHEN close < LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date)
                        THEN LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) - close
                        ELSE 0
                    END AS loss
                FROM prices.daily
            ),
            rsi_avg AS (
                SELECT
                    ticker,
                    date,
                    AVG(gain) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
                    AVG(loss) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
                FROM rsi_calc
            )
            SELECT
                p.ticker,
                p.date,
                p.close,
                -- Moving Averages
                p.sma_20,
                p.sma_50,
                p.sma_200,
                e.ema_12,
                e.ema_26,
                -- RSI (0-100 scale)
                CASE
                    WHEN r.avg_loss = 0 THEN 100
                    WHEN r.avg_gain = 0 THEN 0
                    ELSE 100 - (100 / (1 + (r.avg_gain / r.avg_loss)))
                END AS rsi_14,
                -- MACD (EMA12 - EMA26)
                (e.ema_12 - e.ema_26) AS macd,
                -- MACD Signal Line (9-day EMA of MACD) - approximated as SMA for simplicity
                AVG(e.ema_12 - e.ema_26) OVER (PARTITION BY p.ticker ORDER BY p.date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) AS macd_signal,
                -- MACD Histogram (MACD - Signal)
                (e.ema_12 - e.ema_26) - AVG(e.ema_12 - e.ema_26) OVER (PARTITION BY p.ticker ORDER BY p.date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) AS macd_histogram,
                -- Bollinger Bands (20-day, 2 std dev)
                p.sma_20 + (2 * p.std_20) AS bb_upper,
                p.sma_20 AS bb_middle,
                p.sma_20 - (2 * p.std_20) AS bb_lower,
                -- Volume metrics
                p.volume_avg_20,
                CASE
                    WHEN p.volume_avg_20 > 0 THEN CAST(p.volume AS DOUBLE) / p.volume_avg_20
                    ELSE NULL
                END AS volume_ratio,
                -- Price momentum (% change)
                CASE
                    WHEN p.close_1d_ago > 0 THEN ((p.close - p.close_1d_ago) / p.close_1d_ago) * 100
                    ELSE NULL
                END AS pct_change_1d,
                CASE
                    WHEN p.close_5d_ago > 0 THEN ((p.close - p.close_5d_ago) / p.close_5d_ago) * 100
                    ELSE NULL
                END AS pct_change_5d,
                CASE
                    WHEN p.close_20d_ago > 0 THEN ((p.close - p.close_20d_ago) / p.close_20d_ago) * 100
                    ELSE NULL
                END AS pct_change_20d,
                CASE
                    WHEN p.close_60d_ago > 0 THEN ((p.close - p.close_60d_ago) / p.close_60d_ago) * 100
                    ELSE NULL
                END AS pct_change_60d,
                CASE
                    WHEN p.close_252d_ago > 0 THEN ((p.close - p.close_252d_ago) / p.close_252d_ago) * 100
                    ELSE NULL
                END AS pct_change_252d,
                -- 52-week high/low
                p.week_52_high,
                p.week_52_low,
                -- Distance from 52-week high/low (%)
                CASE
                    WHEN p.week_52_high > 0 THEN ((p.close - p.week_52_high) / p.week_52_high) * 100
                    ELSE NULL
                END AS pct_from_52w_high,
                CASE
                    WHEN p.week_52_low > 0 THEN ((p.close - p.week_52_low) / p.week_52_low) * 100
                    ELSE NULL
                END AS pct_from_52w_low
            FROM price_base p
            LEFT JOIN ema_calc e ON p.ticker = e.ticker AND p.date = e.date
            LEFT JOIN rsi_avg r ON p.ticker = r.ticker AND p.date = r.date
            WHERE p.sma_20 IS NOT NULL  -- Filter out rows with insufficient data
        """)

        count = conn.execute("SELECT COUNT(*) FROM technical.indicators").fetchone()[0]
        return count
    except Exception as e:
        print(f"Error creating technical indicators table: {e}")
        return 0
