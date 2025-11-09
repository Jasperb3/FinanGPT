"""
Multi-currency valuation metrics calculation.

This module extends the valuation metrics to support stocks denominated
in different currencies, normalizing all values to a base currency (USD).

Features:
- Automatic FX rate fetching
- Currency normalization for metrics
- Support for 12+ major currencies
- Backward compatible with USD-only data

Author: FinanGPT Enhancement Plan 3 - Phase 2
Created: 2025-11-09
"""

import duckdb
from datetime import date, timedelta
from typing import Optional
import logging


def create_multicurrency_valuation_table(
    conn: duckdb.DuckDBPyConnection,
    base_currency: str = "USD",
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Create valuation metrics table with multi-currency support.

    This function:
    1. Gets latest prices and financials for each ticker
    2. Identifies the currency for each stock
    3. Converts values to base currency using FX rates
    4. Calculates valuation ratios

    Args:
        conn: DuckDB connection
        base_currency: Currency to normalize to (default: USD)
        logger: Optional logger

    Returns:
        Number of rows created

    Example:
        >>> conn = duckdb.connect("financial_data.duckdb")
        >>> rows = create_multicurrency_valuation_table(conn, base_currency="USD")
        >>> print(f"Created {rows} valuation entries")
    """
    if logger:
        logger.info(f"Creating multi-currency valuation table (base: {base_currency})")

    try:
        conn.execute("CREATE SCHEMA IF NOT EXISTS valuation")
        conn.execute("DROP TABLE IF EXISTS valuation.metrics_multicurrency")

        # Create helper function for FX rate lookup
        conn.execute("""
            CREATE OR REPLACE FUNCTION get_fx_rate(
                from_currency VARCHAR,
                to_currency VARCHAR,
                as_of_date DATE
            ) AS (
                -- Try direct rate
                COALESCE(
                    (SELECT rate
                     FROM currency.exchange_rates
                     WHERE base_currency = from_currency
                     AND quote_currency = to_currency
                     AND date = as_of_date
                     LIMIT 1),
                    -- Try inverse rate
                    1.0 / NULLIF((SELECT rate
                                  FROM currency.exchange_rates
                                  WHERE base_currency = to_currency
                                  AND quote_currency = from_currency
                                  AND date = as_of_date
                                  LIMIT 1), 0),
                    -- Default to 1.0 if same currency or not found
                    CASE WHEN from_currency = to_currency THEN 1.0 ELSE NULL END
                )
            )
        """)

        # Create valuation metrics table with currency conversion
        result = conn.execute(f"""
            CREATE TABLE valuation.metrics_multicurrency AS
            WITH latest_prices AS (
                SELECT
                    ticker,
                    close AS price,
                    date AS price_date,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM prices.daily
            ),
            latest_financials AS (
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
            company_info AS (
                SELECT
                    ticker,
                    currency AS local_currency,
                    sharesOutstanding AS shares
                FROM company.metadata
            ),
            latest_dividends AS (
                SELECT
                    ticker,
                    SUM(amount) AS annual_dividend
                FROM dividends.history
                WHERE date >= CURRENT_DATE - INTERVAL '365 days'
                GROUP BY ticker
            )
            SELECT
                lp.ticker,
                lp.price_date,
                lf.fiscal_date,
                ci.local_currency,
                '{base_currency}' AS base_currency,

                -- Local currency values
                lp.price AS price_local,
                (lp.price * ci.shares) AS market_cap_local,
                lf.totalRevenue AS revenue_local,
                lf.netIncome AS net_income_local,
                lf.shareholderEquity AS equity_local,

                -- FX rate (local → base)
                get_fx_rate(
                    ci.local_currency,
                    '{base_currency}',
                    lp.price_date
                ) AS fx_rate,

                -- Base currency values (normalized)
                lp.price * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date)
                    AS price_{base_currency.lower()},

                (lp.price * ci.shares) * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date)
                    AS market_cap_{base_currency.lower()},

                lf.totalRevenue * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date)
                    AS revenue_{base_currency.lower()},

                lf.netIncome * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date)
                    AS net_income_{base_currency.lower()},

                lf.shareholderEquity * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date)
                    AS equity_{base_currency.lower()},

                -- Valuation ratios (currency-neutral)
                (lp.price * ci.shares) / NULLIF(lf.netIncome, 0) AS pe_ratio,
                (lp.price * ci.shares) / NULLIF(lf.totalRevenue, 0) AS ps_ratio,
                (lp.price * ci.shares) / NULLIF(lf.shareholderEquity, 0) AS pb_ratio,

                -- Price per share ratios
                lp.price / NULLIF(lf.netIncome / NULLIF(ci.shares, 0), 0) AS price_to_eps,

                -- Dividend yield (if available)
                CASE
                    WHEN ld.annual_dividend IS NOT NULL AND lp.price > 0
                    THEN (ld.annual_dividend / lp.price) * 100
                    ELSE NULL
                END AS dividend_yield_pct,

                -- Market cap category (in base currency)
                CASE
                    WHEN (lp.price * ci.shares) * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date) >= 10000000000
                        THEN 'Large Cap'
                    WHEN (lp.price * ci.shares) * get_fx_rate(ci.local_currency, '{base_currency}', lp.price_date) >= 2000000000
                        THEN 'Mid Cap'
                    ELSE 'Small Cap'
                END AS market_cap_category

            FROM latest_prices lp
            JOIN latest_financials lf ON lp.ticker = lf.ticker AND lp.rn = 1 AND lf.rn = 1
            JOIN company_info ci ON lp.ticker = ci.ticker
            LEFT JOIN latest_dividends ld ON lp.ticker = ld.ticker
            WHERE ci.local_currency IS NOT NULL
        """)

        count = conn.execute("SELECT COUNT(*) FROM valuation.metrics_multicurrency").fetchone()[0]

        if logger:
            logger.info(f"Created {count} multi-currency valuation entries")

        return count

    except Exception as e:
        if logger:
            logger.error(f"Error creating multi-currency valuation table: {e}")
        raise


def ensure_fx_rates_available(
    conn: duckdb.DuckDBPyConnection,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Ensure FX rates are available for all stocks in the database.

    This function:
    1. Finds all unique currencies in company.metadata
    2. Fetches FX rates for those currencies (if not already cached)
    3. Returns count of rates fetched

    Args:
        conn: DuckDB connection
        logger: Optional logger

    Returns:
        Number of FX rates fetched

    Example:
        >>> rates_fetched = ensure_fx_rates_available(conn)
        >>> print(f"Fetched {rates_fetched} FX rates")
    """
    from src.data.currency import CurrencyConverter

    try:
        # Get unique currencies from company metadata
        currencies = conn.execute("""
            SELECT DISTINCT currency
            FROM company.metadata
            WHERE currency IS NOT NULL
            AND currency != 'USD'
        """).fetchall()

        if not currencies:
            if logger:
                logger.info("No non-USD currencies found")
            return 0

        converter = CurrencyConverter(conn, logger)

        # Get date range from prices
        date_range = conn.execute("""
            SELECT MIN(date), MAX(date)
            FROM prices.daily
        """).fetchone()

        if not date_range[0]:
            if logger:
                logger.warning("No price data found")
            return 0

        start_date = date_range[0]
        end_date = date_range[1] or date.today()

        total_fetched = 0

        for (currency,) in currencies:
            if logger:
                logger.info(f"Fetching {currency}/USD rates...")

            # Fetch currency → USD rates
            count = converter.fetch_rates(
                currency,
                "USD",
                start_date,
                end_date
            )

            total_fetched += count

        if logger:
            logger.info(f"Fetched {total_fetched} total FX rates")

        return total_fetched

    except Exception as e:
        if logger:
            logger.error(f"Error ensuring FX rates: {e}")
        return 0
