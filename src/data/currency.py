"""
Currency conversion module with historical FX rates.

This module provides multi-currency support by fetching and caching
historical exchange rates from yfinance.

Features:
- Fetch historical exchange rates (e.g., EUR/USD, GBP/USD)
- Cache rates in DuckDB for offline access
- Point-in-time conversions (for historical financials)
- Cross-rate calculations (EUR→GBP via USD)
- Support for 10+ major currencies

Author: FinanGPT Enhancement Plan 3 - Phase 2
Created: 2025-11-09
"""

from typing import Optional, Dict, List
from datetime import date, timedelta, datetime
import duckdb
import pandas as pd
import yfinance as yf
import logging


# Supported currencies
SUPPORTED_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CNY", "CAD",
    "AUD", "CHF", "HKD", "SGD", "KRW", "INR"
}


class CurrencyConverter:
    """
    Convert financial values between currencies using historical FX rates.

    This class fetches exchange rates from yfinance and caches them in
    DuckDB for fast offline access.

    Example:
        >>> conn = duckdb.connect("financial_data.duckdb")
        >>> converter = CurrencyConverter(conn)
        >>>
        >>> # Fetch EUR/USD rates for last year
        >>> converter.fetch_rates("EUR", "USD",
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31)
        ... )
        >>>
        >>> # Convert €100 to USD as of specific date
        >>> usd_value = converter.convert(100, "EUR", "USD", date(2024, 6, 1))
        >>> print(f"€100 = ${usd_value:.2f}")
        €100 = $107.50
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, logger: Optional[logging.Logger] = None):
        """
        Initialize currency converter.

        Args:
            conn: DuckDB connection for caching rates
            logger: Optional logger for tracking operations
        """
        self.conn = conn
        self.logger = logger or logging.getLogger(__name__)
        self._init_fx_table()

    def _init_fx_table(self):
        """Create exchange rates table if not exists."""
        self.conn.execute("""
            CREATE SCHEMA IF NOT EXISTS currency
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS currency.exchange_rates (
                base_currency VARCHAR,
                quote_currency VARCHAR,
                date DATE,
                rate DOUBLE,
                source VARCHAR DEFAULT 'yfinance',
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (base_currency, quote_currency, date)
            )
        """)

    def fetch_rates(
        self,
        base: str,
        quote: str,
        start_date: date,
        end_date: date
    ) -> int:
        """
        Fetch historical exchange rates from yfinance.

        yfinance ticker format: EURUSD=X for EUR/USD exchange rate

        Args:
            base: Base currency (e.g., "EUR")
            quote: Quote currency (e.g., "USD")
            start_date: Start date for rates
            end_date: End date for rates

        Returns:
            Number of rates fetched and cached

        Example:
            >>> converter.fetch_rates("GBP", "USD",
            ...     start_date=date(2024, 1, 1),
            ...     end_date=date(2024, 12, 31)
            ... )
            365  # Fetched 365 daily rates
        """
        # Validate currencies
        if base not in SUPPORTED_CURRENCIES:
            self.logger.warning(f"Currency {base} not in supported list")

        if quote not in SUPPORTED_CURRENCIES:
            self.logger.warning(f"Currency {quote} not in supported list")

        # yfinance ticker format: EUR/USD = EURUSD=X
        ticker_symbol = f"{base}{quote}=X"

        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                self.logger.warning(f"No data returned for {ticker_symbol}")
                return 0

            # Prepare DataFrame for insertion
            rates_df = pd.DataFrame({
                "base_currency": base,
                "quote_currency": quote,
                "date": hist.index.date,
                "rate": hist["Close"],
                "source": "yfinance"
            })

            # Upsert to DuckDB (replace existing rates)
            self.conn.execute("""
                INSERT OR REPLACE INTO currency.exchange_rates
                    (base_currency, quote_currency, date, rate, source)
                SELECT base_currency, quote_currency, date, rate, source
                FROM rates_df
            """)

            count = len(rates_df)
            self.logger.info(f"Fetched {count} {base}/{quote} rates")

            return count

        except Exception as e:
            self.logger.error(f"Failed to fetch {base}/{quote} rates: {e}")
            return 0

    def get_rate(
        self,
        base: str,
        quote: str,
        as_of_date: date
    ) -> Optional[float]:
        """
        Get exchange rate from cache or fetch if missing.

        Args:
            base: Base currency
            quote: Quote currency
            as_of_date: Date for exchange rate

        Returns:
            Exchange rate or None if not available

        Example:
            >>> rate = converter.get_rate("EUR", "USD", date(2024, 6, 1))
            >>> print(f"EUR/USD: {rate:.4f}")
            EUR/USD: 1.0750
        """
        # Query cache
        result = self.conn.execute("""
            SELECT rate
            FROM currency.exchange_rates
            WHERE base_currency = ? AND quote_currency = ? AND date = ?
        """, [base, quote, as_of_date]).fetchone()

        if result:
            return result[0]

        # Fetch if missing (up to 60 days before requested date)
        start = as_of_date - timedelta(days=60)
        fetched = self.fetch_rates(base, quote, start, as_of_date)

        if fetched > 0:
            # Retry query
            result = self.conn.execute("""
                SELECT rate
                FROM currency.exchange_rates
                WHERE base_currency = ? AND quote_currency = ? AND date = ?
            """, [base, quote, as_of_date]).fetchone()

            if result:
                return result[0]

            # Try nearest date (within 7 days)
            result = self.conn.execute("""
                SELECT rate, date
                FROM currency.exchange_rates
                WHERE base_currency = ? AND quote_currency = ?
                AND ABS(DATEDIFF('day', date, ?)) <= 7
                ORDER BY ABS(DATEDIFF('day', date, ?))
                LIMIT 1
            """, [base, quote, as_of_date, as_of_date]).fetchone()

            if result:
                self.logger.warning(
                    f"Using nearest rate for {base}/{quote} on {as_of_date}: "
                    f"{result[1]} (rate={result[0]:.4f})"
                )
                return result[0]

        return None

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        as_of_date: date
    ) -> float:
        """
        Convert amount from one currency to another as of specific date.

        Supports cross-rate calculations (e.g., EUR→GBP via USD).

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            as_of_date: Date for conversion rate

        Returns:
            Converted amount

        Raises:
            ValueError: If exchange rate not available

        Example:
            >>> usd_amount = converter.convert(100, "EUR", "USD", date(2024, 6, 1))
            >>> print(f"€100 = ${usd_amount:.2f}")
            €100 = $107.50
        """
        # Shortcut: same currency
        if from_currency == to_currency:
            return amount

        # Try direct rate (FROM/TO)
        rate = self.get_rate(from_currency, to_currency, as_of_date)

        if rate is not None:
            return amount * rate

        # Try inverse rate (TO/FROM)
        inverse_rate = self.get_rate(to_currency, from_currency, as_of_date)

        if inverse_rate is not None:
            return amount / inverse_rate

        # Try cross rate via USD (FROM→USD→TO)
        if from_currency != "USD" and to_currency != "USD":
            try:
                usd_amount = self.convert(amount, from_currency, "USD", as_of_date)
                return self.convert(usd_amount, "USD", to_currency, as_of_date)
            except ValueError:
                pass  # Fall through to error

        raise ValueError(
            f"Exchange rate not available: {from_currency}/{to_currency} on {as_of_date}. "
            f"Try fetching rates first with fetch_rates()"
        )

    def prefetch_common_pairs(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, int]:
        """
        Prefetch common currency pairs for faster conversions.

        Fetches all major currencies vs USD.

        Args:
            start_date: Start date for rates
            end_date: End date for rates

        Returns:
            Dictionary of pair → count fetched

        Example:
            >>> results = converter.prefetch_common_pairs(
            ...     date(2024, 1, 1),
            ...     date(2024, 12, 31)
            ... )
            >>> print(f"Fetched {sum(results.values())} total rates")
        """
        # Common pairs (all vs USD)
        major_currencies = ["EUR", "GBP", "JPY", "CNY", "CAD", "AUD", "CHF"]

        results = {}

        for currency in major_currencies:
            if currency == "USD":
                continue

            # Fetch both directions
            count1 = self.fetch_rates(currency, "USD", start_date, end_date)
            count2 = self.fetch_rates("USD", currency, start_date, end_date)

            results[f"{currency}/USD"] = count1
            results[f"USD/{currency}"] = count2

        return results

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about cached exchange rates.

        Returns:
            Dictionary with total_rates, pairs, date_range

        Example:
            >>> stats = converter.get_cache_stats()
            >>> print(f"Cached {stats['total_rates']} rates for {stats['pairs']} pairs")
        """
        # Total rates
        total = self.conn.execute("""
            SELECT COUNT(*) FROM currency.exchange_rates
        """).fetchone()[0]

        # Unique pairs
        pairs = self.conn.execute("""
            SELECT COUNT(DISTINCT base_currency || '/' || quote_currency)
            FROM currency.exchange_rates
        """).fetchone()[0]

        # Date range
        date_range = self.conn.execute("""
            SELECT MIN(date), MAX(date)
            FROM currency.exchange_rates
        """).fetchone()

        return {
            "total_rates": total,
            "pairs": pairs,
            "date_range_start": date_range[0] if date_range[0] else None,
            "date_range_end": date_range[1] if date_range[1] else None
        }
