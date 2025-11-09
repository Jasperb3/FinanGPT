"""
Test suite for Phase 2: Global Market Support.

Tests cover:
- Market validators (global vs us_only vs custom)
- Currency converter with FX rates
- Multi-currency valuation metrics
- Configuration parsing

Author: FinanGPT Enhancement Plan 3 - Phase 2
Created: 2025-11-09
"""

import pytest
import duckdb
from datetime import date, timedelta
from unittest.mock import Mock, MagicMock, patch


# Test market validators
class TestMarketValidators:
    """Tests for src/ingest/validators.py"""

    def test_global_mode_accepts_all_countries(self):
        """Test that global mode accepts any country."""
        from src.ingest.validators import validate_instrument, DEFAULT_MARKET_CONFIG

        # US stock
        us_info = {
            "symbol": "AAPL",
            "country": "United States",
            "currency": "USD",
            "quoteType": "EQUITY"
        }
        validate_instrument(us_info, DEFAULT_MARKET_CONFIG)  # Should not raise

        # EU stock
        eu_info = {
            "symbol": "BMW.DE",
            "country": "Germany",
            "currency": "EUR",
            "quoteType": "EQUITY"
        }
        validate_instrument(eu_info, DEFAULT_MARKET_CONFIG)  # Should not raise

        # Asia stock
        asia_info = {
            "symbol": "7203.T",
            "country": "Japan",
            "currency": "JPY",
            "quoteType": "EQUITY"
        }
        validate_instrument(asia_info, DEFAULT_MARKET_CONFIG)  # Should not raise

    def test_us_only_mode_rejects_non_us(self):
        """Test that us_only mode rejects non-US stocks."""
        from src.ingest.validators import validate_instrument, US_ONLY_MARKET_CONFIG, UnsupportedInstrument

        # US stock - should pass
        us_info = {
            "symbol": "AAPL",
            "country": "United States",
            "currency": "USD",
            "quoteType": "EQUITY"
        }
        validate_instrument(us_info, US_ONLY_MARKET_CONFIG)  # OK

        # EU stock - should fail
        eu_info = {
            "symbol": "BMW.DE",
            "country": "Germany",
            "currency": "EUR",
            "quoteType": "EQUITY"
        }

        with pytest.raises(UnsupportedInstrument, match="Country"):
            validate_instrument(eu_info, US_ONLY_MARKET_CONFIG)

    def test_etf_detection(self):
        """Test ETF detection across different indicators."""
        from src.ingest.validators import is_etf

        # Detect via quoteType
        assert is_etf({"quoteType": "ETF"}) == True

        # Detect via longName
        assert is_etf({"quoteType": "EQUITY", "longName": "SPDR S&P 500 ETF"}) == True

        # Detect via flag
        assert is_etf({"quoteType": "EQUITY", "isETF": True}) == True

        # Regular stock - not ETF
        assert is_etf({"quoteType": "EQUITY", "longName": "Apple Inc"}) == False

    def test_mutual_fund_detection(self):
        """Test mutual fund detection."""
        from src.ingest.validators import is_mutualfund

        assert is_mutualfund({"quoteType": "MUTUALFUND"}) == True
        assert is_mutualfund({"quoteType": "FUND"}) == True
        assert is_mutualfund({"quoteType": "EQUITY"}) == False

    def test_crypto_detection(self):
        """Test cryptocurrency detection."""
        from src.ingest.validators import is_crypto

        assert is_crypto({"quoteType": "CRYPTOCURRENCY"}) == True
        assert is_crypto({"quoteType": "CRYPTO"}) == True
        assert is_crypto({"quoteType": "EQUITY"}) == False

    def test_custom_config_from_dict(self):
        """Test parsing custom configuration from dict."""
        from src.ingest.validators import get_market_config_from_dict

        config_dict = {
            "mode": "custom",
            "custom": {
                "allowed_countries": ["United States", "United Kingdom"],
                "allowed_currencies": ["USD", "GBP"],
                "allowed_exchanges": ["NYSE", "LSE"]
            },
            "exclude_etfs": True
        }

        config = get_market_config_from_dict(config_dict)

        assert config.allowed_countries == {"United States", "United Kingdom"}
        assert config.allowed_currencies == {"USD", "GBP"}
        assert config.allowed_exchanges == {"NYSE", "LSE"}
        assert config.exclude_etfs == True

    def test_eu_only_mode(self):
        """Test EU-only market configuration."""
        from src.ingest.validators import validate_instrument, EU_ONLY_MARKET_CONFIG

        # German stock - should pass
        de_stock = {
            "symbol": "BMW.DE",
            "country": "Germany",
            "currency": "EUR",
            "quoteType": "EQUITY"
        }
        validate_instrument(de_stock, EU_ONLY_MARKET_CONFIG)  # OK

        # UK stock - should pass
        uk_stock = {
            "symbol": "HSBA.L",
            "country": "United Kingdom",
            "currency": "GBP",
            "quoteType": "EQUITY"
        }
        validate_instrument(uk_stock, EU_ONLY_MARKET_CONFIG)  # OK


# Test currency converter
class TestCurrencyConverter:
    """Tests for src/data/currency.py"""

    def test_fx_table_creation(self):
        """Test that FX rates table is created."""
        from src.data.currency import CurrencyConverter

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        # Verify table exists
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'currency'
        """).fetchall()

        assert ("exchange_rates",) in tables

    @patch('yfinance.Ticker')
    def test_fetch_rates(self, mock_ticker_class):
        """Test fetching FX rates from yfinance."""
        from src.data.currency import CurrencyConverter
        import pandas as pd

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        # Mock yfinance response
        mock_ticker = Mock()
        mock_hist = pd.DataFrame({
            "Close": [1.10, 1.11, 1.12]
        }, index=pd.date_range("2024-01-01", periods=3))
        mock_ticker.history.return_value = mock_hist
        mock_ticker_class.return_value = mock_ticker

        # Fetch rates
        count = converter.fetch_rates(
            "EUR",
            "USD",
            date(2024, 1, 1),
            date(2024, 1, 3)
        )

        assert count == 3

        # Verify rates in database
        rates = conn.execute("""
            SELECT COUNT(*)
            FROM currency.exchange_rates
            WHERE base_currency = 'EUR' AND quote_currency = 'USD'
        """).fetchone()[0]

        assert rates == 3

    def test_same_currency_conversion(self):
        """Test that same currency conversion returns same amount."""
        from src.data.currency import CurrencyConverter

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        result = converter.convert(100, "USD", "USD", date(2024, 1, 1))

        assert result == 100

    def test_conversion_with_cached_rate(self):
        """Test conversion using cached FX rate."""
        from src.data.currency import CurrencyConverter

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        # Manually insert rate
        conn.execute("""
            INSERT INTO currency.exchange_rates
                (base_currency, quote_currency, date, rate, source)
            VALUES ('EUR', 'USD', '2024-01-01', 1.10, 'test')
        """)

        # Convert
        result = converter.convert(100, "EUR", "USD", date(2024, 1, 1))

        assert result == pytest.approx(110.0)

    def test_inverse_rate_conversion(self):
        """Test conversion using inverse rate."""
        from src.data.currency import CurrencyConverter

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        # Insert USD/EUR rate (inverse of EUR/USD)
        conn.execute("""
            INSERT INTO currency.exchange_rates
                (base_currency, quote_currency, date, rate, source)
            VALUES ('USD', 'EUR', '2024-01-01', 0.9091, 'test')
        """)

        # Convert EUR to USD (should use inverse)
        result = converter.convert(100, "EUR", "USD", date(2024, 1, 1))

        # 100 EUR = 100 / 0.9091 USD ≈ 110 USD
        assert result == pytest.approx(110.0, rel=0.01)

    def test_cache_stats(self):
        """Test FX cache statistics."""
        from src.data.currency import CurrencyConverter

        conn = duckdb.connect(":memory:")
        converter = CurrencyConverter(conn)

        # Insert some rates
        conn.execute("""
            INSERT INTO currency.exchange_rates
                (base_currency, quote_currency, date, rate, source)
            VALUES
                ('EUR', 'USD', '2024-01-01', 1.10, 'test'),
                ('EUR', 'USD', '2024-01-02', 1.11, 'test'),
                ('GBP', 'USD', '2024-01-01', 1.25, 'test')
        """)

        stats = converter.get_cache_stats()

        assert stats['total_rates'] == 3
        assert stats['pairs'] == 2  # EUR/USD and GBP/USD
        assert stats['date_range_start'] == date(2024, 1, 1)
        assert stats['date_range_end'] == date(2024, 1, 2)


# Test multi-currency valuation
class TestMulticurrencyValuation:
    """Tests for src/data/valuation_multicurrency.py"""

    def test_fx_rate_function_creation(self):
        """Test that get_fx_rate UDF is created."""
        from src.data.valuation_multicurrency import create_multicurrency_valuation_table

        conn = duckdb.connect(":memory:")

        # Create minimal schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS currency")
        conn.execute("""
            CREATE TABLE currency.exchange_rates (
                base_currency VARCHAR,
                quote_currency VARCHAR,
                date DATE,
                rate DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                PRIMARY KEY (base_currency, quote_currency, date)
            )
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS prices")
        conn.execute("""
            CREATE TABLE prices.daily (
                ticker VARCHAR,
                date DATE,
                close DOUBLE
            )
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS financials")
        conn.execute("""
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                netIncome DOUBLE,
                totalRevenue DOUBLE,
                shareholderEquity DOUBLE,
                totalAssets DOUBLE
            )
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS company")
        conn.execute("""
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                currency VARCHAR,
                sharesOutstanding BIGINT
            )
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS dividends")
        conn.execute("""
            CREATE TABLE dividends.history (
                ticker VARCHAR,
                date DATE,
                amount DOUBLE
            )
        """)

        # Create table (will create UDF)
        try:
            create_multicurrency_valuation_table(conn)
        except:
            pass  # May fail due to empty tables, but UDF should be created

        # Test UDF exists
        result = conn.execute("""
            SELECT get_fx_rate('USD', 'USD', '2024-01-01'::DATE)
        """).fetchone()

        assert result[0] == 1.0  # Same currency returns 1.0

    def test_valuation_with_fx_conversion(self):
        """Test valuation calculation with FX conversion."""
        from src.data.valuation_multicurrency import create_multicurrency_valuation_table

        conn = duckdb.connect(":memory:")

        # Setup schemas and tables
        conn.execute("CREATE SCHEMA IF NOT EXISTS currency")
        conn.execute("""
            CREATE TABLE currency.exchange_rates (
                base_currency VARCHAR,
                quote_currency VARCHAR,
                date DATE,
                rate DOUBLE,
                source VARCHAR DEFAULT 'test',
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert EUR/USD rate
        conn.execute("""
            INSERT INTO currency.exchange_rates
                (base_currency, quote_currency, date, rate)
            VALUES ('EUR', 'USD', '2024-01-01', 1.10)
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS prices")
        conn.execute("""
            CREATE TABLE prices.daily (ticker VARCHAR, date DATE, close DOUBLE)
        """)
        conn.execute("""
            INSERT INTO prices.daily VALUES ('BMW.DE', '2024-01-01', 100.0)
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS financials")
        conn.execute("""
            CREATE TABLE financials.annual (
                ticker VARCHAR,
                date DATE,
                netIncome DOUBLE,
                totalRevenue DOUBLE,
                shareholderEquity DOUBLE,
                totalAssets DOUBLE
            )
        """)
        conn.execute("""
            INSERT INTO financials.annual VALUES
                ('BMW.DE', '2023-12-31', 1000000000, 5000000000, 3000000000, 8000000000)
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS company")
        conn.execute("""
            CREATE TABLE company.metadata (
                ticker VARCHAR,
                currency VARCHAR,
                sharesOutstanding BIGINT
            )
        """)
        conn.execute("""
            INSERT INTO company.metadata VALUES ('BMW.DE', 'EUR', 1000000)
        """)

        conn.execute("CREATE SCHEMA IF NOT EXISTS dividends")
        conn.execute("""
            CREATE TABLE dividends.history (ticker VARCHAR, date DATE, amount DOUBLE)
        """)

        # Create valuation table
        rows = create_multicurrency_valuation_table(conn, base_currency="USD")

        assert rows == 1

        # Verify conversion
        result = conn.execute("""
            SELECT
                ticker,
                local_currency,
                price_local,
                price_usd,
                fx_rate
            FROM valuation.metrics_multicurrency
            WHERE ticker = 'BMW.DE'
        """).fetchone()

        assert result[0] == "BMW.DE"
        assert result[1] == "EUR"
        assert result[2] == 100.0  # Price in EUR
        assert result[3] == pytest.approx(110.0)  # Price in USD (100 * 1.10)
        assert result[4] == 1.10  # FX rate


# Test configuration parsing
class TestGlobalMarketConfiguration:
    """Test configuration loading for global markets."""

    def test_parse_global_mode(self):
        """Test parsing global mode configuration."""
        from src.ingest.validators import get_market_config_from_dict

        config = get_market_config_from_dict({"mode": "global"})

        assert config.allowed_countries is None  # No restrictions
        assert config.allowed_currencies is None
        assert config.exclude_etfs == True

    def test_parse_us_only_mode(self):
        """Test parsing us_only mode configuration."""
        from src.ingest.validators import get_market_config_from_dict

        config = get_market_config_from_dict({"mode": "us_only"})

        assert "United States" in config.allowed_countries
        assert "USD" in config.allowed_currencies

    def test_parse_custom_mode_empty_lists(self):
        """Test that empty lists in custom mode mean no restrictions."""
        from src.ingest.validators import get_market_config_from_dict

        config = get_market_config_from_dict({
            "mode": "custom",
            "custom": {
                "allowed_countries": [],
                "allowed_currencies": [],
                "allowed_exchanges": []
            }
        })

        assert config.allowed_countries is None
        assert config.allowed_currencies is None
        assert config.allowed_exchanges is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
