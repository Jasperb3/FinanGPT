"""
Global market validators for flexible instrument filtering.

This module provides configurable validation for stock market instruments,
supporting US-only mode (legacy), global mode, or custom restrictions.

Features:
- Configurable country/currency/exchange restrictions
- ETF/mutual fund/crypto detection
- Flexible validation modes (global, us_only, custom)
- Detailed rejection reasons

Author: FinanGPT Enhancement Plan 3 - Phase 2
Created: 2025-11-09
"""

from typing import Dict, Set, Optional, Any, Mapping
from dataclasses import dataclass


# Allowed equity types (exclude funds, trusts, etc.)
ALLOWED_EQUITY_TYPES = {
    "EQUITY",
    "COMMON_STOCK",
    "COMMONSTOCK",
    "STOCK",
}


@dataclass
class MarketConfig:
    """
    Configuration for market/currency restrictions.

    This allows flexible filtering of instruments based on geography,
    currency, or exchange.

    Attributes:
        allowed_countries: Set of allowed countries (None = all allowed)
        allowed_currencies: Set of allowed currencies (None = all allowed)
        allowed_exchanges: Set of allowed exchanges (None = all allowed)
        exclude_etfs: Whether to exclude ETFs (default: True)
        exclude_mutualfunds: Whether to exclude mutual funds (default: True)
        exclude_crypto: Whether to exclude cryptocurrencies (default: True)

    Example:
        >>> # Global mode (accept everything except ETFs/funds/crypto)
        >>> config = MarketConfig()
        >>>
        >>> # US-only mode (legacy behavior)
        >>> config = MarketConfig(
        ...     allowed_countries={"United States"},
        ...     allowed_currencies={"USD"}
        ... )
        >>>
        >>> # European stocks only
        >>> config = MarketConfig(
        ...     allowed_countries={"United Kingdom", "Germany", "France"},
        ...     allowed_currencies={"GBP", "EUR"}
        ... )
    """
    allowed_countries: Optional[Set[str]] = None
    allowed_currencies: Optional[Set[str]] = None
    allowed_exchanges: Optional[Set[str]] = None
    exclude_etfs: bool = True
    exclude_mutualfunds: bool = True
    exclude_crypto: bool = True


# Predefined configurations
DEFAULT_MARKET_CONFIG = MarketConfig(
    allowed_countries=None,  # All countries
    allowed_currencies=None,  # All currencies
    allowed_exchanges=None,  # All exchanges
    exclude_etfs=True,
    exclude_mutualfunds=True,
    exclude_crypto=True
)

US_ONLY_MARKET_CONFIG = MarketConfig(
    allowed_countries={"United States", "United States of America", "USA", "US"},
    allowed_currencies={"USD"},
    allowed_exchanges=None,  # Don't restrict by exchange
    exclude_etfs=True,
    exclude_mutualfunds=True,
    exclude_crypto=True
)

EU_ONLY_MARKET_CONFIG = MarketConfig(
    allowed_countries={"United Kingdom", "Germany", "France", "Netherlands",
                      "Spain", "Italy", "Ireland", "Belgium", "Austria",
                      "Sweden", "Denmark", "Finland", "Norway"},
    allowed_currencies={"EUR", "GBP", "SEK", "DKK", "NOK", "CHF"},
    allowed_exchanges=None,
    exclude_etfs=True,
    exclude_mutualfunds=True,
    exclude_crypto=True
)


class UnsupportedInstrument(RuntimeError):
    """Raised when a ticker fails validation checks."""
    pass


def _as_text(value: Any) -> str:
    """Safely convert value to string."""
    if value is None:
        return ""
    return str(value).strip()


def is_etf(info: Mapping[str, Any]) -> bool:
    """
    Check if instrument is an ETF.

    Uses multiple heuristics to detect ETFs:
    - quoteType field
    - Boolean flags (isETF, fundFamily)
    - Name contains "ETF"

    Args:
        info: yfinance info dictionary

    Returns:
        True if ETF detected, False otherwise

    Example:
        >>> info = {"quoteType": "ETF", "longName": "SPDR S&P 500 ETF"}
        >>> is_etf(info)
        True
    """
    quote_type = _as_text(info.get("quoteType")).upper()

    # Check quote type
    if quote_type == "ETF":
        return True

    # Fail closed if no quote type
    if not quote_type:
        return True

    # Check if quote type is not allowed
    if quote_type not in ALLOWED_EQUITY_TYPES:
        return True

    # Check boolean flags
    boolean_flags = (
        info.get("isETF"),
        info.get("isEtf"),
        info.get("fundFamily"),
    )
    if any(str(flag).lower() == "true" for flag in boolean_flags):
        return True

    # Check name
    long_name = _as_text(info.get("longName")).upper()
    return " ETF" in f" {long_name} "


def is_mutualfund(info: Mapping[str, Any]) -> bool:
    """
    Check if instrument is a mutual fund.

    Args:
        info: yfinance info dictionary

    Returns:
        True if mutual fund detected

    Example:
        >>> info = {"quoteType": "MUTUALFUND"}
        >>> is_mutualfund(info)
        True
    """
    quote_type = _as_text(info.get("quoteType")).upper()
    return quote_type in ("MUTUALFUND", "FUND")


def is_crypto(info: Mapping[str, Any]) -> bool:
    """
    Check if instrument is a cryptocurrency.

    Args:
        info: yfinance info dictionary

    Returns:
        True if cryptocurrency detected

    Example:
        >>> info = {"quoteType": "CRYPTOCURRENCY"}
        >>> is_crypto(info)
        True
    """
    quote_type = _as_text(info.get("quoteType")).upper()
    return quote_type in ("CRYPTOCURRENCY", "CRYPTO")


def validate_instrument(
    info: Mapping[str, Any],
    config: MarketConfig = DEFAULT_MARKET_CONFIG
) -> None:
    """
    Validate if instrument meets configured criteria.

    Raises UnsupportedInstrument if validation fails, with detailed reason.

    Args:
        info: yfinance info dictionary
        config: Market configuration

    Raises:
        UnsupportedInstrument: If instrument doesn't meet criteria

    Example:
        >>> info = {"country": "United States", "currency": "USD", "quoteType": "EQUITY"}
        >>> validate_instrument(info, US_ONLY_MARKET_CONFIG)  # OK
        >>>
        >>> info = {"country": "Japan", "currency": "JPY", "quoteType": "EQUITY"}
        >>> validate_instrument(info, US_ONLY_MARKET_CONFIG)  # Raises
        UnsupportedInstrument: Country 'Japan' not in allowed list...
    """
    symbol = info.get("symbol", "Unknown")

    # ETF check (always applied if configured)
    if config.exclude_etfs and is_etf(info):
        raise UnsupportedInstrument(
            f"{symbol}: Rejected - ETF not supported"
        )

    # Mutual fund check
    if config.exclude_mutualfunds and is_mutualfund(info):
        raise UnsupportedInstrument(
            f"{symbol}: Rejected - Mutual fund not supported"
        )

    # Crypto check
    if config.exclude_crypto and is_crypto(info):
        raise UnsupportedInstrument(
            f"{symbol}: Rejected - Cryptocurrency not supported"
        )

    # Country check (if configured)
    if config.allowed_countries is not None:
        country = _as_text(info.get("country"))
        country_normalized = country.lower().strip()

        # Normalize allowed countries for case-insensitive comparison
        allowed_lower = {c.lower().strip() for c in config.allowed_countries}

        if country_normalized not in allowed_lower:
            raise UnsupportedInstrument(
                f"{symbol}: Country '{country}' not in allowed list. "
                f"Allowed: {', '.join(sorted(config.allowed_countries))}"
            )

    # Currency check (if configured)
    if config.allowed_currencies is not None:
        currency = _as_text(
            info.get("financialCurrency") or info.get("currency")
        ).upper()

        if currency not in config.allowed_currencies:
            raise UnsupportedInstrument(
                f"{symbol}: Currency '{currency}' not in allowed list. "
                f"Allowed: {', '.join(sorted(config.allowed_currencies))}"
            )

    # Exchange check (if configured)
    if config.allowed_exchanges is not None:
        exchange = _as_text(info.get("exchange"))
        exchange_normalized = exchange.upper().strip()

        # Normalize allowed exchanges
        allowed_exchanges_upper = {ex.upper().strip() for ex in config.allowed_exchanges}

        if exchange_normalized not in allowed_exchanges_upper:
            raise UnsupportedInstrument(
                f"{symbol}: Exchange '{exchange}' not in allowed list. "
                f"Allowed: {', '.join(sorted(config.allowed_exchanges))}"
            )


def get_market_config_from_dict(config_dict: Dict[str, Any]) -> MarketConfig:
    """
    Create MarketConfig from configuration dictionary.

    Args:
        config_dict: Configuration dictionary (from config.yaml)

    Returns:
        MarketConfig instance

    Example:
        >>> config_dict = {
        ...     "mode": "us_only"
        ... }
        >>> config = get_market_config_from_dict(config_dict)
        >>> config.allowed_countries
        {'United States', 'USA', ...}
    """
    mode = config_dict.get("mode", "global")

    if mode == "us_only":
        return US_ONLY_MARKET_CONFIG
    elif mode == "eu_only":
        return EU_ONLY_MARKET_CONFIG
    elif mode == "global":
        return DEFAULT_MARKET_CONFIG
    elif mode == "custom":
        # Build custom config
        custom = config_dict.get("custom", {})

        # Convert lists to sets (None if empty)
        countries = custom.get("allowed_countries", [])
        currencies = custom.get("allowed_currencies", [])
        exchanges = custom.get("allowed_exchanges", [])

        return MarketConfig(
            allowed_countries=set(countries) if countries else None,
            allowed_currencies=set(currencies) if currencies else None,
            allowed_exchanges=set(exchanges) if exchanges else None,
            exclude_etfs=config_dict.get("exclude_etfs", True),
            exclude_mutualfunds=config_dict.get("exclude_mutualfunds", True),
            exclude_crypto=config_dict.get("exclude_crypto", True)
        )
    else:
        # Default to global
        return DEFAULT_MARKET_CONFIG
