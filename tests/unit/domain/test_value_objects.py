from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finangpt.domain.value_objects import (
    CurrencyMismatchError,
    DateRange,
    Money,
    Ticker,
)


def test_money_add_requires_matching_currency():
    usd = Money(Decimal("100"), "usd")
    eur = Money(Decimal("50"), "eur")

    with pytest.raises(CurrencyMismatchError):
        _ = usd + eur  # type: ignore[operator]


def test_money_convert_to_target_currency():
    usd = Money(Decimal("100"), "USD")
    eur = usd.convert_to("EUR", Decimal("0.9"))

    assert eur.amount == Decimal("90.0000")
    assert eur.currency == "EUR"


def test_date_range_rejects_invalid_order():
    with pytest.raises(ValueError):
        DateRange(start=date(2024, 1, 10), end=date(2024, 1, 1))


def test_ticker_normalizes_and_limits_length():
    ticker = Ticker(" aapl ")
    assert str(ticker) == "AAPL"

    with pytest.raises(ValueError):
        Ticker("X" * 11)
