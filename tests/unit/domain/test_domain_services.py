from __future__ import annotations

from decimal import Decimal

import pytest

from finangpt.domain.services import FinancialAnalyzer
from finangpt.domain.value_objects import CurrencyMismatchError, Money


def test_growth_rate_computes_decimal_ratio():
    current = Money(Decimal("120"), "USD")
    previous = Money(Decimal("100"), "USD")

    growth = FinancialAnalyzer.growth_rate(current, previous)
    assert growth == Decimal("0.2000")


def test_growth_rate_rejects_currency_mismatch():
    with pytest.raises(CurrencyMismatchError):
        FinancialAnalyzer.growth_rate(
            Money(Decimal("120"), "USD"),
            Money(Decimal("100"), "EUR"),
        )


def test_margin_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        FinancialAnalyzer.margin(
            Money(Decimal("10"), "USD"),
            Money(Decimal("0"), "USD"),
        )
