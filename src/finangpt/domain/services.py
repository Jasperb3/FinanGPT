"""Domain services encapsulating reusable financial calculations."""

from __future__ import annotations

from decimal import Decimal

from .value_objects import CurrencyMismatchError, Money

__all__ = ["FinancialAnalyzer"]


class FinancialAnalyzer:
    """Pure calculation helpers. Stateless by design for easy testing."""

    @staticmethod
    def growth_rate(current: Money, previous: Money) -> Decimal:
        """Return percentage growth (0.10 == 10%)."""

        FinancialAnalyzer._ensure_same_currency(current, previous)
        if previous.amount == 0:
            raise ZeroDivisionError("previous amount cannot be zero for growth calculation")
        raw = (current.amount - previous.amount) / previous.amount
        return raw.quantize(Decimal("0.0001"))

    @staticmethod
    def margin(numerator: Money, denominator: Money) -> Decimal:
        """Return numerator / denominator as a decimal ratio."""

        FinancialAnalyzer._ensure_same_currency(numerator, denominator)
        if denominator.amount == 0:
            raise ZeroDivisionError("denominator cannot be zero for margin calculation")
        return (numerator.amount / denominator.amount).quantize(Decimal("0.0001"))

    @staticmethod
    def _ensure_same_currency(left: Money, right: Money) -> None:
        if left.currency != right.currency:
            raise CurrencyMismatchError("Money values must share currency")
