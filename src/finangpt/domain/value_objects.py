"""Core immutable value objects used across the FinanGPT domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

__all__ = ["Money", "DateRange", "Ticker", "CurrencyMismatchError"]


class CurrencyMismatchError(ValueError):
    """Raised when arithmetic is attempted on money values with different currencies."""


@dataclass(frozen=True)
class Ticker:
    """Immutable ticker symbol with basic validation."""

    symbol: str

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Ticker symbol cannot be empty")
        normalized = self.symbol.strip().upper()
        if len(normalized) > 10:
            raise ValueError("Ticker symbol is too long")
        object.__setattr__(self, "symbol", normalized)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol


@dataclass(frozen=True)
class Money:
    """Simple money representation used for financial metrics."""

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise TypeError("amount must be a Decimal instance")
        if not self.currency or not self.currency.strip():
            raise ValueError("currency code is required")
        object.__setattr__(self, "currency", self.currency.upper())

    def _ensure_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError("Cannot operate on different currencies")

    def convert_to(self, target_currency: str, fx_rate: Decimal) -> "Money":
        if not isinstance(fx_rate, Decimal):
            raise TypeError("fx_rate must be Decimal")
        if fx_rate <= Decimal("0"):
            raise ValueError("fx_rate must be positive")
        converted_amount = (self.amount * fx_rate).quantize(Decimal("0.0001"))
        return Money(converted_amount, target_currency)

    def __add__(self, other: "Money") -> "Money":  # pragma: no cover - simple wrapper
        self._ensure_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":  # pragma: no cover - simple wrapper
        self._ensure_same_currency(other)
        return Money(self.amount - other.amount, self.currency)


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("start date must be on or before end date")

    def contains(self, moment: date) -> bool:
        return self.start <= moment <= self.end
