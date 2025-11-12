"""Domain entities representing companies and their financial data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable, Tuple

from .value_objects import DateRange, Money, Ticker

__all__ = ["Company", "FinancialStatement", "PricePoint", "PriceSeries"]


@dataclass(frozen=True)
class Company:
    ticker: Ticker
    name: str
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Company name cannot be empty")
        object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True)
class FinancialStatement:
    company: Company
    period: DateRange
    revenue: Money | None = None
    net_income: Money | None = None
    operating_income: Money | None = None

    def net_margin(self) -> Decimal | None:
        if not (self.revenue and self.net_income):
            return None
        if self.revenue.amount == 0:
            return None
        if self.revenue.currency != self.net_income.currency:
            return None
        return (self.net_income.amount / self.revenue.amount).quantize(Decimal("0.0001"))

    def operating_margin(self) -> Decimal | None:
        if not (self.revenue and self.operating_income):
            return None
        if self.revenue.amount == 0:
            return None
        if self.revenue.currency != self.operating_income.currency:
            return None
        return (self.operating_income.amount / self.revenue.amount).quantize(Decimal("0.0001"))


@dataclass(frozen=True)
class PricePoint:
    date: date
    close: Decimal


@dataclass(frozen=True)
class PriceSeries:
    ticker: Ticker
    prices: Tuple[PricePoint, ...]

    @classmethod
    def from_iterable(cls, ticker: Ticker, prices: Iterable[PricePoint]) -> "PriceSeries":
        data = tuple(sorted(prices, key=lambda point: point.date))
        return cls(ticker=ticker, prices=data)

    def latest(self) -> PricePoint | None:
        if not self.prices:
            return None
        return self.prices[-1]
