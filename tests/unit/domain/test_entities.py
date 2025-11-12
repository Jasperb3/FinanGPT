from __future__ import annotations

from datetime import date
from decimal import Decimal

from finangpt.domain.entities import Company, FinancialStatement, PricePoint, PriceSeries
from finangpt.domain.value_objects import DateRange, Money, Ticker


def make_company() -> Company:
    return Company(ticker=Ticker("AAPL"), name="Apple")


def test_financial_statement_margins_require_consistent_currency():
    statement = FinancialStatement(
        company=make_company(),
        period=DateRange(date(2024, 1, 1), date(2024, 3, 31)),
        revenue=Money(Decimal("1000"), "USD"),
        net_income=Money(Decimal("200"), "USD"),
        operating_income=Money(Decimal("150"), "USD"),
    )

    assert statement.net_margin() == Decimal("0.2000")
    assert statement.operating_margin() == Decimal("0.1500")


def test_price_series_orders_points_and_returns_latest():
    series = PriceSeries.from_iterable(
        ticker=Ticker("AAPL"),
        prices=[
            PricePoint(date=date(2024, 1, 10), close=Decimal("180")),
            PricePoint(date=date(2024, 1, 9), close=Decimal("179")),
        ],
    )

    latest = series.latest()
    assert latest is not None
    assert latest.date == date(2024, 1, 10)
    assert latest.close == Decimal("180")
