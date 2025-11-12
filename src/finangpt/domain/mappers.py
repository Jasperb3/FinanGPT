"""Utility helpers for converting raw records to domain objects."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Iterable, Mapping, MutableMapping

from .entities import Company, FinancialStatement, PricePoint, PriceSeries
from .value_objects import DateRange, Money, Ticker

__all__ = [
    "dict_to_company",
    "company_to_dict",
    "dict_to_financial_statement",
    "financial_statement_to_dict",
    "dict_to_price_series",
    "price_series_to_dict",
]


def _dict_to_money(payload: Mapping[str, object]) -> Money:
    return Money(Decimal(str(payload["amount"])), str(payload["currency"]))


def _money_to_dict(money: Money) -> dict[str, object]:
    return {"amount": str(money.amount), "currency": money.currency}


def dict_to_company(payload: Mapping[str, object]) -> Company:
    return Company(
        ticker=Ticker(str(payload["ticker"])),
        name=str(payload["name"]),
        sector=payload.get("sector"),
        industry=payload.get("industry"),
        currency=payload.get("currency"),
    )


def company_to_dict(company: Company) -> dict[str, object]:
    return {
        "ticker": company.ticker.symbol,
        "name": company.name,
        "sector": company.sector,
        "industry": company.industry,
        "currency": company.currency,
    }


def dict_to_financial_statement(payload: Mapping[str, object]) -> FinancialStatement:
    company = dict_to_company(payload["company"])
    period_data = payload["period"]
    period = DateRange(
        start=date.fromisoformat(str(period_data["start"])),
        end=date.fromisoformat(str(period_data["end"])),
    )

    def optional_money(key: str) -> Money | None:
        part = payload.get(key)
        if part is None:
            return None
        return _dict_to_money(part)

    return FinancialStatement(
        company=company,
        period=period,
        revenue=optional_money("revenue"),
        net_income=optional_money("net_income"),
        operating_income=optional_money("operating_income"),
    )


def financial_statement_to_dict(statement: FinancialStatement) -> dict[str, object]:
    payload: dict[str, object] = {
        "company": company_to_dict(statement.company),
        "period": {
            "start": statement.period.start.isoformat(),
            "end": statement.period.end.isoformat(),
        },
    }
    if statement.revenue:
        payload["revenue"] = _money_to_dict(statement.revenue)
    if statement.net_income:
        payload["net_income"] = _money_to_dict(statement.net_income)
    if statement.operating_income:
        payload["operating_income"] = _money_to_dict(statement.operating_income)
    return payload


def dict_to_price_series(payload: Mapping[str, object]) -> PriceSeries:
    ticker = Ticker(str(payload["ticker"]))
    prices_iter: Iterable[PricePoint] = (
        PricePoint(
            date=date.fromisoformat(str(item["date"])),
            close=Decimal(str(item["close"])),
        )
        for item in payload.get("prices", [])
    )
    return PriceSeries.from_iterable(ticker, prices_iter)


def price_series_to_dict(series: PriceSeries) -> dict[str, object]:
    return {
        "ticker": series.ticker.symbol,
        "prices": [
            {"date": point.date.isoformat(), "close": str(point.close)}
            for point in series.prices
        ],
    }
