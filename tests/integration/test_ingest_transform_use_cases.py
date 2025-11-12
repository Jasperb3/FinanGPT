from __future__ import annotations

from datetime import datetime

import duckdb
import mongomock
import pandas as pd
import pytest

from finangpt.application.ingestion.ingest_use_case import IngestUseCase
from finangpt.application.transformation.transform_use_case import TransformUseCase
from finangpt.infrastructure.persistence.duckdb_repository import DuckDBRepository
from finangpt.infrastructure.persistence.mongo_repository import MongoRepository


class PersistentMongoClient(mongomock.MongoClient):
    def close(self):  # pragma: no cover - keep in-memory data alive across contexts
        pass


def _seed_document(collection, doc):
    collection.update_one({"ticker": doc["ticker"], "date": doc.get("date")}, {"$set": doc}, upsert=True)


def test_ingest_transform_use_cases(tmp_path, monkeypatch):
    shared_client = PersistentMongoClient()
    mongo_repo = MongoRepository(uri="mongodb://localhost/testdb", client_factory=lambda uri: shared_client)
    duckdb_path = tmp_path / "sample.duckdb"
    duck_repo = DuckDBRepository(path=duckdb_path)

    def fake_ingest_symbol(ticker, collections, logger, refresh_mode=False, force_mode=False, refresh_days=7):
        annual_payload = {
            "ticker": ticker,
            "date": "2023-12-31",
            "income_statement": {
                "totalRevenue": 1000,
                "netIncome": 120,
                "grossProfit": 400,
                "ebitda": 220,
            },
            "balance_sheet": {
                "shareholderEquity": 500,
                "totalAssets": 1500,
            },
            "cash_flow": {
                "operatingCashFlow": 200,
                "freeCashFlow": 150,
            },
        }
        collections["annual"].update_one(
            {"ticker": ticker, "date": annual_payload["date"]},
            {"$set": {"ticker": ticker, "date": annual_payload["date"], "payload": annual_payload}},
            upsert=True,
        )
        quarterly_payload = {
            **annual_payload,
            "date": "2024-03-31",
        }
        collections["quarterly"].update_one(
            {"ticker": ticker, "date": quarterly_payload["date"]},
            {"$set": {"ticker": ticker, "date": quarterly_payload["date"], "payload": quarterly_payload}},
            upsert=True,
        )
        price_doc = {
            "ticker": ticker,
            "date": "2024-01-02",
            "open": 10.0,
            "high": 12.0,
            "low": 9.5,
            "close": 11.0,
            "adj_close": 11.0,
            "volume": 1000,
        }
        _seed_document(collections["prices"], price_doc)
        _seed_document(collections["dividends"], {"ticker": ticker, "date": "2023-10-01", "amount": 0.5})
        _seed_document(collections["splits"], {"ticker": ticker, "date": "2022-06-01", "ratio": 2.0})
        collections["company_metadata"].update_one(
            {"ticker": ticker},
            {"$set": {"ticker": ticker, "metadata": {"longName": f"{ticker} Inc.", "sector": "Tech"}}},
            upsert=True,
        )
        _seed_document(
            collections["earnings_history"],
            {
                "ticker": ticker,
                "date": "2023-12-31",
                "fiscal_period": "Q4",
                "eps_estimate": 1.0,
                "eps_actual": 1.2,
                "revenue_estimate": 900,
                "revenue_actual": 1000,
            },
        )
        _seed_document(
            collections["earnings_calendar"],
            {
                "ticker": ticker,
                "earnings_date": "2024-04-30",
                "period_ending": "Q1",
                "estimate": 1.3,
            },
        )
        _seed_document(
            collections["analyst_recommendations"],
            {
                "ticker": ticker,
                "date": "2024-01-15",
                "firm": "AnalystCo",
                "from_grade": "Hold",
                "to_grade": "Buy",
                "action": "upgrade",
            },
        )
        _seed_document(
            collections["price_targets"],
            {
                "ticker": ticker,
                "date": "2024-01-10",
                "current_price": 11.0,
                "target_low": 10.0,
                "target_mean": 13.0,
                "target_high": 15.0,
                "num_analysts": 5,
            },
        )
        _seed_document(
            collections["analyst_consensus"],
            {
                "ticker": ticker,
                "date": "2024-01-10",
                "strong_buy": 1,
                "buy": 2,
                "hold": 1,
                "sell": 0,
                "strong_sell": 0,
            },
        )
        _seed_document(
            collections["growth_estimates"],
            {
                "ticker": ticker,
                "date": "2024-01-10",
                "current_qtr_growth": 0.1,
                "next_qtr_growth": 0.12,
                "current_year_growth": 0.15,
                "next_year_growth": 0.18,
                "next_5yr_growth": 0.2,
            },
        )

    monkeypatch.setattr("src.ingestion.ingest.ingest_symbol", fake_ingest_symbol)
    monkeypatch.setattr(
        "finangpt.application.ingestion.ingest_use_case.invalidate_cache_for_tickers",
        lambda tickers: None,
    )

    monkeypatch.setattr(
        "src.transformation.transform._load_transform_settings",
        lambda: {
            "chunk_size": 1000,
            "run_integrity_checks": False,
            "integrity_tolerance_pct": 1.0,
            "enable_streaming_requested": False,
        },
    )

    ingest_use_case = IngestUseCase(mongo_repo)
    result = ingest_use_case.run(["aapl"], refresh=False)
    assert result.success == ("AAPL",)

    transform_use_case = TransformUseCase(mongo_repo, duck_repo)
    transform_result = transform_use_case.run()
    assert "annual_rows" in transform_result.metrics

    conn = duckdb.connect(str(duckdb_path))
    rows = conn.execute("SELECT ticker, totalRevenue FROM financials.annual").fetchall()
    conn.close()
    assert rows and rows[0][0] == "AAPL"
