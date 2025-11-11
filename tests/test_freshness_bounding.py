from __future__ import annotations

from datetime import datetime, UTC, timedelta

import pytest

from src.query_engine import query


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def find_one(self, filter, sort=None):
        return self.docs.get(filter["ticker"])


class FakeDB:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        if name == "ingestion_metadata":
            return self.collection
        raise KeyError


def test_check_data_freshness_timeout(monkeypatch):
    def fake_bpm(func, items, max_workers, timeout):
        return [], [(item, TimeoutError("batch timeout")) for item in items]

    monkeypatch.setattr(query, 'bounded_parallel_map', fake_bpm)
    config = query.FreshnessConfig()
    config.max_batch_size = 2
    config.timeout_ms = 1
    config.max_retries = 0
    monkeypatch.setattr(query, 'FRESHNESS_CFG', config)

    fake_db = FakeDB(FakeCollection({}))
    result = query.check_data_freshness(fake_db, ["AAPL"], threshold_days=1)
    assert result["status"] == "unknown"
    assert result["freshness_info"]["AAPL"] == "unknown"


def test_check_data_freshness_success(monkeypatch):
    now = datetime.now(UTC)
    docs = {
        "AAPL": {"last_fetched": (now - timedelta(days=1)).isoformat()},
        "MSFT": {"last_fetched": (now - timedelta(days=10)).isoformat()},
    }
    fake_db = FakeDB(FakeCollection(docs))
    config = query.FreshnessConfig()
    config.max_batch_size = 10
    config.timeout_ms = 1000
    config.max_retries = 0
    monkeypatch.setattr(query, 'FRESHNESS_CFG', config)

    result = query.check_data_freshness(fake_db, ["AAPL", "MSFT"], threshold_days=7)
    assert result["status"] == "ok"
    assert "AAPL" not in result["stale_tickers"]
    assert "MSFT" in result["stale_tickers"]


def test_build_cache_metadata_marks_status():
    metadata = query.build_cache_metadata(["AAPL"], {}, freshness_status="unknown")
    assert metadata["freshness_status"] == "unknown"
