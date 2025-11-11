import pandas as pd

from src.query.cache import QueryCache


def test_invalidate_by_ticker_and_metadata():
    cache = QueryCache(ttl_seconds=60, max_entries=10)
    sql = "SELECT * FROM financials.annual WHERE ticker = 'AAPL'"
    df = pd.DataFrame({"ticker": ["AAPL"], "value": [1]})
    metadata = {
        "tickers": ["AAPL"],
        "last_ingest_ts": {"AAPL": "2025-01-01T00:00:00Z"},
    }

    cache.set(sql, df, metadata=metadata)
    assert cache.get(sql) is not None

    stored_metadata = cache.get_entry_metadata(sql)
    assert stored_metadata is not None
    assert stored_metadata["tickers"] == ["AAPL"]

    # Invalidate using a different ticker - should remain
    cache.invalidate(tickers=["MSFT"])
    assert cache.get(sql) is not None

    # Invalidate using matching ticker - should evict
    cache.invalidate(tickers=["AAPL"])
    assert cache.get(sql) is None


def test_invalidate_by_sql():
    cache = QueryCache(ttl_seconds=60, max_entries=10)
    sql = "SELECT 1"
    df = pd.DataFrame({"x": [1]})
    cache.set(sql, df, metadata={"tickers": [], "last_ingest_ts": "now"})

    assert cache.get(sql) is not None
    cache.invalidate(sql=sql)
    assert cache.get(sql) is None
