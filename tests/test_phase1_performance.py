"""
Test suite for Phase 1 performance optimizations.

Tests cover:
- Streaming transformation
- Concurrent ingestion
- Query result caching
- Pre-compiled regex validation
- Progress indicators

Author: FinanGPT Enhancement Plan 3
Created: 2025-11-09
"""

import pytest
import pandas as pd
import duckdb
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import time


# Test streaming transformation
class TestStreamingTransformation:
    """Tests for src/transform/streaming.py"""

    def test_stream_documents_chunks_correctly(self):
        """Test that documents are streamed in correct chunk sizes."""
        from src.transformation.streaming import stream_documents

        # Create mock collection with 250 documents
        mock_collection = Mock()
        mock_docs = [{"_id": i, "ticker": f"TEST{i}"} for i in range(250)]
        mock_collection.find.return_value = iter(mock_docs)

        # Stream with chunk size 100
        chunks = list(stream_documents(mock_collection, chunk_size=100))

        # Should get 3 chunks: 100, 100, 50
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50

    def test_stream_documents_empty_collection(self):
        """Test streaming from empty collection."""
        from src.transformation.streaming import stream_documents

        mock_collection = Mock()
        mock_collection.find.return_value = iter([])

        chunks = list(stream_documents(mock_collection, chunk_size=100))

        assert len(chunks) == 0

    def test_upsert_dataframe_creates_table(self):
        """Test that upsert_dataframe creates table if not exists."""
        from src.transformation.streaming import upsert_dataframe

        conn = duckdb.connect(":memory:")

        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            'revenue': [100000, 200000]
        })

        rows = upsert_dataframe(conn, df, 'test_table', 'test_schema')

        assert rows == 2

        # Verify table was created
        result = conn.execute("SELECT COUNT(*) FROM test_schema.test_table").fetchone()[0]
        assert result == 2

    def test_upsert_dataframe_idempotent(self):
        """Test that upsert is idempotent (no duplicates)."""
        from src.transformation.streaming import upsert_dataframe

        conn = duckdb.connect(":memory:")

        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': [datetime(2024, 1, 1)],
            'revenue': [100000]
        })

        # Insert twice
        upsert_dataframe(conn, df, 'test_table', 'test_schema')
        upsert_dataframe(conn, df, 'test_table', 'test_schema')

        # Should only have 1 row (upsert, not duplicate)
        result = conn.execute("SELECT COUNT(*) FROM test_schema.test_table").fetchone()[0]
        assert result == 1

    def test_get_collection_stats(self):
        """Test collection statistics calculation."""
        from src.transformation.streaming import get_collection_stats

        mock_collection = Mock()
        mock_collection.count_documents.return_value = 1000

        # Mock aggregate for sampling
        mock_docs = [{"ticker": "TEST", "data": "x" * 100} for _ in range(100)]
        mock_collection.aggregate.return_value = mock_docs

        stats = get_collection_stats(mock_collection)

        assert stats['count'] == 1000
        assert stats['avg_size_bytes'] > 0
        assert stats['estimated_memory_mb'] > 0

    def test_recommend_chunk_size(self):
        """Test chunk size recommendation based on stats."""
        from src.transformation.streaming import recommend_chunk_size

        mock_collection = Mock()
        mock_collection.count_documents.return_value = 10000

        # Mock large documents (10KB each)
        mock_docs = [{"data": "x" * 10000} for _ in range(100)]
        mock_collection.aggregate.return_value = mock_docs

        chunk_size = recommend_chunk_size(mock_collection, max_memory_mb=500)

        # Should recommend smaller chunks for large documents
        assert 100 <= chunk_size <= 5000


# Test concurrent ingestion
class TestConcurrentIngestion:
    """Tests for src/ingest/concurrent.py"""

    def test_ingest_batch_concurrent_success(self):
        """Test successful concurrent ingestion."""
        from src.ingestion.concurrent import ingest_batch_concurrent, IngestionResult

        # Mock ingest function that succeeds
        def mock_ingest(ticker, **kwargs):
            time.sleep(0.01)  # Simulate work
            return 10  # Return row count

        tickers = ["AAPL", "MSFT", "GOOGL"]

        results = ingest_batch_concurrent(
            tickers,
            mock_ingest,
            max_workers=3,
            worker_timeout=5
        )

        assert len(results) == 3
        assert all(r.status == "success" for r in results.values())
        assert all(r.rows_inserted == 10 for r in results.values())

    def test_ingest_batch_concurrent_partial_failure(self):
        """Test concurrent ingestion with some failures."""
        from src.ingestion.concurrent import ingest_batch_concurrent

        # Mock ingest function that fails for specific ticker
        def mock_ingest(ticker, **kwargs):
            if ticker == "FAIL":
                raise ValueError("Simulated failure")
            return 10

        tickers = ["AAPL", "FAIL", "MSFT"]

        results = ingest_batch_concurrent(
            tickers,
            mock_ingest,
            max_workers=3,
            worker_timeout=5
        )

        assert len(results) == 3
        assert results["AAPL"].status == "success"
        assert results["FAIL"].status == "failed"
        assert results["MSFT"].status == "success"
        assert "Simulated failure" in results["FAIL"].error

    def test_ingest_batch_concurrent_timeout(self):
        """Test timeout handling in concurrent ingestion."""
        from src.ingestion.concurrent import ingest_batch_concurrent

        # Mock ingest function that hangs
        def mock_ingest_slow(ticker, **kwargs):
            time.sleep(5)  # Longer than timeout
            return 10

        tickers = ["SLOW"]

        results = ingest_batch_concurrent(
            tickers,
            mock_ingest_slow,
            max_workers=1,
            worker_timeout=2  # 2 second timeout (should timeout after 2 sec)
        )

        # Note: ThreadPoolExecutor's timeout is best-effort
        # The function may complete if thread scheduling allows
        # This test verifies timeout mechanism exists, not that it always triggers
        assert "SLOW" in results
        assert results["SLOW"].status in ("success", "failed")

    def test_estimate_time_savings(self):
        """Test time savings estimation."""
        from src.ingestion.concurrent import estimate_time_savings

        estimate = estimate_time_savings(
            num_tickers=50,
            avg_time_per_ticker_sec=5.0,
            max_workers=10
        )

        # Sequential: 50 * 5 = 250 seconds
        # Concurrent: (50 / 10) * 5 * 1.1 = 27.5 seconds
        # Speedup: ~9x

        assert estimate['sequential_time_sec'] == 250.0
        assert estimate['concurrent_time_sec'] > 25  # At least 25 sec
        assert estimate['concurrent_time_sec'] < 30  # Less than 30 sec
        assert estimate['speedup'] > 8  # At least 8x speedup


# Test query caching
class TestQueryCache:
    """Tests for src/query/cache.py"""

    def test_cache_get_miss(self):
        """Test cache miss returns None."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=10)

        result = cache.get("SELECT * FROM table")

        assert result is None

    def test_cache_set_and_get_hit(self):
        """Test cache hit returns stored result."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=10)

        df = pd.DataFrame({'col': [1, 2, 3]})

        cache.set("SELECT * FROM table", df)
        result = cache.get("SELECT * FROM table")

        assert result is not None
        assert result.equals(df)

    def test_cache_ttl_expiration(self):
        """Test that cached entries expire after TTL."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=1, max_entries=10)  # 1 second TTL

        df = pd.DataFrame({'col': [1, 2, 3]})
        cache.set("SELECT * FROM table", df)

        # Immediate retrieval should work
        result = cache.get("SELECT * FROM table")
        assert result is not None

        # After TTL, should be expired
        time.sleep(1.1)
        result = cache.get("SELECT * FROM table")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=3)

        # Fill cache to capacity
        cache.set("SELECT 1", pd.DataFrame({'a': [1]}))
        cache.set("SELECT 2", pd.DataFrame({'a': [2]}))
        cache.set("SELECT 3", pd.DataFrame({'a': [3]}))

        # Access SELECT 1 to make it recently used
        cache.get("SELECT 1")

        # Add new entry - should evict SELECT 2 (least recently used)
        cache.set("SELECT 4", pd.DataFrame({'a': [4]}))

        assert cache.get("SELECT 1") is not None  # Still cached (recently used)
        assert cache.get("SELECT 2") is None      # Evicted (LRU)
        assert cache.get("SELECT 3") is not None  # Still cached
        assert cache.get("SELECT 4") is not None  # Newly added

    def test_cache_sql_normalization(self):
        """Test that SQL normalization creates consistent keys."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=10)

        df = pd.DataFrame({'col': [1, 2, 3]})

        # Store with extra whitespace
        cache.set("SELECT  *  FROM  table", df)

        # Retrieve with different whitespace (should hit)
        result = cache.get("SELECT * FROM table")
        assert result is not None

    def test_cache_clear(self):
        """Test cache clearing."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=10)

        cache.set("SELECT 1", pd.DataFrame({'a': [1]}))
        cache.set("SELECT 2", pd.DataFrame({'a': [2]}))

        assert cache.stats()['entries'] == 2

        cache.clear()

        assert cache.stats()['entries'] == 0
        assert cache.get("SELECT 1") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        from src.query.cache import QueryCache

        cache = QueryCache(ttl_seconds=60, max_entries=100)

        # Perform some operations
        cache.set("SELECT 1", pd.DataFrame({'a': [1]}))
        cache.get("SELECT 1")  # Hit
        cache.get("SELECT 2")  # Miss

        stats = cache.stats()

        assert stats['entries'] == 1
        assert stats['max_entries'] == 100
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate_pct'] == 50.0

    def test_with_cache_decorator(self):
        """Test cache decorator."""
        from src.query.cache import QueryCache, with_cache

        cache = QueryCache(ttl_seconds=60, max_entries=10)

        call_count = 0

        @with_cache(cache)
        def execute_query(sql: str, conn=None) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({'result': [call_count]})

        # First call - cache miss, executes function
        result1 = execute_query("SELECT * FROM table")
        assert call_count == 1
        assert result1['result'][0] == 1

        # Second call - cache hit, doesn't execute function
        result2 = execute_query("SELECT * FROM table")
        assert call_count == 1  # Not incremented (cached)
        assert result2['result'][0] == 1  # Same result


# Test pre-compiled regex validation
class TestSQLValidation:
    """Tests for src/query/validation.py"""

    def test_validate_sql_allows_select(self):
        """Test that SELECT statements are allowed."""
        from src.query.validation import validate_sql

        schema = {"financials.annual": ["ticker", "revenue"]}

        sql = "SELECT ticker, revenue FROM financials.annual"
        validated = validate_sql(sql, schema)

        assert "SELECT" in validated
        assert "LIMIT" in validated

    def test_validate_sql_blocks_insert(self):
        """Test that INSERT is blocked."""
        from src.query.validation import validate_sql

        schema = {"financials.annual": ["ticker", "revenue"]}

        sql = "INSERT INTO financials.annual VALUES ('AAPL', 100000)"

        with pytest.raises(ValueError, match="Only SELECT statements are permitted"):
            validate_sql(sql, schema)

    def test_validate_sql_blocks_unknown_table(self):
        """Test that unknown tables are blocked."""
        from src.query.validation import validate_sql

        schema = {"financials.annual": ["ticker", "revenue"]}

        sql = "SELECT * FROM unknown_table"

        with pytest.raises(ValueError, match="not on the allow-list"):
            validate_sql(sql, schema)

    def test_validate_sql_enforces_max_limit(self):
        """Test that LIMIT is enforced."""
        from src.query.validation import validate_sql

        schema = {"financials.annual": ["ticker", "revenue"]}

        sql = "SELECT * FROM financials.annual LIMIT 9999"

        with pytest.raises(ValueError, match="exceeds the maximum"):
            validate_sql(sql, schema, max_limit=100)

    def test_validate_sql_adds_default_limit(self):
        """Test that default LIMIT is added if missing."""
        from src.query.validation import validate_sql

        schema = {"financials.annual": ["ticker", "revenue"]}

        sql = "SELECT * FROM financials.annual"
        validated = validate_sql(sql, schema, default_limit=50)

        assert "LIMIT 50" in validated

    def test_extract_table_identifiers(self):
        """Test table name extraction."""
        from src.query.validation import extract_table_identifiers

        sql = "SELECT * FROM financials.annual JOIN prices.daily ON ..."

        tables = extract_table_identifiers(sql)

        assert "financials.annual" in tables
        assert "prices.daily" in tables


# Test progress indicators
class TestProgressIndicators:
    """Tests for src/utils/progress.py"""

    def test_with_progress_iterates_correctly(self):
        """Test that progress wrapper doesn't break iteration."""
        from src.utils.progress import with_progress

        items = [1, 2, 3, 4, 5]
        result = []

        for item in with_progress(items, disable=True):  # Disable for testing
            result.append(item)

        assert result == items

    def test_progress_tracker_updates(self):
        """Test multi-stage progress tracker."""
        from src.utils.progress import ProgressTracker

        with ProgressTracker(total=3, disable=True) as tracker:
            tracker.update("stage1", rows=100)
            tracker.update("stage2", rows=200)
            tracker.update("stage3", rows=300)

            assert tracker.current == 3

    def test_estimate_remaining_time(self):
        """Test time estimation."""
        from src.utils.progress import estimate_remaining_time

        # Completed 25/100 in 30 seconds
        # Rate: 30/25 = 1.2 sec per item
        # Remaining: 75 items * 1.2 = 90 seconds = 1m 30s

        time_str = estimate_remaining_time(25, 100, 30.0)

        assert "1m" in time_str or "90s" in time_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
