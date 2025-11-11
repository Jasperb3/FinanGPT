"""
Query result caching module with LRU eviction and TTL expiration.

This module implements a memory-efficient cache for DuckDB query results,
providing 100-1000x speedup for repeated queries.

Features:
- Time-based expiration (TTL)
- Size-based eviction (LRU when cache full)
- SQL normalization for consistent cache keys
- Thread-safe operations
- Cache statistics and monitoring

Author: FinanGPT Enhancement Plan 3
Created: 2025-11-09
"""

from typing import Optional, Dict, Tuple, Any, Callable, Sequence
from time import time
from functools import wraps
import pandas as pd
import hashlib
import threading
import json
from pathlib import Path
from datetime import datetime, UTC

from src.utils.paths import get_cache_metrics_path


DEFAULT_METRICS_PATH = get_cache_metrics_path()


class QueryCache:
    """
    LRU cache with TTL for query results.

    This cache provides significant performance improvements for repeated
    queries while managing memory usage through LRU eviction.

    Features:
    - Time-based expiration (default: 5 minutes)
    - Size-based eviction (LRU when cache full)
    - SQL-based cache keys (normalized for consistency)
    - Thread-safe for concurrent access

    Example:
        >>> cache = QueryCache(ttl_seconds=300, max_entries=100)
        >>> sql = "SELECT * FROM financials.annual WHERE ticker = 'AAPL' LIMIT 10"
        >>>
        >>> # First query - cache miss
        >>> result = cache.get(sql)
        >>> if result is None:
        ...     result = conn.execute(sql).df()
        ...     cache.set(sql, result)
        >>>
        >>> # Second query - cache hit (100x faster)
        >>> result = cache.get(sql)  # Returns instantly from cache
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 100,
        metrics_path: Optional[str | Path] = None,
    ):
        """
        Initialize query cache.

        Args:
            ttl_seconds: Time-to-live for cached entries (default: 5 minutes)
            max_entries: Maximum number of cached queries (default: 100)
        """
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()  # Thread-safe operations
        self._hits = 0
        self._misses = 0
        self._latency_total = 0.0
        self._latency_samples = 0
        self._metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_metrics()

    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for consistent cache keys.

        Normalization includes:
        - Remove extra whitespace
        - Convert to lowercase
        - Hash for fixed-length key

        Args:
            sql: Raw SQL query string

        Returns:
            MD5 hash of normalized SQL (consistent key)

        Example:
            >>> cache._normalize_sql("SELECT * FROM table")
            'a1b2c3d4e5f6...'
            >>> cache._normalize_sql("  SELECT  *  FROM  table  ")
            'a1b2c3d4e5f6...'  # Same hash (normalized)
        """
        # Remove extra whitespace, convert to lowercase
        normalized = " ".join(sql.lower().split())

        # Hash for fixed-length key (prevents memory issues with long SQL)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _load_metrics(self) -> None:
        if not self._metrics_path.exists():
            return
        try:
            data = json.loads(self._metrics_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return

        self._hits = int(data.get('hits', self._hits))
        self._misses = int(data.get('misses', self._misses))
        self._latency_total = float(data.get('latency_total', self._latency_total))
        self._latency_samples = int(data.get('latency_samples', self._latency_samples))

    def _save_metrics(self) -> None:
        data = {
            'hits': self._hits,
            'misses': self._misses,
            'latency_total': self._latency_total,
            'latency_samples': self._latency_samples,
            'last_updated': datetime.now(UTC).isoformat(),
        }
        try:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_path.write_text(json.dumps(data), encoding='utf-8')
        except OSError:
            pass

    def get(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached result if available and not expired.

        Args:
            sql: SQL query string

        Returns:
            Cached DataFrame or None if not found/expired

        Example:
            >>> result = cache.get("SELECT * FROM financials.annual LIMIT 10")
            >>> if result is None:
            ...     print("Cache miss - executing query")
            ... else:
            ...     print("Cache hit - instant result!")
        """
        with self._lock:
            cache_key = self._normalize_sql(sql)

            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]

                # Check expiration
                if time() - timestamp < self.ttl:
                    # Update LRU access time
                    self._access_times[cache_key] = time()
                    self._hits += 1
                    self._save_metrics()

                    # Return copy to prevent mutation
                    return result.copy()

                # Expired - remove from cache
                del self._cache[cache_key]
                del self._access_times[cache_key]
                self._metadata.pop(cache_key, None)

            self._misses += 1
            self._save_metrics()
            return None

    def set(
        self,
        sql: str,
        result: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache query result with current timestamp.

        If cache is at capacity, evicts the least recently used entry.

        Args:
            sql: SQL query string
            result: Query result DataFrame

        Example:
            >>> result = conn.execute(sql).df()
            >>> cache.set(sql, result)
        """
        with self._lock:
            cache_key = self._normalize_sql(sql)

            # Evict LRU entry if at capacity
            if len(self._cache) >= self.max_entries and cache_key not in self._cache:
                # Find least recently used entry
                lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                del self._cache[lru_key]
                del self._access_times[lru_key]
                self._metadata.pop(lru_key, None)

            # Store result (copy to prevent external mutations)
            self._cache[cache_key] = (result.copy(), time())
            self._access_times[cache_key] = time()
            if metadata is not None:
                self._metadata[cache_key] = metadata
            else:
                self._metadata.pop(cache_key, None)

    def clear(self) -> None:
        """
        Clear all cached entries.

        Example:
            >>> cache.clear()
            >>> print("Cache cleared")
        """
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._metadata.clear()
            self._hits = 0
            self._misses = 0
            self._latency_total = 0.0
            self._latency_samples = 0
            self._save_metrics()

    def record_query_latency(self, seconds: float) -> None:
        """Record observed query latency (seconds)."""

        if seconds is None or seconds < 0:
            return
        with self._lock:
            self._latency_total += seconds
            self._latency_samples += 1
            self._save_metrics()

    def invalidate(
        self,
        sql: Optional[str] = None,
        tickers: Optional[Sequence[str]] = None,
    ) -> int:
        """Invalidate cached queries by SQL text or ticker dependency."""

        with self._lock:
            targets = set()

            if sql:
                targets.add(self._normalize_sql(sql))

            if tickers:
                lookup = {ticker.upper() for ticker in tickers}
                for key, meta in self._metadata.items():
                    deps = [t.upper() for t in meta.get('tickers', [])]
                    if any(t in lookup for t in deps):
                        targets.add(key)

            removed = 0
            for cache_key in targets:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self._access_times.pop(cache_key, None)
                    self._metadata.pop(cache_key, None)
                    removed += 1

            return removed

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics for monitoring.

        Returns:
            Dictionary with entries, hits, misses, hit_rate, utilization

        Example:
            >>> stats = cache.stats()
            >>> print(f"Cache hit rate: {stats['hit_rate_pct']:.1f}%")
            >>> print(f"Utilization: {stats['utilization_pct']:.1f}%")
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            avg_latency_ms = (
                (self._latency_total / self._latency_samples) * 1000
                if self._latency_samples > 0
                else 0
            )

            return {
                'entries': len(self._cache),
                'max_entries': self.max_entries,
                'ttl_seconds': self.ttl,
                'hits': self._hits,
                'misses': self._misses,
                'total_requests': total_requests,
                'hit_rate_pct': hit_rate,
                'utilization_pct': (len(self._cache) / self.max_entries * 100) if self.max_entries > 0 else 0,
                'latency_samples': self._latency_samples,
                'avg_latency_ms': avg_latency_ms,
            }

    def print_stats(self) -> None:
        """
        Print formatted cache statistics.

        Example:
            >>> cache.print_stats()
            📊 Cache Statistics:
              Entries: 42/100 (42.0% full)
              Hits: 156 | Misses: 23 (87.2% hit rate)
              TTL: 300s
        """
        stats = self.stats()

        print(f"\n📊 Cache Statistics:")
        print(f"  Entries: {stats['entries']}/{stats['max_entries']} "
              f"({stats['utilization_pct']:.1f}% full)")
        print(f"  Hits: {stats['hits']} | Misses: {stats['misses']} "
              f"({stats['hit_rate_pct']:.1f}% hit rate)")
        print(f"  TTL: {stats['ttl_seconds']}s")

    def get_entry_metadata(self, sql: str) -> Optional[Dict[str, Any]]:
        """Return metadata associated with a cached SQL query, if present."""

        cache_key = self._normalize_sql(sql)
        with self._lock:
            meta = self._metadata.get(cache_key)
            return dict(meta) if meta else None


def with_cache(cache: QueryCache):
    """
    Decorator to add caching to query functions.

    Args:
        cache: QueryCache instance to use

    Returns:
        Decorated function with caching

    Example:
        >>> query_cache = QueryCache(ttl_seconds=300)
        >>>
        >>> @with_cache(query_cache)
        ... def execute_query(sql: str, conn) -> pd.DataFrame:
        ...     return conn.execute(sql).df()
        >>>
        >>> # First call - cache miss, executes query
        >>> result1 = execute_query("SELECT * FROM table", conn)
        >>>
        >>> # Second call - cache hit, instant result
        >>> result2 = execute_query("SELECT * FROM table", conn)  # 100x faster!
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(sql: str, *args, **kwargs):
            # Try cache first
            cached_result = cache.get(sql)

            if cached_result is not None:
                if kwargs.get('debug'):
                    print("🚀 Cache hit!")
                return cached_result

            # Cache miss - execute query
            if kwargs.get('debug'):
                print("💾 Cache miss - executing query...")

            result = func(sql, *args, **kwargs)

            # Cache result if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                cache.set(sql, result)

            return result

        return wrapper
    return decorator


# Global cache instance (can be configured from config.yaml)
_global_cache: Optional[QueryCache] = None


def get_global_cache(ttl_seconds: int = 300, max_entries: int = 100) -> QueryCache:
    """
    Get or create global cache instance.

    Args:
        ttl_seconds: TTL for cache entries
        max_entries: Maximum cache size

    Returns:
        Global QueryCache instance

    Example:
        >>> cache = get_global_cache(ttl_seconds=600, max_entries=200)
        >>> result = cache.get("SELECT * FROM table")
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = QueryCache(ttl_seconds=ttl_seconds, max_entries=max_entries)

    return _global_cache


def read_persisted_metrics() -> Dict[str, Any]:
    """Read cached metrics from disk without instantiating the cache."""

    if not DEFAULT_METRICS_PATH.exists():
        return {}
    try:
        return json.loads(DEFAULT_METRICS_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
