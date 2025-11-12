"""Execute query plan steps against DuckDB with basic caching."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from finangpt.application.analysis.schemas import QueryStep

__all__ = ["DataRetriever"]


class DataRetriever:
    def __init__(
        self,
        repository,
        cache_repository,
        config,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache_repository
        self._ttl = getattr(config, "cache_ttl_seconds", 300)
        self._clock = clock or time.monotonic

    def execute_step(self, step: QueryStep, dependencies: dict[int, pd.DataFrame]) -> pd.DataFrame:
        cache_key = step.sql_query.strip()
        now = self._clock()
        cached = self._cache.get(cache_key)
        if cached and now - cached["timestamp"] < self._ttl:
            return cached["data"].copy()

        frame = self._repository.execute(step.sql_query)
        self._cache.set(cache_key, {"timestamp": now, "data": frame})
        return frame.copy()
