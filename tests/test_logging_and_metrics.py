from __future__ import annotations

import json
import logging

import pandas as pd

from src.utils.logging import configure_logger
from src.query.cache import QueryCache


def teardown_logger(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_rotating_json_logs(tmp_path):
    logger_name = "test_rotation_logger"
    logger = configure_logger(
        logger_name,
        log_dir=str(tmp_path),
        log_format='json',
        console_output=False,
        max_bytes=256,
        backup_count=2,
    )

    for i in range(200):
        logger.info("log line %s", i)

    log_files = sorted(tmp_path.glob(f"{logger_name}.log*"))
    assert log_files, "expected rotated logs to exist"
    assert len(log_files) <= 3  # base + 2 backups
    teardown_logger(logger_name)


def test_cache_metrics_persist(tmp_path):
    metrics_path = tmp_path / "cache_metrics.json"
    cache = QueryCache(ttl_seconds=60, max_entries=10, metrics_path=metrics_path)

    df = pd.DataFrame({"value": [1]})
    cache.set("SELECT 1", df, metadata={'tickers': ['AAPL']})
    assert cache.get("SELECT 1") is not None
    cache.record_query_latency(0.05)

    stats = cache.stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 0
    assert stats['latency_samples'] == 1

    data = json.loads(metrics_path.read_text())
    assert data['hits'] == 1
    assert data['latency_samples'] == 1
