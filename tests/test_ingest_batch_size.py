from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingestion import ingest


def test_read_tickers_respects_configured_cap(monkeypatch):
    args = SimpleNamespace(
        tickers="AAPL,MSFT,GOOGL,AMZN",
        tickers_file=None,
    )

    logger = MagicMock()

    monkeypatch.setattr(ingest, "get_max_tickers_per_run", lambda: 2)

    events = []

    def fake_log_event(logger_obj, **kwargs):
        events.append(kwargs)

    monkeypatch.setattr(ingest, "log_event", fake_log_event)

    tickers = ingest.read_tickers(args, logger)

    assert tickers == ["AAPL", "MSFT"]
    assert any("first 2" in evt.get("message", "") for evt in events)
