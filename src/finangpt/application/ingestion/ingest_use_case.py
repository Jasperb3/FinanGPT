"""Application use case that wraps the legacy ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv

from finangpt.infrastructure.persistence.mongo_repository import MongoRepository
from src.core.config_loader import load_config
from src.ingestion import ingest as legacy_ingest
from src.query_engine.query import invalidate_cache_for_tickers
from src.utils.logging import configure_logger, log_event

__all__ = ["IngestUseCase", "IngestResult"]


@dataclass(frozen=True)
class IngestResult:
    tickers: tuple[str, ...]
    success: tuple[str, ...]
    failures: tuple[str, ...]


class IngestUseCase:
    def __init__(self, mongo_repository: MongoRepository, logger=None) -> None:
        self._mongo_repository = mongo_repository
        self._logger = logger or configure_logger("ingest")

    def run(
        self,
        tickers: Iterable[str],
        *,
        refresh: bool = False,
        refresh_days: int = 7,
        force: bool = False,
    ) -> IngestResult:
        load_dotenv()
        normalized = self._normalize_tickers(tickers)
        if not normalized:
            raise ValueError("At least one ticker must be provided")

        config = load_config()
        ingestion_cfg = config.get("ingestion", {}) if hasattr(config, "get") else {}
        use_concurrent = bool(ingestion_cfg.get("use_concurrent", False))
        max_workers = ingestion_cfg.get("max_workers", 10)
        worker_timeout = ingestion_cfg.get("worker_timeout", 120)

        mode = "force" if force else "refresh" if refresh else "normal"
        log_event(
            self._logger,
            phase="start",
            mode=mode,
            refresh_days=refresh_days,
            tickers=len(normalized),
        )

        success: list[str] = []
        failures: list[str] = []

        with self._mongo_repository.connect() as client:
            database = self._mongo_repository.get_database(client)
            collections = self._build_collections(database)
            self._ensure_indexes(collections)

            if use_concurrent and legacy_ingest.CONCURRENT_AVAILABLE and len(normalized) > 1:
                def ingest_func(ticker: str, **kwargs):
                    legacy_ingest.ingest_symbol(ticker, collections, self._logger, **kwargs)
                    return 1

                results = legacy_ingest.ingest_batch_concurrent(
                    tickers=normalized,
                    ingest_func=ingest_func,
                    max_workers=max_workers,
                    worker_timeout=worker_timeout,
                    logger=self._logger,
                    refresh_mode=refresh,
                    force_mode=force,
                    refresh_days=refresh_days,
                )
                legacy_ingest.print_ingestion_summary(results)
                for ticker, outcome in results.items():
                    if getattr(outcome, "status", None) == "success":
                        success.append(ticker)
                    else:
                        failures.append(ticker)
            else:
                for ticker in normalized:
                    try:
                        legacy_ingest.ingest_symbol(
                            ticker,
                            collections,
                            self._logger,
                            refresh_mode=refresh,
                            force_mode=force,
                            refresh_days=refresh_days,
                        )
                        success.append(ticker)
                    except Exception:
                        failures.append(ticker)
                        continue

        if success:
            invalidate_cache_for_tickers(success)

        log_event(
            self._logger,
            phase="complete",
            success=len(success),
            failures=len(failures),
        )

        return IngestResult(tuple(normalized), tuple(success), tuple(failures))

    def _normalize_tickers(self, tickers: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for ticker in tickers:
            symbol = (ticker or "").strip().upper()
            if not symbol:
                continue
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
        max_tickers = legacy_ingest.get_max_tickers_per_run()
        if len(ordered) > max_tickers:
            ordered = ordered[:max_tickers]
        return ordered

    def _build_collections(self, database) -> dict[str, any]:  # pragma: no cover - thin wrapper
        return {
            "annual": database["raw_annual"],
            "quarterly": database["raw_quarterly"],
            "prices": database["stock_prices_daily"],
            "dividends": database["dividends_history"],
            "splits": database["splits_history"],
            "company_metadata": database["company_metadata"],
            "metadata": database["ingestion_metadata"],
            "earnings_history": database["earnings_history"],
            "earnings_calendar": database["earnings_calendar"],
            "analyst_recommendations": database["analyst_recommendations"],
            "price_targets": database["price_targets"],
            "analyst_consensus": database["analyst_consensus"],
            "growth_estimates": database["growth_estimates"],
        }

    def _ensure_indexes(self, collections: dict) -> None:
        legacy_ingest.ensure_indexes(collections["annual"])
        legacy_ingest.ensure_indexes(collections["quarterly"])
        legacy_ingest.ensure_indexes(collections["prices"])
        legacy_ingest.ensure_indexes(collections["dividends"])
        legacy_ingest.ensure_indexes(collections["splits"])
        legacy_ingest.ensure_indexes(collections["earnings_history"])
        legacy_ingest.ensure_indexes(collections["earnings_calendar"])
        legacy_ingest.ensure_indexes(collections["analyst_recommendations"])
        legacy_ingest.ensure_indexes(collections["price_targets"])
        legacy_ingest.ensure_indexes(collections["analyst_consensus"])
        legacy_ingest.ensure_indexes(collections["growth_estimates"])
        collections["company_metadata"].create_index("ticker", unique=True)
        collections["metadata"].create_index([("ticker", 1), ("data_type", 1)], unique=True)
