"""Application use case that wraps the legacy transformation workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from finangpt.infrastructure.persistence.duckdb_repository import DuckDBRepository
from finangpt.infrastructure.persistence.mongo_repository import MongoRepository
from src.transformation import transform as legacy_transform
from src.utils.logging import configure_logger, log_event

__all__ = ["TransformUseCase", "TransformResult"]


@dataclass(frozen=True)
class TransformResult:
    metrics: dict[str, int]


class TransformUseCase:
    def __init__(
        self,
        mongo_repository: MongoRepository,
        duckdb_repository: DuckDBRepository,
        logger=None,
    ) -> None:
        self._mongo_repository = mongo_repository
        self._duckdb_repository = duckdb_repository
        self._logger = logger or configure_logger("transform")

    def run(self) -> TransformResult:
        load_dotenv()
        settings = legacy_transform._load_transform_settings()
        metrics: dict[str, int] = {}

        enable_streaming = settings["enable_streaming_requested"] and legacy_transform.STREAMING_AVAILABLE
        if settings["enable_streaming_requested"] and not enable_streaming:
            self._logger.warning("Streaming requested but unavailable; falling back to in-memory mode")

        conn = self._duckdb_repository.connect()
        try:
            metadata_frame = pd.DataFrame()
            earnings_history_frame = pd.DataFrame()
            earnings_calendar_frame = pd.DataFrame()
            analyst_recs_frame = pd.DataFrame()
            price_targets_frame = pd.DataFrame()
            analyst_consensus_frame = pd.DataFrame()
            growth_estimates_frame = pd.DataFrame()

            with self._mongo_repository.connect() as client:
                database = self._mongo_repository.get_database(client)

                metrics["annual_rows"] = legacy_transform.process_upsertable_collection(
                    database["raw_annual"],
                    conn,
                    schema="financials",
                    table_name="annual",
                    prepare_func=legacy_transform.prepare_dataframe,
                    enable_streaming=enable_streaming,
                    chunk_size=settings["chunk_size"],
                    logger=self._logger,
                    run_integrity_checks=settings["run_integrity_checks"],
                    integrity_tolerance_pct=settings["integrity_tolerance_pct"],
                    dataset_label="annual",
                    key_columns=["ticker", "date"],
                )
                log_event(self._logger, phase="transform.annual", rows=metrics["annual_rows"])

                metrics["quarterly_rows"] = legacy_transform.process_upsertable_collection(
                    database["raw_quarterly"],
                    conn,
                    schema="financials",
                    table_name="quarterly",
                    prepare_func=legacy_transform.prepare_dataframe,
                    enable_streaming=enable_streaming,
                    chunk_size=settings["chunk_size"],
                    logger=self._logger,
                    run_integrity_checks=settings["run_integrity_checks"],
                    integrity_tolerance_pct=settings["integrity_tolerance_pct"],
                    dataset_label="quarterly",
                    key_columns=["ticker", "date"],
                )
                log_event(self._logger, phase="transform.quarterly", rows=metrics["quarterly_rows"])

                metrics["prices_rows"] = legacy_transform.process_upsertable_collection(
                    database["stock_prices_daily"],
                    conn,
                    schema="prices",
                    table_name="daily",
                    prepare_func=legacy_transform.prepare_prices_dataframe,
                    enable_streaming=enable_streaming,
                    chunk_size=settings["chunk_size"],
                    logger=self._logger,
                    run_integrity_checks=settings["run_integrity_checks"],
                    integrity_tolerance_pct=settings["integrity_tolerance_pct"],
                    dataset_label="prices",
                    key_columns=["ticker", "date"],
                )
                log_event(self._logger, phase="transform.prices", rows=metrics["prices_rows"])

                metrics["dividends_rows"] = legacy_transform.process_upsertable_collection(
                    database["dividends_history"],
                    conn,
                    schema="dividends",
                    table_name="history",
                    prepare_func=legacy_transform.prepare_dividends_dataframe,
                    enable_streaming=enable_streaming,
                    chunk_size=settings["chunk_size"],
                    logger=self._logger,
                    run_integrity_checks=settings["run_integrity_checks"],
                    integrity_tolerance_pct=settings["integrity_tolerance_pct"],
                    dataset_label="dividends",
                    key_columns=["ticker", "date"],
                )
                log_event(self._logger, phase="transform.dividends", rows=metrics["dividends_rows"])

                metrics["splits_rows"] = legacy_transform.process_upsertable_collection(
                    database["splits_history"],
                    conn,
                    schema="splits",
                    table_name="history",
                    prepare_func=legacy_transform.prepare_splits_dataframe,
                    enable_streaming=enable_streaming,
                    chunk_size=settings["chunk_size"],
                    logger=self._logger,
                    run_integrity_checks=settings["run_integrity_checks"],
                    integrity_tolerance_pct=settings["integrity_tolerance_pct"],
                    dataset_label="splits",
                    key_columns=["ticker", "date"],
                )
                log_event(self._logger, phase="transform.splits", rows=metrics["splits_rows"])

                metadata_frame = legacy_transform.prepare_metadata_dataframe(legacy_transform.fetch_documents(database["company_metadata"]))
                earnings_history_frame = legacy_transform.prepare_earnings_history_dataframe(legacy_transform.fetch_documents(database["earnings_history"]))
                earnings_calendar_frame = legacy_transform.prepare_earnings_calendar_dataframe(legacy_transform.fetch_documents(database["earnings_calendar"]))
                analyst_recs_frame = legacy_transform.prepare_analyst_recommendations_dataframe(legacy_transform.fetch_documents(database["analyst_recommendations"]))
                price_targets_frame = legacy_transform.prepare_price_targets_dataframe(legacy_transform.fetch_documents(database["price_targets"]))
                analyst_consensus_frame = legacy_transform.prepare_analyst_consensus_dataframe(legacy_transform.fetch_documents(database["analyst_consensus"]))
                growth_estimates_frame = legacy_transform.prepare_growth_estimates_dataframe(legacy_transform.fetch_documents(database["growth_estimates"]))

            conn.execute("CREATE SCHEMA IF NOT EXISTS company")
            if not metadata_frame.empty:
                conn.register("metadata_frame", metadata_frame)
                conn.execute("DROP TABLE IF EXISTS company.metadata")
                conn.execute("CREATE TABLE company.metadata AS SELECT * FROM metadata_frame")
                conn.unregister("metadata_frame")
                metrics["metadata_rows"] = len(metadata_frame)
            else:
                metrics["metadata_rows"] = 0
            log_event(self._logger, phase="transform.metadata", rows=metrics["metadata_rows"])

            conn.execute("CREATE SCHEMA IF NOT EXISTS earnings")
            if not earnings_history_frame.empty:
                metrics["earnings_history_rows"] = legacy_transform.upsert_dataframe(
                    conn,
                    earnings_history_frame,
                    "earnings.history_raw",
                    "earnings",
                    key_columns=["ticker", "report_date", "fiscal_period"],
                )
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS earnings.history_raw (ticker VARCHAR, report_date DATE, fiscal_period VARCHAR, eps_estimate DOUBLE, eps_actual DOUBLE, eps_surprise DOUBLE, surprise_pct DOUBLE, revenue_estimate DOUBLE, revenue_actual DOUBLE)")
                metrics["earnings_history_rows"] = 0
            log_event(self._logger, phase="transform.earnings_history", rows=metrics["earnings_history_rows"])

            if not earnings_calendar_frame.empty:
                conn.register("earnings_calendar_frame", earnings_calendar_frame)
                conn.execute("DROP TABLE IF EXISTS earnings.calendar")
                conn.execute("CREATE TABLE earnings.calendar AS SELECT * FROM earnings_calendar_frame")
                conn.unregister("earnings_calendar_frame")
                metrics["earnings_calendar_rows"] = len(earnings_calendar_frame)
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS earnings.calendar (ticker VARCHAR, earnings_date DATE, period_ending VARCHAR, estimate DOUBLE)")
                metrics["earnings_calendar_rows"] = 0
            log_event(self._logger, phase="transform.earnings_calendar", rows=metrics["earnings_calendar_rows"])

            conn.execute("CREATE SCHEMA IF NOT EXISTS analyst")
            if not analyst_recs_frame.empty:
                metrics["analyst_recs_rows"] = legacy_transform.upsert_dataframe(
                    conn,
                    analyst_recs_frame,
                    "analyst.recommendations_raw",
                    "analyst",
                )
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS analyst.recommendations_raw (ticker VARCHAR, date DATE, firm VARCHAR, from_grade VARCHAR, to_grade VARCHAR, action VARCHAR)")
                metrics["analyst_recs_rows"] = 0
            log_event(self._logger, phase="transform.analyst_recommendations", rows=metrics["analyst_recs_rows"])

            if not price_targets_frame.empty:
                conn.register("price_targets_frame", price_targets_frame)
                conn.execute("DROP TABLE IF EXISTS analyst.price_targets_raw")
                conn.execute("CREATE TABLE analyst.price_targets_raw AS SELECT * FROM price_targets_frame")
                conn.unregister("price_targets_frame")
                metrics["price_targets_rows"] = len(price_targets_frame)
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS analyst.price_targets_raw (ticker VARCHAR, date DATE, current_price DOUBLE, target_low DOUBLE, target_mean DOUBLE, target_high DOUBLE, num_analysts INTEGER)")
                metrics["price_targets_rows"] = 0
            log_event(self._logger, phase="transform.price_targets", rows=metrics["price_targets_rows"])

            if not analyst_consensus_frame.empty:
                conn.register("analyst_consensus_frame", analyst_consensus_frame)
                conn.execute("DROP TABLE IF EXISTS analyst.consensus_raw")
                conn.execute("CREATE TABLE analyst.consensus_raw AS SELECT * FROM analyst_consensus_frame")
                conn.unregister("analyst_consensus_frame")
                metrics["analyst_consensus_rows"] = len(analyst_consensus_frame)
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS analyst.consensus_raw (ticker VARCHAR, date DATE, strong_buy INTEGER, buy INTEGER, hold INTEGER, sell INTEGER, strong_sell INTEGER)")
                metrics["analyst_consensus_rows"] = 0
            log_event(self._logger, phase="transform.analyst_consensus", rows=metrics["analyst_consensus_rows"])

            if not growth_estimates_frame.empty:
                conn.register("growth_estimates_frame", growth_estimates_frame)
                conn.execute("DROP TABLE IF EXISTS analyst.growth_estimates_raw")
                conn.execute("CREATE TABLE analyst.growth_estimates_raw AS SELECT * FROM growth_estimates_frame")
                conn.unregister("growth_estimates_frame")
                metrics["growth_estimates_rows"] = len(growth_estimates_frame)
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS analyst.growth_estimates_raw (ticker VARCHAR, date DATE, current_qtr_growth DOUBLE, next_qtr_growth DOUBLE, current_year_growth DOUBLE, next_year_growth DOUBLE, next_5yr_growth DOUBLE)")
                metrics["growth_estimates_rows"] = 0
            log_event(self._logger, phase="transform.growth_estimates", rows=metrics["growth_estimates_rows"])

            metrics["ratios_rows"] = legacy_transform.create_ratios_table(conn)
            log_event(self._logger, phase="transform.ratios", rows=metrics["ratios_rows"])

            metrics["growth_view_rows"] = legacy_transform.create_growth_view(conn)
            log_event(self._logger, phase="transform.growth_view", rows=metrics["growth_view_rows"])

            metrics["peer_groups_rows"] = legacy_transform.create_peer_groups_table(conn)
            log_event(self._logger, phase="transform.peer_groups", rows=metrics["peer_groups_rows"])

            metrics["portfolio_rows"] = legacy_transform.create_portfolio_table(conn)
            log_event(self._logger, phase="transform.portfolios", rows=metrics["portfolio_rows"])

            from src.analysis.valuation import create_valuation_metrics_table, create_earnings_history_table, create_earnings_calendar_view

            metrics["valuation_rows"] = create_valuation_metrics_table(conn)
            log_event(self._logger, phase="transform.valuation_metrics", rows=metrics["valuation_rows"])

            metrics["earnings_history_view_rows"] = create_earnings_history_table(conn)
            log_event(self._logger, phase="transform.earnings_history_view", rows=metrics["earnings_history_view_rows"])

            metrics["earnings_calendar_view_rows"] = create_earnings_calendar_view(conn)
            log_event(self._logger, phase="transform.earnings_calendar_view", rows=metrics["earnings_calendar_view_rows"])

            from src.analysis.analyst import (
                create_analyst_recommendations_table,
                create_price_targets_table,
                create_analyst_consensus_table,
                create_growth_estimates_table,
            )

            metrics["analyst_recs_view_rows"] = create_analyst_recommendations_table(conn)
            log_event(self._logger, phase="transform.analyst_recommendations_view", rows=metrics["analyst_recs_view_rows"])

            metrics["price_targets_view_rows"] = create_price_targets_table(conn)
            log_event(self._logger, phase="transform.price_targets_view", rows=metrics["price_targets_view_rows"])

            metrics["analyst_consensus_view_rows"] = create_analyst_consensus_table(conn)
            log_event(self._logger, phase="transform.analyst_consensus_view", rows=metrics["analyst_consensus_view_rows"])

            metrics["growth_estimates_view_rows"] = create_growth_estimates_table(conn)
            log_event(self._logger, phase="transform.growth_estimates_view", rows=metrics["growth_estimates_view_rows"])

            from src.analysis.technical import create_technical_indicators_table

            metrics["technical_rows"] = create_technical_indicators_table(conn)
            log_event(self._logger, phase="transform.technical_indicators", rows=metrics["technical_rows"])

        finally:
            conn.close()

        return TransformResult(metrics)
