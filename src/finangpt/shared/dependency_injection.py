"""Dependency injection container for the new FinanGPT architecture."""

from __future__ import annotations

import sqlite3
from typing import Any

import duckdb
import pymongo
from dependency_injector import containers, providers

from finangpt.application.analysis.data_retriever import DataRetriever
from finangpt.application.analysis.insight_synthesizer import InsightSynthesizer
from finangpt.application.analysis.orchestrator import AnalysisOrchestrator
from finangpt.application.analysis.query_planner import QueryPlanner
from finangpt.application.analysis.result_analyzer import ResultAnalyzer
from finangpt.application.analysis.visualization_detector import VisualizationDetector
from finangpt.application.conversation.context_builder import ContextBuilder
from finangpt.application.conversation.conversation_manager import ConversationManager
from finangpt.infrastructure.conversation.sqlite_repository import SQLiteConversationRepository
from finangpt.infrastructure.llm.ollama_service import OllamaLLMService

from .config import load_config

__all__ = [
    "Container",
    "DuckDBFinancialRepository",
    "RedisCacheRepository",
]


class DuckDBFinancialRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def execute(self, sql: str) -> "pd.DataFrame":
        import pandas as pd

        return self._connection.execute(sql).fetchdf()


class MongoDBRawDataRepository:
    def __init__(self, database: pymongo.database.Database) -> None:
        self._database = database


class RedisCacheRepository:
    """In-memory stub that simulates Redis interactions for tests."""

    def __init__(self, url: str, default_ttl_seconds: int = 300) -> None:
        self.url = url
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def get(self, key: str) -> Any:
        return self._cache.get(key)


class DuckDBSchemaProvider:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def get_schema_description(self) -> str:  # pragma: no cover - stub
        tables = self._connection.execute("SHOW TABLES").fetchall()
        return "\n".join(table[0] for table in tables)


class Container(containers.DeclarativeContainer):
    """Dependency injection container following the project specification."""

    config = providers.Singleton(load_config)

    # Infrastructure - Databases
    mongodb_client = providers.Singleton(
        pymongo.MongoClient,
        config.provided.database.mongodb.uri,
        maxPoolSize=config.provided.database.mongodb.pool_size,
        serverSelectionTimeoutMS=config.provided.database.mongodb.timeout_ms,
    )

    mongodb_database = providers.Singleton(
        lambda client, settings: client[settings.database.mongodb.database_name],
        mongodb_client,
        config,
    )

    duckdb_connection = providers.Singleton(
        duckdb.connect,
        config.provided.database.duckdb.path,
    )

    # Infrastructure - LLM
    llm_service = providers.Factory(
        OllamaLLMService,
        base_url=config.provided.llm.ollama.url,
        model=config.provided.llm.ollama.model,
        timeout=config.provided.llm.ollama.timeout,
        max_retries=config.provided.llm.ollama.max_retries,
    )

    # Infrastructure - Repositories
    financial_repository = providers.Factory(
        DuckDBFinancialRepository,
        connection=duckdb_connection,
    )

    raw_data_repository = providers.Factory(
        MongoDBRawDataRepository,
        database=mongodb_database,
    )

    conversation_repository = providers.Factory(
        SQLiteConversationRepository,
        path=config.provided.database.conversation.path,
    )

    cache_repository = providers.Factory(
        RedisCacheRepository,
        url=config.provided.cache.redis_url,
        default_ttl_seconds=config.provided.cache.default_ttl_seconds,
    )

    # Application - Analysis Components
    schema_provider = providers.Factory(
        DuckDBSchemaProvider,
        connection=duckdb_connection,
    )

    query_planner = providers.Factory(
        QueryPlanner,
        llm_service=llm_service,
        schema_provider=schema_provider,
    )

    data_retriever = providers.Factory(
        DataRetriever,
        repository=financial_repository,
        cache_repository=cache_repository,
        config=config.provided.analysis,
    )

    result_analyzer = providers.Factory(
        ResultAnalyzer,
        llm_service=llm_service,
    )

    insight_synthesizer = providers.Factory(
        InsightSynthesizer,
        llm_service=llm_service,
    )

    visualization_detector = providers.Factory(
        VisualizationDetector,
        llm_service=llm_service,
    )

    # Application - Conversation
    context_builder = providers.Factory(
        ContextBuilder,
        llm_service=llm_service,
        summary_enabled=config.provided.analysis.conversation.context_summary_enabled,
    )

    conversation_manager = providers.Factory(
        ConversationManager,
        conversation_repo=conversation_repository,
        context_builder=context_builder,
        history_limit=config.provided.analysis.conversation.max_history_length,
    )

    analysis_orchestrator = providers.Factory(
        AnalysisOrchestrator,
        query_planner=query_planner,
        data_retriever=data_retriever,
        result_analyzer=result_analyzer,
        insight_synthesizer=insight_synthesizer,
        viz_detector=visualization_detector,
        conversation_manager=conversation_manager,
    )
