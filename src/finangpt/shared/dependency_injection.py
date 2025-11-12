"""Dependency injection container for the new FinanGPT architecture."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import duckdb
import pymongo
from dependency_injector import containers, providers

from .config import AppConfig, load_config

__all__ = ["Container"]


class OllamaLLMService:
    """Minimal stub for the Ollama-backed LLM service."""

    def __init__(self, base_url: str, model: str, timeout: int, max_retries: int = 3) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:  # pragma: no cover - illustrative stub
        return f"[LLM:{self.model}] {prompt}"


class DuckDBFinancialRepository:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection


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


class SQLiteConversationRepository:
    def __init__(self, path: str) -> None:
        self._path = path
        self._connection = sqlite3.connect(path)


class DuckDBSchemaProvider:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def get_schema_description(self) -> str:  # pragma: no cover - stub
        return "Stub schema description"


class QueryPlanner:
    def __init__(self, llm_service: OllamaLLMService, schema_provider: DuckDBSchemaProvider) -> None:
        self._llm = llm_service
        self._schema = schema_provider


class DataRetriever:
    def __init__(
        self,
        repository: DuckDBFinancialRepository,
        cache_repository: RedisCacheRepository,
        config: Any,
    ) -> None:
        self._repository = repository
        self._cache = cache_repository
        self._config = config


class ResultAnalyzer:
    def __init__(self, llm_service: OllamaLLMService) -> None:
        self._llm = llm_service


class InsightSynthesizer:
    def __init__(self, llm_service: OllamaLLMService) -> None:
        self._llm = llm_service


class VisualizationDetector:
    def __init__(self, llm_service: OllamaLLMService) -> None:
        self._llm = llm_service


class ContextBuilder:
    def __init__(self, llm_service: OllamaLLMService) -> None:
        self._llm = llm_service


class ConversationManager:
    def __init__(
        self,
        conversation_repo: SQLiteConversationRepository,
        context_builder: ContextBuilder,
    ) -> None:
        self._repo = conversation_repo
        self._context_builder = context_builder


@dataclass
class AnalysisPipelineComponents:
    query_planner: QueryPlanner
    data_retriever: DataRetriever
    result_analyzer: ResultAnalyzer
    insight_synthesizer: InsightSynthesizer
    viz_detector: VisualizationDetector
    conversation_manager: ConversationManager


class AnalysisOrchestrator:
    def __init__(self, components: AnalysisPipelineComponents) -> None:
        self._components = components

    def run(self, question: str) -> dict[str, Any]:  # pragma: no cover - stub
        return {
            "question": question,
            "steps": ["plan", "retrieve", "analyze", "synthesize", "visualize"],
        }


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
    )

    conversation_manager = providers.Factory(
        ConversationManager,
        conversation_repo=conversation_repository,
        context_builder=context_builder,
    )

    pipeline_components = providers.Factory(
        AnalysisPipelineComponents,
        query_planner=query_planner,
        data_retriever=data_retriever,
        result_analyzer=result_analyzer,
        insight_synthesizer=insight_synthesizer,
        viz_detector=visualization_detector,
        conversation_manager=conversation_manager,
    )

    analysis_orchestrator = providers.Factory(
        AnalysisOrchestrator,
        components=pipeline_components,
    )
