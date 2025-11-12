"""Typed configuration objects and loader for the FinanGPT stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

__all__ = [
    "AppConfig",
    "load_config",
    "AnalysisConfig",
]


class MongoDBConfig(BaseModel):
    uri: str = Field(default="mongodb://localhost:27017/financial_data")
    database_name: str = Field(default="financial_data")
    pool_size: int = 10
    timeout_ms: int = 5000


class DuckDBConfig(BaseModel):
    path: str = Field(default="./data/financial_data.duckdb")
    readonly: bool = False
    memory_limit: str = "2GB"


class ConversationDBConfig(BaseModel):
    path: str = Field(default="./data/conversations.db")


class DatabaseConfig(BaseModel):
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    conversation: ConversationDBConfig = Field(default_factory=ConversationDBConfig)


class OllamaConfig(BaseModel):
    url: str = Field(default="http://localhost:11434")
    model: str = Field(default="phi4:latest")
    timeout: int = 60
    max_retries: int = 3


class LLMGenerationConfig(BaseModel):
    planning_temperature: float = 0.1
    analysis_temperature: float = 0.3
    synthesis_temperature: float = 0.4
    max_context_tokens: int = 4000
    max_output_tokens: int = 1000


class LLMConfig(BaseModel):
    provider: str = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    generation: LLMGenerationConfig = Field(default_factory=LLMGenerationConfig)


class MarketRestrictionsConfig(BaseModel):
    mode: str = Field(default="global")
    exclude_etfs: bool = True
    exclude_mutualfunds: bool = True


class IngestionConfig(BaseModel):
    max_workers: int = 10
    worker_timeout: int = 120
    max_tickers_per_batch: int = 500
    price_lookback_days: int = 365
    auto_refresh_threshold_days: int = 7
    market_restrictions: MarketRestrictionsConfig = Field(default_factory=MarketRestrictionsConfig)


class TransformationConfig(BaseModel):
    chunk_size: int = 1000
    max_memory_mb: int = 2048
    enable_streaming: bool = True
    run_integrity_checks: bool = True


class AnalysisQueryPlanningConfig(BaseModel):
    max_retries: int = 2
    enable_validation: bool = True


class AnalysisResultAnalysisConfig(BaseModel):
    max_insights: int = 10
    min_confidence_threshold: float = 0.5


class AnalysisConversationConfig(BaseModel):
    max_history_length: int = 10
    context_summary_enabled: bool = True


class AnalysisConfig(BaseModel):
    max_query_steps: int = 10
    max_data_rows_per_step: int = 10000
    enable_query_caching: bool = True
    cache_ttl_seconds: int = 300
    v2_analysis_enabled: bool = False
    query_planning: AnalysisQueryPlanningConfig = Field(default_factory=AnalysisQueryPlanningConfig)
    result_analysis: AnalysisResultAnalysisConfig = Field(default_factory=AnalysisResultAnalysisConfig)
    conversation: AnalysisConversationConfig = Field(default_factory=AnalysisConversationConfig)


class VisualizationConfig(BaseModel):
    enable: bool = True
    output_dir: str = Field(default="./data/charts")
    dpi: int = 300
    default_style: str = Field(default="seaborn-v0_8")
    max_chart_retention: int = 1000


class StructuredLoggingConfig(BaseModel):
    enable: bool = True
    include_context: bool = True


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    directory: str = Field(default="./data/logs")
    max_file_size_mb: int = 10
    backup_count: int = 5
    structured: StructuredLoggingConfig = Field(default_factory=StructuredLoggingConfig)


class CacheConfig(BaseModel):
    redis_url: str = Field(default="redis://localhost:6379/0")
    default_ttl_seconds: int = 300
    enabled: bool = True


class MonitoringConfig(BaseModel):
    enable_metrics: bool = False
    metrics_port: int = 9090


class AppConfig(BaseSettings):
    """Application configuration loaded from YAML, env vars, and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINANGPT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    transformation: TransformationConfig = Field(default_factory=TransformationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    _yaml_env_var: ClassVar[str] = "FINANGPT_CONFIG_FILE"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_path = cls._determine_yaml_path()
        yaml_source = ()
        if yaml_path is not None:
            yaml_source = (YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),)

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            *yaml_source,
            file_secret_settings,
        )

    @classmethod
    def _determine_yaml_path(cls) -> Path | None:
        override = os.getenv(cls._yaml_env_var)
        if override:
            candidate = Path(override).expanduser()
            if candidate.is_file():
                return candidate
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "config" / "config.yaml",
            repo_root / "config.yaml",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None


_CONFIG_CACHE: AppConfig | None = None


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(*, reload: bool = False) -> AppConfig:
    """Load the application configuration with caching."""

    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or reload:
        config = AppConfig()
        env_override = os.getenv("FINANGPT_V2_ANALYSIS")
        if env_override is not None:
            flag = _coerce_bool(env_override)
            config = config.model_copy(
                update={
                    "analysis": config.analysis.model_copy(
                        update={"v2_analysis_enabled": flag}
                    )
                }
            )
        _CONFIG_CACHE = config
    return _CONFIG_CACHE
