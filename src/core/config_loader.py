#!/usr/bin/env python3
"""Configuration loader for FinanGPT with fallback to environment variables.

Phase 7: Unified Workflow & Automation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

import yaml  # type: ignore


class Config:
    """Configuration container for FinanGPT settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        self._config = config_dict

    @property
    def mongo_uri(self) -> str:
        """Get MongoDB URI."""
        return self._config.get("database", {}).get("mongo_uri", "mongodb://localhost:27017/financial_data")

    @property
    def duckdb_path(self) -> str:
        """Get DuckDB database path."""
        return self._config.get("database", {}).get("duckdb_path", "financial_data.duckdb")

    @property
    def ollama_url(self) -> str:
        """Get Ollama API URL."""
        return self._config.get("ollama", {}).get("url", "http://localhost:11434")

    @property
    def model_name(self) -> str:
        """Get Ollama model name."""
        return self._config.get("ollama", {}).get("model", "phi4:latest")

    @property
    def ollama_timeout(self) -> int:
        """Get Ollama request timeout in seconds."""
        return self._config.get("ollama", {}).get("timeout", 60)

    @property
    def ollama_max_retries(self) -> int:
        """Get maximum retries for Ollama requests."""
        return self._config.get("ollama", {}).get("max_retries", 3)

    @property
    def price_lookback_days(self) -> int:
        """Get price lookback days for ingestion."""
        return self._config.get("ingestion", {}).get("price_lookback_days", 365)

    @property
    def auto_refresh_threshold_days(self) -> int:
        """Get auto-refresh threshold in days."""
        return self._config.get("ingestion", {}).get("auto_refresh_threshold_days", 7)

    @property
    def ingestion_batch_size(self) -> int:
        """Get ingestion batch size."""
        return self._config.get("ingestion", {}).get("batch_size", 50)

    @property
    def retry_backoff(self) -> list:
        """Get retry backoff intervals."""
        return self._config.get("ingestion", {}).get("retry_backoff", [1, 2, 4])

    @property
    def default_limit(self) -> int:
        """Get default query result limit."""
        return self._config.get("query", {}).get("default_limit", 25)

    @property
    def max_limit(self) -> int:
        """Get maximum query result limit."""
        return self._config.get("query", {}).get("max_limit", 100)

    @property
    def enable_visualizations(self) -> bool:
        """Get visualization enablement flag."""
        return self._config.get("query", {}).get("enable_visualizations", True)

    @property
    def chart_output_dir(self) -> str:
        """Get chart output directory."""
        return self._config.get("query", {}).get("chart_output_dir", "charts/")

    @property
    def export_formats(self) -> list:
        """Get supported export formats."""
        return self._config.get("query", {}).get("export_formats", ["csv", "json", "excel"])

    @property
    def conversational_mode(self) -> bool:
        """Get conversational mode enablement flag."""
        return self._config.get("features", {}).get("conversational_mode", True)

    @property
    def auto_error_recovery(self) -> bool:
        """Get auto error recovery flag."""
        return self._config.get("features", {}).get("auto_error_recovery", True)

    @property
    def query_suggestions(self) -> bool:
        """Get query suggestions flag."""
        return self._config.get("features", {}).get("query_suggestions", True)

    @property
    def portfolio_tracking(self) -> bool:
        """Get portfolio tracking enablement flag."""
        return self._config.get("features", {}).get("portfolio_tracking", False)

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self._config.get("logging", {}).get("level", "INFO")

    @property
    def log_directory(self) -> str:
        """Get log directory path."""
        return self._config.get("logging", {}).get("directory", "logs/")

    @property
    def log_format(self) -> str:
        """Get log format (json or text)."""
        return self._config.get("logging", {}).get("format", "json")

    def get(self, key: str, default: Any = None) -> Any:
        """Get arbitrary configuration value."""
        return self._config.get(key, default)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with environment variable fallback.

    Args:
        config_path: Path to config.yaml file. If None, looks in current directory.

    Returns:
        Config object with merged settings.

    Priority order (highest to lowest):
    1. Environment variables (.env file or system)
    2. config.yaml file
    3. Default values
    """
    # Load environment variables from .env only when using default config path.
    if config_path is None:
        load_dotenv()

    # Try to load config file
    config_dict: Dict[str, Any] = {}

    if config_path is None:
        config_path = "config.yaml"

    config_file = Path(config_path)
    if config_file.exists():
        if yaml is None:
            print(
                "Warning: config.yaml found but PyYAML is not installed. "
                "Install it via `pip install -r requirements.txt` to use custom configuration."
            )
        else:
            try:
                with config_file.open("r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:  # type: ignore[attr-defined]
                print(f"Warning: Failed to parse {config_path}: {e}")
                config_dict = {}

    # Override with environment variables if present
    if os.getenv("MONGO_URI"):
        config_dict.setdefault("database", {})["mongo_uri"] = os.getenv("MONGO_URI")

    if os.getenv("OLLAMA_URL"):
        config_dict.setdefault("ollama", {})["url"] = os.getenv("OLLAMA_URL")

    if os.getenv("MODEL_NAME"):
        config_dict.setdefault("ollama", {})["model"] = os.getenv("MODEL_NAME")

    if os.getenv("PRICE_LOOKBACK_DAYS"):
        try:
            config_dict.setdefault("ingestion", {})["price_lookback_days"] = int(os.getenv("PRICE_LOOKBACK_DAYS"))
        except ValueError:
            pass

    return Config(config_dict)


def get_config() -> Config:
    """Get global configuration instance.

    This is a convenience function that returns a Config object
    loaded from the default config.yaml location.
    """
    return load_config()
