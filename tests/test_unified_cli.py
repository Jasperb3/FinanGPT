#!/usr/bin/env python3
"""Tests for Phase 7: Unified Workflow & Automation."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import Config, load_config


class TestConfigLoader:
    """Test configuration loader functionality."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = Config({})

        # Check default values
        assert config.mongo_uri == "mongodb://localhost:27017/financial_data"
        assert config.duckdb_path == "financial_data.duckdb"
        assert config.ollama_url == "http://localhost:11434"
        assert config.model_name == "phi4:latest"
        assert config.default_limit == 25
        assert config.max_limit == 100

    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "database": {
                "mongo_uri": "mongodb://test:27017/test_db",
                "duckdb_path": "test.duckdb",
            },
            "ollama": {
                "url": "http://test:11434",
                "model": "test-model",
                "timeout": 120,
            },
            "query": {
                "default_limit": 50,
                "max_limit": 200,
            },
        }

        config = Config(config_dict)

        assert config.mongo_uri == "mongodb://test:27017/test_db"
        assert config.duckdb_path == "test.duckdb"
        assert config.ollama_url == "http://test:11434"
        assert config.model_name == "test-model"
        assert config.ollama_timeout == 120
        assert config.default_limit == 50
        assert config.max_limit == 200

    def test_config_properties(self):
        """Test all configuration properties."""
        config_dict = {
            "database": {
                "mongo_uri": "mongodb://localhost:27017/test",
                "duckdb_path": "test.duckdb",
            },
            "ollama": {
                "url": "http://localhost:11434",
                "model": "phi4:latest",
                "timeout": 60,
                "max_retries": 3,
            },
            "ingestion": {
                "price_lookback_days": 365,
                "auto_refresh_threshold_days": 7,
                "batch_size": 50,
                "retry_backoff": [1, 2, 4],
            },
            "query": {
                "default_limit": 25,
                "max_limit": 100,
                "enable_visualizations": True,
                "chart_output_dir": "charts/",
                "export_formats": ["csv", "json", "excel"],
            },
            "features": {
                "conversational_mode": True,
                "auto_error_recovery": True,
                "query_suggestions": True,
                "portfolio_tracking": False,
            },
            "logging": {
                "level": "INFO",
                "directory": "logs/",
                "format": "json",
            },
        }

        config = Config(config_dict)

        # Database
        assert config.mongo_uri == "mongodb://localhost:27017/test"
        assert config.duckdb_path == "test.duckdb"

        # Ollama
        assert config.ollama_url == "http://localhost:11434"
        assert config.model_name == "phi4:latest"
        assert config.ollama_timeout == 60
        assert config.ollama_max_retries == 3

        # Ingestion
        assert config.price_lookback_days == 365
        assert config.auto_refresh_threshold_days == 7
        assert config.ingestion_batch_size == 50
        assert config.retry_backoff == [1, 2, 4]

        # Query
        assert config.default_limit == 25
        assert config.max_limit == 100
        assert config.enable_visualizations is True
        assert config.chart_output_dir == "charts/"
        assert config.export_formats == ["csv", "json", "excel"]

        # Features
        assert config.conversational_mode is True
        assert config.auto_error_recovery is True
        assert config.query_suggestions is True
        assert config.portfolio_tracking is False

        # Logging
        assert config.log_level == "INFO"
        assert config.log_directory == "logs/"
        assert config.log_format == "json"

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "database": {
                "mongo_uri": "mongodb://localhost:27017/test_db",
            },
            "ollama": {
                "model": "custom-model",
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.mongo_uri == "mongodb://localhost:27017/test_db"
            assert config.model_name == "custom-model"
            # Defaults should still be loaded
            assert config.duckdb_path == "financial_data.duckdb"
        finally:
            os.unlink(temp_path)

    @patch.dict(os.environ, {
        'MONGO_URI': 'mongodb://env:27017/env_db',
        'OLLAMA_URL': 'http://env:11434',
        'MODEL_NAME': 'env-model',
    })
    def test_env_vars_override_config(self):
        """Test that environment variables override config file."""
        config_data = {
            "database": {
                "mongo_uri": "mongodb://file:27017/file_db",
            },
            "ollama": {
                "url": "http://file:11434",
                "model": "file-model",
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Environment variables should override file
            assert config.mongo_uri == "mongodb://env:27017/env_db"
            assert config.ollama_url == "http://env:11434"
            assert config.model_name == "env-model"
        finally:
            os.unlink(temp_path)

    def test_invalid_yaml_fallback(self):
        """Test that invalid YAML falls back to defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Should fallback to defaults
            assert config.mongo_uri == "mongodb://localhost:27017/financial_data"
        finally:
            os.unlink(temp_path)

    def test_missing_config_file(self):
        """Test loading when config file doesn't exist."""
        config = load_config("/nonexistent/path/config.yaml")
        # Should use defaults
        assert config.mongo_uri == "mongodb://localhost:27017/financial_data"
        assert config.ollama_url == "http://localhost:11434"

    def test_config_get_method(self):
        """Test the generic get method."""
        config_dict = {
            "custom_key": "custom_value",
            "nested": {
                "key": "value",
            },
        }

        config = Config(config_dict)

        assert config.get("custom_key") == "custom_value"
        assert config.get("nested") == {"key": "value"}
        assert config.get("nonexistent", "default") == "default"


class TestStatusCommand:
    """Test status command functionality."""

    @patch('finangpt.MongoClient')
    @patch('finangpt.duckdb.connect')
    def test_get_status_structure(self, mock_duckdb, mock_mongo):
        """Test that get_status returns proper structure."""
        from finangpt import get_status

        # Mock MongoDB
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.distinct.return_value = ["AAPL", "MSFT"]
        mock_collection.find_one.return_value = None
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo.return_value.get_default_database.return_value = mock_db

        # Mock DuckDB
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (100,)
        mock_conn.execute.return_value = mock_result
        mock_duckdb.return_value = mock_conn

        status = get_status()

        # Check structure
        assert "timestamp" in status
        assert "database" in status
        assert "data_freshness" in status
        assert "configuration" in status

    def test_print_status(self, capsys):
        """Test status printing."""
        from finangpt import print_status

        status = {
            "timestamp": "2025-11-09T12:00:00Z",
            "database": {
                "mongodb": "connected",
                "duckdb": "connected",
                "ticker_count": 10,
                "table_counts": {
                    "financials.annual": 50,
                },
            },
            "data_freshness": {
                "average_age_days": 3.5,
                "oldest_age_days": 7,
                "newest_age_days": 1,
                "stale_ticker_count": 2,
                "stale_threshold_days": 7,
            },
            "configuration": {
                "mongo_uri": "mongodb://localhost:27017/financial_data",
                "ollama_url": "http://localhost:11434",
                "model": "phi4:latest",
                "duckdb_path": "financial_data.duckdb",
            },
        }

        print_status(status)

        captured = capsys.readouterr()
        assert "FinanGPT System Status" in captured.out
        assert "MongoDB: connected" in captured.out
        assert "Ticker Count: 10" in captured.out
        assert "Average Age: 3.5 days" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
