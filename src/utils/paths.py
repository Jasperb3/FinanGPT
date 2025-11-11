"""Centralized data-directory helpers for FinanGPT."""

from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from src.core.config_loader import load_config

LOGGER = logging.getLogger("paths")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ENV_VAR = "FINANGPT_DATA_DIR"
CHART_RETENTION_ENV = "FINANGPT_CHART_RETENTION"


def _resolve_base_path(path_value: Optional[str]) -> Path:
    if not path_value:
        return PROJECT_ROOT
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    env_path = os.getenv(DATA_ENV_VAR)
    config = load_config()
    paths_cfg = config.get('paths', {}) if hasattr(config, 'get') else {}
    config_path = paths_cfg.get('data_dir') if isinstance(paths_cfg, dict) else None

    base = env_path or config_path or "data"
    data_dir = _resolve_base_path(base)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def reset_data_paths_cache_for_tests() -> None:
    get_data_dir.cache_clear()


def _legacy_path(relative: str) -> Path:
    return (PROJECT_ROOT / relative).resolve()


def _migrate_legacy_file(relative: str, destination: Path, description: str) -> None:
    legacy = _legacy_path(relative)
    if legacy.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(destination))
        LOGGER.info("Migrated %s to %s", description, destination)


def _migrate_legacy_dir(relative: str, destination: Path) -> None:
    legacy = _legacy_path(relative)
    if not legacy.exists() or not legacy.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for child in legacy.iterdir():
            target = destination / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        shutil.rmtree(legacy)
    else:
        shutil.move(str(legacy), str(destination))
    LOGGER.info("Migrated directory %s to %s", relative, destination)


def get_logs_dir() -> Path:
    logs_dir = get_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_dir("logs", logs_dir)
    return logs_dir


def get_charts_dir() -> Path:
    charts_dir = get_data_dir() / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_dir("charts", charts_dir)
    return charts_dir


def get_tmp_dir() -> Path:
    tmp_dir = get_data_dir() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_dir("tmp", tmp_dir)
    return tmp_dir


def get_duckdb_path(config_obj: Optional[object] = None) -> Path:
    config = config_obj or load_config()
    raw_path = config.duckdb_path if hasattr(config, 'duckdb_path') else "financial_data.duckdb"
    duckdb_path = _resolve_relative_to_data(raw_path)
    if not Path(raw_path).is_absolute():
        _migrate_legacy_file(raw_path, duckdb_path, "DuckDB database")
    return duckdb_path


def get_history_db_path() -> Path:
    history_path = get_data_dir() / "query_history.db"
    _migrate_legacy_file("query_history.db", history_path, "query history database")
    return history_path


def get_cache_metrics_path() -> Path:
    return get_logs_dir() / "cache_metrics.json"


def _resolve_relative_to_data(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (get_data_dir() / candidate).resolve()


def get_chart_retention_limit() -> int:
    env_value = os.getenv(CHART_RETENTION_ENV)
    if env_value:
        try:
            limit = int(env_value)
            return max(limit, 0)
        except ValueError:
            pass

    config = load_config()
    candidates = []
    if hasattr(config, 'get'):
        viz_cfg = config.get('visualization', {}) or {}
        query_cfg = config.get('query', {}) or {}
        candidates.append(viz_cfg.get('chart_retention_limit'))
        candidates.append(query_cfg.get('chart_retention_limit'))
    for value in candidates:
        if value is None:
            continue
        try:
            limit = int(value)
            return max(limit, 0)
        except (TypeError, ValueError):
            continue
    return 1000


def prune_old_charts(limit: Optional[int] = None) -> None:
    keep = get_chart_retention_limit() if limit is None else max(int(limit), 0)
    charts_dir = get_charts_dir()
    files = [
        path for path in charts_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    ]
    if len(files) <= keep:
        return
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[keep:]:
        try:
            path.unlink()
        except OSError:
            pass


def ensure_tmp_subdir(name: str) -> Path:
    path = get_tmp_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
