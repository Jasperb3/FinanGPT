"""DuckDB persistence adapter for FinanGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb

from src.utils.paths import get_duckdb_path

__all__ = ["DuckDBRepository"]


class DuckDBRepository:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            path = get_duckdb_path()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._path))
