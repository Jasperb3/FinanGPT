"""Deprecated SQL validation wrappers.

This module now proxies to ``src.query_engine.query`` to keep backward
compatibility with older imports (tests and external scripts). Use the
query engine implementations directly for new development.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence, Set

from src.query_engine.query import (
    ensure_select_columns_are_known as _ensure_select_columns_are_known,
    extract_cte_names as _extract_cte_names,
    extract_table_identifiers as _extract_table_identifiers,
    find_main_select_index as _find_main_select_index,
    validate_sql as _validate_sql,
)

_DEPRECATION_MESSAGE = (
    "src.query.validation is deprecated; import from src.query_engine.query instead."
)


def _warn() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


def validate_sql(
    sql: str,
    schema: Mapping[str, Sequence[str]],
    default_limit: int | None = None,
    max_limit: int | None = None,
    question: str | None = None,
) -> str:
    """Compatibility wrapper for :func:`src.query_engine.query.validate_sql`."""

    _warn()
    return _validate_sql(
        sql,
        schema,
        default_limit=default_limit,
        max_limit=max_limit,
        question=question,
    )


def extract_table_identifiers(sql: str) -> list[str]:
    """Compatibility wrapper for table extraction."""

    _warn()
    return _extract_table_identifiers(sql)


def extract_cte_names(sql: str, main_select_idx: int) -> list[str]:
    """Compatibility wrapper for CTE name extraction."""

    _warn()
    return _extract_cte_names(sql, main_select_idx)


def find_main_select_index(sql: str) -> int:
    """Compatibility wrapper for SELECT index detection."""

    _warn()
    return _find_main_select_index(sql)


def ensure_select_columns_are_known(
    sql: str,
    known_columns: Set[str],
    main_select_idx: int,
) -> None:
    """Compatibility wrapper for SELECT column validation."""

    _warn()
    _ensure_select_columns_are_known(sql, known_columns, main_select_idx)


__all__ = [
    "validate_sql",
    "extract_table_identifiers",
    "extract_cte_names",
    "find_main_select_index",
    "ensure_select_columns_are_known",
]
