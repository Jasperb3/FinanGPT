#!/usr/bin/env python3
"""
Query module entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

from src.query_engine.query import (  # noqa: E402
    main as query_main,
    extract_tickers_from_sql,
    check_data_freshness,
    build_system_prompt,
    ALLOWED_TABLES,
)

__all__ = [
    "main",
    "extract_tickers_from_sql",
    "check_data_freshness",
    "build_system_prompt",
    "ALLOWED_TABLES",
]


def main() -> int:
    return query_main()


if __name__ == "__main__":
    sys.exit(main())
