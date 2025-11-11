#!/usr/bin/env python3
"""
Ingest module entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

from src.ingestion.ingest import (  # noqa: E402
    main as ingest_main,
    get_last_fetch_info,
    is_data_stale,
    should_skip_ticker,
    get_last_price_date,
)

__all__ = [
    "main",
    "get_last_fetch_info",
    "is_data_stale",
    "should_skip_ticker",
    "get_last_price_date",
]


def main() -> int:
    return ingest_main()


if __name__ == "__main__":
    sys.exit(main())
