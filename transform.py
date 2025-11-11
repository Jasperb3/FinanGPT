#!/usr/bin/env python3
"""
Transform module entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

from src.transformation.transform import (  # noqa: E402
    main as transform_main,
    prepare_dataframe,
    prepare_prices_dataframe,
    prepare_dividends_dataframe,
    prepare_splits_dataframe,
    prepare_metadata_dataframe,
    prepare_earnings_history_dataframe,
    prepare_earnings_calendar_dataframe,
)

__all__ = [
    "main",
    "prepare_dataframe",
    "prepare_prices_dataframe",
    "prepare_dividends_dataframe",
    "prepare_splits_dataframe",
    "prepare_metadata_dataframe",
    "prepare_earnings_history_dataframe",
    "prepare_earnings_calendar_dataframe",
]


def main() -> int:
    return transform_main()


if __name__ == "__main__":
    sys.exit(main())
