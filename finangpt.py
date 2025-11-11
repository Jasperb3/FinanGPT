#!/usr/bin/env python3
"""
FinanGPT CLI entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import os
import sys
from pathlib import Path

import duckdb  # re-exported for tests that patch finangpt.duckdb
from pymongo import MongoClient  # re-exported for status tests

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

# Now import and run the actual CLI
from src.cli.finangpt import main as cli_main, get_status, print_status

__all__ = ["main", "get_status", "print_status", "duckdb", "MongoClient"]


def main() -> int:
    return cli_main()

if __name__ == "__main__":
    sys.exit(main())
