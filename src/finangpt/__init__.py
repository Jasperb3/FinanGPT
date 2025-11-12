"""FinanGPT package exports application layer and CLI helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CLI_PATH = Path(__file__).resolve().parents[2] / "finangpt.py"
_CLI_SPEC = importlib.util.spec_from_file_location("_finangpt_cli", _CLI_PATH)
if _CLI_SPEC and _CLI_SPEC.loader:
    _CLI_MODULE = importlib.util.module_from_spec(_CLI_SPEC)
    _CLI_SPEC.loader.exec_module(_CLI_MODULE)
else:  # pragma: no cover
    raise ImportError("Unable to load FinanGPT CLI module")

main = _CLI_MODULE.main
get_status = _CLI_MODULE.get_status
print_status = _CLI_MODULE.print_status
duckdb = _CLI_MODULE.duckdb
MongoClient = _CLI_MODULE.MongoClient

__all__ = [
    "main",
    "get_status",
    "print_status",
    "duckdb",
    "MongoClient",
]
