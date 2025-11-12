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
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

from src.ingestion import ingest as _ingest_module  # noqa: E402
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
    "yf",
    "UnsupportedInstrument",
    "ensure_supported_instrument",
]

yf = _ingest_module.yf
UnsupportedInstrument = _ingest_module.UnsupportedInstrument
ensure_supported_instrument = _ingest_module.ensure_supported_instrument

# Ensure `ingest.yf` can be imported for backward-compat tests
sys.modules.setdefault(f"{__name__}.yf", yf)


def __getattr__(name: str):
    if hasattr(_ingest_module, name):
        return getattr(_ingest_module, name)
    raise AttributeError(name)


def main() -> int:
    return ingest_main()


if __name__ == "__main__":
    sys.exit(main())
