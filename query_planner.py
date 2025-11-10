#!/usr/bin/env python3
"""
Backward-compatible wrapper for query_planner.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.query.planner directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from query_planner.py is deprecated. "
    "Use 'from src.query.planner import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.query.planner import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.query.planner import main
    sys.exit(main())
