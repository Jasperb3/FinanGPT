#!/usr/bin/env python3
"""
Backward-compatible wrapper for query.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.query.executor directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from query.py is deprecated. "
    "Use 'from src.query.executor import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.query.executor import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.query.executor import main
    sys.exit(main())
