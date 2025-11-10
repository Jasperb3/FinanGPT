#!/usr/bin/env python3
"""
Backward-compatible wrapper for date_parser.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.utils.date_parser directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from date_parser.py is deprecated. "
    "Use 'from src.utils.date_parser import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.utils.date_parser import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.utils.date_parser import main
    sys.exit(main())
