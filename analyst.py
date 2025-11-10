#!/usr/bin/env python3
"""
Backward-compatible wrapper for analyst.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.intelligence.analyst directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from analyst.py is deprecated. "
    "Use 'from src.intelligence.analyst import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.intelligence.analyst import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.intelligence.analyst import main
    sys.exit(main())
