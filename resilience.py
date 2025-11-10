#!/usr/bin/env python3
"""
Backward-compatible wrapper for resilience.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.query.resilience directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from resilience.py is deprecated. "
    "Use 'from src.query.resilience import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.query.resilience import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.query.resilience import main
    sys.exit(main())
