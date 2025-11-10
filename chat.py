#!/usr/bin/env python3
"""
Backward-compatible wrapper for chat.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.query.chat directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from chat.py is deprecated. "
    "Use 'from src.query.chat import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.query.chat import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.query.chat import main
    sys.exit(main())
