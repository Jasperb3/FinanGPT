#!/usr/bin/env python3
"""
Backward-compatible wrapper for autocomplete.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.intelligence.autocomplete directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from autocomplete.py is deprecated. "
    "Use 'from src.intelligence.autocomplete import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.intelligence.autocomplete import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.intelligence.autocomplete import main
    sys.exit(main())
