#!/usr/bin/env python3
"""
Backward-compatible wrapper for config_loader.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.utils.config directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from config_loader.py is deprecated. "
    "Use 'from src.utils.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.utils.config import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.utils.config import main
    sys.exit(main())
