#!/usr/bin/env python3
"""
Backward-compatible wrapper for time_utils.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.utils.time_utils directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from time_utils.py is deprecated. "
    "Use 'from src.utils.time_utils import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.utils.time_utils import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.utils.time_utils import main
    sys.exit(main())
