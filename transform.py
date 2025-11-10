#!/usr/bin/env python3
"""
Backward-compatible wrapper for transform.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.transformation.core directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from transform.py is deprecated. "
    "Use 'from src.transformation.core import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.transformation.core import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.transformation.core import main
    sys.exit(main())
