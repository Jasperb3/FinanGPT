#!/usr/bin/env python3
"""
Backward-compatible wrapper for peer_groups.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.utils.peer_groups directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from peer_groups.py is deprecated. "
    "Use 'from src.utils.peer_groups import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.utils.peer_groups import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.utils.peer_groups import main
    sys.exit(main())
