#!/usr/bin/env python3
"""
Backward-compatible wrapper for visualize.py.

This file maintains compatibility with legacy scripts and imports.
New code should import from src.visualization.charts directly.

DEPRECATED: This wrapper will be removed in a future version.
"""
import sys
import warnings

warnings.warn(
    "Direct import from visualize.py is deprecated. "
    "Use 'from src.visualization.charts import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location
from src.visualization.charts import *

if __name__ == "__main__":
    # Maintain CLI compatibility
    from src.visualization.charts import main
    sys.exit(main())
