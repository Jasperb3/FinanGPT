#!/usr/bin/env python3
"""
Transform module entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import and run the actual transform module
from src.transformation.transform import main

if __name__ == "__main__":
    sys.exit(main())
