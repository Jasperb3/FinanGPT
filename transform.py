#!/usr/bin/env python3
"""
Backward compatibility wrapper for transform.py
Redirects to src.transformation.transform module
"""
import sys
import runpy

if __name__ == "__main__":
    sys.exit(runpy.run_module("src.transformation.transform", run_name="__main__"))
