#!/usr/bin/env python3
"""
Backward compatibility wrapper for query.py
Redirects to src.query_engine.query module
"""
import sys
import runpy

if __name__ == "__main__":
    sys.exit(runpy.run_module("src.query_engine.query", run_name="__main__"))
