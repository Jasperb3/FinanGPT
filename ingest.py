#!/usr/bin/env python3
"""
Backward compatibility wrapper for ingest.py
Redirects to src.ingestion.ingest module
"""
import sys
import runpy

if __name__ == "__main__":
    sys.exit(runpy.run_module("src.ingestion.ingest", run_name="__main__"))
