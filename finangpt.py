#!/usr/bin/env python3
"""
Backward compatibility wrapper for finangpt.py
Redirects to src.cli.finangpt module
"""
import sys
import runpy

if __name__ == "__main__":
    sys.exit(runpy.run_module("src.cli.finangpt", run_name="__main__"))
