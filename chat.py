#!/usr/bin/env python3
"""
Backward compatibility wrapper for chat.py
Redirects to src.ui.chat module
"""
import sys
import runpy

if __name__ == "__main__":
    sys.exit(runpy.run_module("src.ui.chat", run_name="__main__"))
