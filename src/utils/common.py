"""
Common utility functions for FinanGPT.

This module provides general utility functions that are used across multiple modules,
eliminating code duplication and ensuring consistent behavior.
"""

from numbers import Number
from typing import Any


def as_text(value: Any) -> str:
    """
    Safely convert a value to string with proper null handling.
    
    Args:
        value: Value to convert to string
        
    Returns:
        String representation with leading/trailing whitespace removed,
        or empty string if value is None
    """
    if value is None:
        return ""
    return str(value).strip()


def as_text_fallback(value: Any) -> str:
    """
    Safely convert a value to string with fallback for falsy values.
    
    Args:
        value: Value to convert to string
        
    Returns:
        String representation with leading/trailing whitespace removed,
        or empty string if value is falsy
    """
    return str(value or "").strip()


def is_numeric_strict(value: Any) -> bool:
    """
    Check if a value is numeric (int, float, Decimal, etc.) but not boolean.
    This version does not check for finite values.
    
    Args:
        value: Value to check
        
    Returns:
        True if the value is numeric (int, float, Decimal, etc.) but not boolean
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, Number):
        return True
    return False


def is_numeric(value: Any) -> bool:
    """
    Check if a value is numeric with additional finite check for floats.
    
    Args:
        value: Value to check
        
    Returns:
        True if the value is numeric (int, float, Decimal, etc.) but not boolean,
        and for floats, it's finite (not inf or NaN)
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, Number):
        # For floats, also check if they're finite (not inf or NaN)
        if isinstance(value, float):
            import math
            return math.isfinite(value)
        return True
    return False