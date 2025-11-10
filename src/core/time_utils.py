#!/usr/bin/env python3
"""Shared helpers for dealing with UTC timestamps."""

from __future__ import annotations

import re
from datetime import UTC, datetime

ISO_OFFSET_PATTERN = re.compile(r"[+-]\d{2}:\d{2}$")


def _normalise_timestamp(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("Missing timestamp value.")
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
    if not ISO_OFFSET_PATTERN.search(cleaned):
        cleaned = f"{cleaned}+00:00"
    return cleaned


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 string (optionally ending with Z) into a UTC datetime."""
    return datetime.fromisoformat(_normalise_timestamp(value)).astimezone(UTC)
