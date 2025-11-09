#!/usr/bin/env python3
"""Peer group definitions for comparative analysis."""

from __future__ import annotations

from typing import Dict, List

# Predefined peer groups for common queries
PEER_GROUPS: Dict[str, List[str]] = {
    "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
    "Magnificent Seven": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "QCOM", "AVGO", "MU"],
    "Cloud Computing": ["AMZN", "MSFT", "GOOGL", "CRM", "ORCL", "IBM"],
    "Social Media": ["META", "SNAP", "PINS", "TWTR", "RDDT"],
    "Streaming": ["NFLX", "DIS", "PARA", "WBD"],
    "E-commerce": ["AMZN", "EBAY", "SHOP", "ETSY", "MELI"],
    "Payment Processors": ["V", "MA", "PYPL", "SQ", "AXP"],
    "Electric Vehicles": ["TSLA", "RIVN", "LCID", "F", "GM"],
    "Airlines": ["AAL", "DAL", "UAL", "LUV", "JBLU"],
    "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Oil & Gas": ["XOM", "CVX", "COP", "SLB", "BP", "SHEL"],
    "Defense": ["LMT", "RTX", "BA", "NOC", "GD"],
    "Retail": ["WMT", "TGT", "COST", "HD", "LOW"],
    "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
    "Telecom": ["T", "VZ", "TMUS", "CMCSA"],
}


def get_peer_group(group_name: str) -> List[str]:
    """Get tickers for a named peer group.

    Args:
        group_name: Name of the peer group (case-insensitive)

    Returns:
        List of ticker symbols in the group

    Raises:
        KeyError: If the group name is not found
    """
    # Case-insensitive lookup
    for key, value in PEER_GROUPS.items():
        if key.lower() == group_name.lower():
            return value
    raise KeyError(f"Peer group '{group_name}' not found")


def list_peer_groups() -> List[str]:
    """Get list of all available peer group names.

    Returns:
        List of peer group names
    """
    return list(PEER_GROUPS.keys())


def get_all_peer_data() -> List[tuple]:
    """Get all peer group data as rows for database insertion.

    Returns:
        List of (ticker, peer_group) tuples
    """
    rows = []
    for group_name, tickers in PEER_GROUPS.items():
        for ticker in tickers:
            rows.append((ticker, group_name))
    return rows


def is_valid_peer_group(group_name: str) -> bool:
    """Check if a peer group name exists.

    Args:
        group_name: Name to check (case-insensitive)

    Returns:
        True if the group exists, False otherwise
    """
    return any(key.lower() == group_name.lower() for key in PEER_GROUPS.keys())
