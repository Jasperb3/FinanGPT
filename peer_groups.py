"""Compatibility wrapper for peer group helpers."""

from src.core.peer_groups import (
    PEER_GROUPS,
    get_peer_group,
    list_peer_groups,
    get_all_peer_data,
    is_valid_peer_group,
)

__all__ = [
    "PEER_GROUPS",
    "get_peer_group",
    "list_peer_groups",
    "get_all_peer_data",
    "is_valid_peer_group",
]
