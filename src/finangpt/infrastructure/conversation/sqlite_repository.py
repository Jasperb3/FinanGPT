"""SQLite-backed persistence for conversation turns and metadata."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

__all__ = ["ConversationTurn", "SQLiteConversationRepository"]


@dataclass(frozen=True)
class ConversationTurn:
    conversation_id: str
    role: str
    content: str
    metadata: dict | None
    created_at: datetime


class SQLiteConversationRepository:
    """Simple repository that stores conversation turns in SQLite."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_turns
            ON conversation_turns(conversation_id, created_at)
            """
        )
        conn.commit()
        conn.close()

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        payload = json.dumps(metadata or {}) if metadata else None
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO conversation_turns (conversation_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, payload),
        )
        conn.commit()
        conn.close()

    def add_turns(self, turns: Iterable[ConversationTurn]) -> int:
        conn = self._connect()
        rows = [
            (
                turn.conversation_id,
                turn.role,
                turn.content,
                json.dumps(turn.metadata or {}),
                turn.created_at.isoformat(timespec="seconds"),
            )
            for turn in turns
        ]
        if not rows:
            conn.close()
            return 0
        conn.executemany(
            """
            INSERT INTO conversation_turns (conversation_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            rows,
        )
        conn.commit()
        conn.close()
        return len(rows)

    def get_history(self, conversation_id: str, limit: int = 10) -> List[ConversationTurn]:
        conn = self._connect()
        cursor = conn.execute(
            """
            SELECT conversation_id, role, content, metadata, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        turns: List[ConversationTurn] = []
        for row in reversed(rows):
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            created_at = (
                datetime.fromisoformat(row["created_at"])
                if isinstance(row["created_at"], str)
                else datetime.fromtimestamp(row["created_at"])
            )
            turns.append(
                ConversationTurn(
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    metadata=metadata,
                    created_at=created_at,
                )
            )
        return turns

    def delete_conversation(self, conversation_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
        conn.close()
