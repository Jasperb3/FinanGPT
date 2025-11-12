#!/usr/bin/env python3
"""Migrate legacy query history entries into the conversation store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if sys.path and sys.path[0] == str(ROOT):
    sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finangpt.infrastructure.conversation.sqlite_repository import SQLiteConversationRepository
from finangpt.shared.config import load_config
from src.utils.paths import get_history_db_path


def main() -> int:
    legacy_db = get_history_db_path()
    if not Path(legacy_db).exists():
        print("No legacy query history found; nothing to migrate.")
        return 0

    config = load_config()
    repo = SQLiteConversationRepository(config.database.conversation.path)

    conn = sqlite3.connect(legacy_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, user_query, generated_sql, row_count, execution_time_ms, timestamp
        FROM queries
        ORDER BY timestamp
        """
    ).fetchall()
    conn.close()

    for row in rows:
        metadata = {
            "generated_sql": row["generated_sql"],
            "row_count": row["row_count"],
            "execution_time_ms": row["execution_time_ms"],
            "source": "legacy_migration",
        }
        repo.add_turn(conversation_id=f"legacy-{row['id']}", role="user", content=row["user_query"], metadata=metadata)

    print(f"Imported {len(rows)} queries into {config.database.conversation.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
