"""
Query History & Favorites Module

Manages user query history, favorites, and recall functionality.
Stores data in SQLite for persistence across sessions.

Author: FinanGPT Team
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

from src.utils.paths import get_history_db_path


class QueryHistory:
    """
    Manages query history and favorites using SQLite database.

    Features:
    - Save queries with metadata (SQL, row count, execution time)
    - Mark queries as favorites
    - Recall past queries
    - Search query history
    - Export history to CSV
    """

    def __init__(self, db_path: Optional[str] = None, max_records: int = 1000):
        """
        Initialize QueryHistory with database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or str(get_history_db_path())
        self.max_records = max_records
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create queries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                generated_sql TEXT,
                row_count INTEGER,
                execution_time_ms INTEGER,
                is_favorite BOOLEAN DEFAULT 0,
                error_message TEXT,
                notes TEXT
            )
        """)

        # Create index for faster searches
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON queries(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_favorites
            ON queries(is_favorite)
            WHERE is_favorite = 1
        """)

        conn.commit()
        conn.close()

    def save_query(
        self,
        user_query: str,
        generated_sql: Optional[str] = None,
        row_count: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Save a query to history.

        Args:
            user_query: Original natural language query
            generated_sql: Generated SQL statement
            row_count: Number of rows returned
            execution_time_ms: Execution time in milliseconds
            error_message: Error message if query failed

        Returns:
            Query ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO queries
            (user_query, generated_sql, row_count, execution_time_ms, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (user_query, generated_sql, row_count, execution_time_ms, error_message))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        self._prune_history()

        return query_id

    def _prune_history(self) -> None:
        """Keep stored rows within the configured limit."""

        if not self.max_records or self.max_records <= 0:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM queries ORDER BY timestamp DESC, id DESC")
        rows = cursor.fetchall()

        if len(rows) <= self.max_records:
            conn.close()
            return

        ids_to_delete = [row[0] for row in rows[self.max_records:]]
        for idx in range(0, len(ids_to_delete), 500):
            chunk = ids_to_delete[idx:idx + 500]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(f"DELETE FROM queries WHERE id IN ({placeholders})", chunk)

        conn.commit()
        conn.close()

    def mark_favorite(self, query_id: int, is_favorite: bool = True):
        """
        Mark or unmark a query as favorite.

        Args:
            query_id: Query ID
            is_favorite: True to mark as favorite, False to unmark
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE queries
            SET is_favorite = ?
            WHERE id = ?
        """, (1 if is_favorite else 0, query_id))

        conn.commit()
        conn.close()

    def add_notes(self, query_id: int, notes: str):
        """
        Add notes to a saved query.

        Args:
            query_id: Query ID
            notes: Notes text
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE queries
            SET notes = ?
            WHERE id = ?
        """, (notes, query_id))

        conn.commit()
        conn.close()

    def get_query(self, query_id: int) -> Optional[Dict]:
        """
        Retrieve a specific query by ID.

        Args:
            query_id: Query ID

        Returns:
            Dictionary with query details, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM queries WHERE id = ?
        """, (query_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """
        Retrieve recent queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of query dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM queries
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_favorites(self) -> List[Dict]:
        """
        Retrieve all favorite queries.

        Returns:
            List of favorite query dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM queries
            WHERE is_favorite = 1
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def search_queries(self, search_term: str, limit: int = 20) -> List[Dict]:
        """
        Search queries by text.

        Args:
            search_term: Text to search for in user_query field
            limit: Maximum number of results

        Returns:
            List of matching query dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM queries
            WHERE user_query LIKE ? OR notes LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", limit))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_query(self, query_id: int):
        """
        Delete a query from history.

        Args:
            query_id: Query ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM queries WHERE id = ?
        """, (query_id,))

        conn.commit()
        conn.close()

    def clear_history(self, keep_favorites: bool = True):
        """
        Clear query history.

        Args:
            keep_favorites: If True, keep favorite queries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if keep_favorites:
            cursor.execute("""
                DELETE FROM queries WHERE is_favorite = 0
            """)
        else:
            cursor.execute("""
                DELETE FROM queries
            """)

        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict:
        """
        Get query history statistics.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total queries
        cursor.execute("SELECT COUNT(*) FROM queries")
        total_queries = cursor.fetchone()[0]

        # Favorite count
        cursor.execute("SELECT COUNT(*) FROM queries WHERE is_favorite = 1")
        favorites_count = cursor.fetchone()[0]

        # Failed queries
        cursor.execute("SELECT COUNT(*) FROM queries WHERE error_message IS NOT NULL")
        failed_queries = cursor.fetchone()[0]

        # Average execution time
        cursor.execute("""
            SELECT AVG(execution_time_ms)
            FROM queries
            WHERE execution_time_ms IS NOT NULL
        """)
        avg_exec_time = cursor.fetchone()[0] or 0

        # Most common query patterns (simple analysis)
        cursor.execute("""
            SELECT user_query, COUNT(*) as count
            FROM queries
            GROUP BY user_query
            HAVING count > 1
            ORDER BY count DESC
            LIMIT 5
        """)
        common_queries = cursor.fetchall()

        conn.close()

        return {
            "total_queries": total_queries,
            "favorites_count": favorites_count,
            "failed_queries": failed_queries,
            "success_rate": (total_queries - failed_queries) / total_queries * 100 if total_queries > 0 else 0,
            "avg_execution_time_ms": round(avg_exec_time, 2),
            "common_queries": [{"query": q[0], "count": q[1]} for q in common_queries]
        }

    def export_to_csv(self, output_path: str, include_favorites_only: bool = False):
        """
        Export query history to CSV file.

        Args:
            output_path: Path to output CSV file
            include_favorites_only: If True, export only favorites
        """
        conn = sqlite3.connect(self.db_path)

        if include_favorites_only:
            query = "SELECT * FROM queries WHERE is_favorite = 1 ORDER BY timestamp DESC"
        else:
            query = "SELECT * FROM queries ORDER BY timestamp DESC"

        df = pd.read_sql_query(query, conn)
        conn.close()

        df.to_csv(output_path, index=False)


def format_query_history(queries: List[Dict], max_query_length: int = 60) -> str:
    """
    Format query history for display in chat interface.

    Args:
        queries: List of query dictionaries
        max_query_length: Maximum length for truncated queries

    Returns:
        Formatted string
    """
    if not queries:
        return "No queries in history."

    lines = []
    lines.append("\n📜 Recent Queries:")
    lines.append("=" * 80)

    for q in queries:
        query_id = q['id']
        timestamp = q['timestamp']
        user_query = q['user_query']
        is_favorite = q['is_favorite']
        error = q['error_message']

        # Truncate long queries
        if len(user_query) > max_query_length:
            user_query = user_query[:max_query_length] + "..."

        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = timestamp

        # Status indicator
        status = "✅" if not error else "❌"
        favorite = "⭐" if is_favorite else "  "

        lines.append(f"  {favorite} [{query_id:3d}] {status} {time_str} | {user_query}")

    lines.append("=" * 80)
    lines.append("\n💡 Use /recall <id> to re-run a query")
    lines.append("💡 Use /favorite <id> to star a query")

    return "\n".join(lines)


def format_favorites(favorites: List[Dict], max_query_length: int = 60) -> str:
    """
    Format favorite queries for display.

    Args:
        favorites: List of favorite query dictionaries
        max_query_length: Maximum length for truncated queries

    Returns:
        Formatted string
    """
    if not favorites:
        return "No favorite queries yet. Use /favorite <id> to star a query."

    lines = []
    lines.append("\n⭐ Favorite Queries:")
    lines.append("=" * 80)

    for q in favorites:
        query_id = q['id']
        user_query = q['user_query']
        notes = q['notes']

        # Truncate long queries
        if len(user_query) > max_query_length:
            user_query = user_query[:max_query_length] + "..."

        lines.append(f"  [{query_id:3d}] {user_query}")
        if notes:
            lines.append(f"        📝 {notes}")

    lines.append("=" * 80)
    lines.append("\n💡 Use /recall <id> to re-run a favorite")

    return "\n".join(lines)
