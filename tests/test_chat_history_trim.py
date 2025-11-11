from __future__ import annotations

import sqlite3

from src.ui.chat import enforce_history_window, MAX_HISTORY_LENGTH
from src.query_engine.query_history import QueryHistory


def _build_history(count: int):
    history = [{"role": "system", "content": "system"}]
    for idx in range(count):
        history.append({"role": "user", "content": f"Question {idx}"})
    return history


def test_enforce_history_window_trims_in_place():
    history = _build_history(MAX_HISTORY_LENGTH + 10)
    original_id = id(history)

    enforce_history_window(history)

    assert id(history) == original_id
    assert len(history) == MAX_HISTORY_LENGTH + 1
    assert history[0]["role"] == "system"
    assert history[-1]["content"] == f"Question {MAX_HISTORY_LENGTH + 9}"


def test_enforce_history_window_noop_when_below_cap():
    history = _build_history(5)
    snapshot = list(history)

    enforce_history_window(history)

    assert history == snapshot


def test_query_history_prunes_records(tmp_path):
    db_path = tmp_path / "history.db"
    history = QueryHistory(db_path=str(db_path), max_records=5)

    for idx in range(8):
        history.save_query(f"query {idx}", generated_sql=f"SELECT {idx}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM queries")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 5
