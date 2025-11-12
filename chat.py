#!/usr/bin/env python3
"""
Chat interface entry point wrapper.
Ensures src module is importable by adding project root to sys.path.
"""
import os
import sys
from pathlib import Path

import duckdb  # expose for tests

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

os.environ.setdefault("FINANGPT_DATA_DIR", str(project_root / "data"))

from src.ui import chat as _chat_module  # noqa: E402
from src.ui.chat import (  # noqa: E402
    main as chat_main,
    trim_conversation_history as _core_trim_conversation_history,
    MAX_HISTORY_LENGTH,
    MAX_RETRIES,
    execute_query_with_retry,
    print_welcome_message,
    print_help,
)
from src.query_engine.query import (  # noqa: E402
    build_system_prompt,
    check_data_freshness,
    extract_sql,
    extract_tickers_from_sql,
    introspect_schema,
    load_mongo_database,
    pretty_print,
    validate_sql,
    call_ollama_chat_with_retry,
)

requests = _chat_module.requests  # allow tests to patch chat.requests
duckdb = duckdb

__all__ = [
    "main",
    "trim_conversation_history",
    "MAX_HISTORY_LENGTH",
    "MAX_RETRIES",
    "call_ollama_chat",
    "execute_query_with_retry",
    "print_welcome_message",
    "print_help",
    "build_system_prompt",
    "check_data_freshness",
    "extract_sql",
    "extract_tickers_from_sql",
    "introspect_schema",
    "load_mongo_database",
    "pretty_print",
    "validate_sql",
    "requests",
    "duckdb",
]


def trim_conversation_history(
    messages,
    max_tokens: int = 4000,
    preserve_recent: int = 5,
):
    trimmed = _core_trim_conversation_history(messages, max_tokens=max_tokens, preserve_recent=preserve_recent)
    limit = MAX_HISTORY_LENGTH
    if not trimmed:
        return trimmed
    system = trimmed[0] if trimmed[0].get("role") == "system" else None
    body = trimmed[1:] if system else trimmed
    if len(body) > limit:
        body = body[-limit:]
    if system:
        return [system] + body
    return body


def call_ollama_chat(*args, **kwargs):
    return call_ollama_chat_with_retry(*args, **kwargs)


def _proxy_call_ollama_chat(*args, **kwargs):
    return call_ollama_chat(*args, **kwargs)


_chat_module.call_ollama_chat_with_retry = _proxy_call_ollama_chat


def main() -> int:
    return chat_main()


if __name__ == "__main__":
    sys.exit(main())
