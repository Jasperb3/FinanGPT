#!/usr/bin/env python3
"""Interactive conversational query interface for FinanGPT with multi-turn context."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

# Import shared functions from query.py
from src.query_engine.query import (
    ALLOWED_TABLES,
    DEFAULT_STALENESS_THRESHOLD_DAYS,
    build_system_prompt,
    check_data_freshness,
    check_ollama_health,
    extract_sql,
    extract_tickers_from_sql,
    introspect_schema,
    load_mongo_database,
    pretty_print,
    validate_sql,
    call_ollama_chat_with_retry,
)

# Import centralized logging
from src.utils.logging import configure_logger, log_event

# Import visualization functions
from src.ui.visualize import (
    create_chart,
    detect_visualization_intent,
    export_to_csv,
    export_to_json,
    export_to_excel,
    pretty_print_formatted,
)

# Import Phase 6 resilience features (optional)
try:
    from src.query_engine.resilience import (
        handle_ollama_failure,
        print_debug_info,
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

# Import Phase 11 query intelligence features (optional)
try:
    from src.query_engine.query_history import (
        QueryHistory,
        format_query_history,
        format_favorites,
    )
    from src.ui.error_handler import SmartErrorHandler
    from src.ui.autocomplete import AutocompleteEngine
    QUERY_INTELLIGENCE_AVAILABLE = True
except ImportError:
    QUERY_INTELLIGENCE_AVAILABLE = False

LOGS_DIR = Path("logs")
MAX_RETRIES = 3
MAX_HISTORY_LENGTH = 20  # Limit conversation history to avoid token overflow


# ============================================================================
# Context Window Management
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


def trim_conversation_history(
    messages: List[Dict[str, str]],
    max_tokens: int = 4000,
    preserve_recent: int = 5
) -> List[Dict[str, str]]:
    """
    Trim conversation history to fit context window.

    Args:
        messages: Conversation history
        max_tokens: Maximum tokens to keep (default: 4000)
        preserve_recent: Always keep last N messages (default: 5)

    Returns:
        Trimmed conversation history
    """
    if len(messages) <= preserve_recent:
        return messages

    # Always keep system message (first) and recent messages
    system_msg = messages[0] if messages and messages[0]['role'] == 'system' else None
    start_idx = 1 if system_msg else 0

    # Split into middle and recent
    if len(messages) > preserve_recent + start_idx:
        middle_msgs = messages[start_idx:-preserve_recent]
        recent_msgs = messages[-preserve_recent:]
    else:
        middle_msgs = []
        recent_msgs = messages[start_idx:]

    # Calculate tokens
    system_tokens = estimate_tokens(system_msg['content']) if system_msg else 0
    recent_tokens = sum(estimate_tokens(m['content']) for m in recent_msgs)

    available_tokens = max_tokens - system_tokens - recent_tokens

    # Add middle messages until we hit limit
    included_middle = []
    for msg in reversed(middle_msgs):
        msg_tokens = estimate_tokens(msg['content'])
        if msg_tokens > available_tokens:
            break
        included_middle.insert(0, msg)
        available_tokens -= msg_tokens

    # Rebuild message list
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(included_middle)
    result.extend(recent_msgs)

    return result


EXAMPLE_QUERIES = [
    "Show AAPL revenue for the last 5 years",
    "Compare AAPL and MSFT profit margins over time",
    "What are the top 10 companies by market cap?",
    "Show TSLA stock price trends for 2024",
    "Which tech stocks have the highest ROE?",
    "List all companies in the semiconductor industry",
    "Show dividend history for AAPL",
    "Compare revenue growth for FAANG stocks",
]

def print_welcome_message(schema: Mapping[str, Sequence[str]]) -> None:
    """Display welcome message with available data and example queries."""
    print("\n" + "=" * 70)
    print("🤖 FinanGPT Interactive Query Interface")
    print("=" * 70)
    print("\n🎨 New: Automatic chart generation for time-series and comparison queries!")
    print("   Conversational mode: Ask follow-up questions and refine your queries!")
    print(f"\n📊 Available tables: {len(schema)}")
    for table in schema.keys():
        print(f"   • {table}")

    print("\n💡 Try asking:")
    for i, example in enumerate(EXAMPLE_QUERIES[:5], 1):
        print(f"   {i}. {example}")
    print(f"   6. Plot AAPL stock price over time")
    print(f"   7. Compare revenue for AAPL, MSFT, GOOGL")

    print("\n📝 Commands:")
    print("   /help    - Show this help message")
    print("   /clear   - Clear conversation history")
    print("   /exit    - Exit the chat")
    print("   /quit    - Exit the chat")

    print("\n📈 Visualization:")
    print("   Charts are automatically created for time-series and comparison queries")
    print("   Use keywords like 'plot', 'chart', 'compare', 'trend' for best results")

    print("\n" + "=" * 70 + "\n")


def print_help() -> None:
    """Print help message."""
    print("\n📚 FinanGPT Chat Help")
    print("-" * 70)
    print("\n💬 How to use:")
    print("   • Type natural language questions about financial data")
    print("   • Ask follow-up questions - the system remembers context")
    print("   • Refine queries based on results")

    print("\n📝 Special Commands:")
    print("   /help          - Show this help message")
    print("   /clear         - Clear conversation history and start fresh")
    print("   /exit          - Exit the chat interface")
    print("   /quit          - Exit the chat interface")

    if QUERY_INTELLIGENCE_AVAILABLE:
        print("\n📚 Query History Commands (Phase 11):")
        print("   /history       - Show recent queries")
        print("   /favorites     - Show favorite queries")
        print("   /recall <id>   - Re-run a previous query by ID")
        print("   /favorite <id> - Mark a query as favorite")
        print("   /search <term> - Search query history")

    print("\n💡 Example conversation:")
    print("   You: Show AAPL revenue for last 5 years")
    print("   AI: [Shows revenue data]")
    print("   You: Now compare to MSFT")
    print("   AI: [Shows comparison with same time range]")

    print("\n🔍 Tips:")
    print("   • Be specific with ticker symbols (e.g., AAPL, MSFT)")
    print("   • Mention time ranges explicitly when needed")
    print("   • Use 'compare', 'show', 'list' for clearer queries")

    print("-" * 70 + "\n")


def execute_query_with_retry(
    conn: duckdb.DuckDBPyConnection,
    base_url: str,
    model: str,
    conversation_history: List[Dict[str, str]],
    schema: Mapping[str, Sequence[str]],
    logger: logging.Logger,
    mongo_db: Optional[Database],
    skip_freshness: bool,
    user_query: str = "",
    debug: bool = False,
) -> Optional[Tuple[List[str], List[Tuple], str, pd.DataFrame]]:
    """Execute query with intelligent error recovery and retry logic.

    Returns: (columns, rows, sql, df) tuple or None if all retries failed.
    """
    error_handler = SmartErrorHandler(schema) if QUERY_INTELLIGENCE_AVAILABLE else None

    for attempt in range(MAX_RETRIES):
        try:
            # Trim conversation history to fit context window
            trimmed_history = trim_conversation_history(conversation_history, max_tokens=4000)

            # Call LLM with graceful degradation
            try:
                response_text = call_ollama_chat_with_retry(base_url, model, trimmed_history)
            except requests.ConnectionError as conn_err:
                # Graceful degradation when Ollama is down
                if RESILIENCE_AVAILABLE and attempt == 0:  # Only on first attempt
                    print(f"\n☢\u2005 Ollama connection error: {conn_err}")
                    sql = handle_ollama_failure(conn_err)
                    if not sql:
                        return None
                    response_text = f"Direct SQL: {sql}"
                else:
                    raise

            # Extract and validate SQL
            sql = extract_sql(response_text)
            sanitised_sql = validate_sql(sql, schema)

            if debug:
                print(f"\n[DEBUG] Attempt {attempt + 1}")
                print(f"[DEBUG] LLM Response:\n{response_text[:200]}...")
                print(f"[DEBUG] Extracted SQL:\n{sql}")
                print(f"[DEBUG] Validated SQL:\n{sanitised_sql}\n")

            # Check freshness before executing
            if mongo_db and not skip_freshness:
                tickers = extract_tickers_from_sql(sanitised_sql)
                if tickers:
                    freshness = check_data_freshness(mongo_db, tickers)
                    if freshness["is_stale"]:
                        print(f"\n☢\u2005 Warning: Data for {', '.join(freshness['stale_tickers'])} may be stale")
                        print(f"   Run: python ingest.py --refresh --tickers {','.join(tickers)}")

            # Execute query
            result = conn.execute(sanitised_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            # Create DataFrame for visualization
            df = pd.DataFrame(rows, columns=columns)

            # Log success
            log_event(
                logger,
                phase="query.success",
                attempt=attempt + 1,
                sql=sanitised_sql,
                rows=len(rows),
            )

            return (columns, rows, sanitised_sql, df)

        except (ValueError, duckdb.Error) as exc:
            error_msg = str(exc)
            log_event(
                logger,
                phase="query.error",
                attempt=attempt + 1,
                error=error_msg,
            )

            if attempt == MAX_RETRIES - 1:
                # Final attempt failed
                print(f"\n❌ Query failed after {MAX_RETRIES} attempts: {error_msg}")
                return None

            # Provide feedback to LLM for retry
            print(f"☢\u2005 Attempt {attempt + 1} failed, retrying...")
            if error_handler:
                feedback = error_handler.get_feedback(sql, error_msg, user_query)
            else:
                feedback = (
                    f"The previous SQL query failed with error: {error_msg}\n"
                    f"Please revise the query to fix this issue. "
                    f"Remember to only use tables from the allowed list and ensure all columns exist."
                )
            conversation_history.append({"role": "system", "content": feedback})

        except requests.RequestException as exc:
            # Network error - don't retry
            print(f"\n❌ Network error communicating with Ollama: {exc}")
            log_event(logger, phase="query.network_error", error=str(exc))
            return None

    return None


def handle_command(user_input: str, query_history: Optional[QueryHistory], conversation_history: List[Dict[str, str]], system_prompt: str) -> bool:
    """
    Handle slash commands.
    Returns True if the command was handled, False otherwise.
    """
    if user_input.lower() in {"/exit", "/quit"}:
        print("\n👋 Goodbye! Your session has been logged.")
        return True

    if user_input.lower() == "/help":
        print_help()
        return True

    if user_input.lower() == "/clear":
        conversation_history.clear()
        conversation_history.append({"role": "system", "content": system_prompt})
        print("\n🔄 Conversation history cleared.\n")
        return True

    if query_history and user_input.lower() == "/history":
        recent = query_history.get_recent_queries(limit=20)
        print(format_query_history(recent))
        return True

    if query_history and user_input.lower() == "/favorites":
        favorites = query_history.get_favorites()
        print(format_favorites(favorites))
        return True

    if query_history and user_input.lower().startswith("/recall "):
        try:
            query_id = int(user_input.split()[1])
            saved_query = query_history.get_query(query_id)
            if saved_query:
                user_input = saved_query["user_query"]
                print(f"\n🔄 Recalling query #{query_id}: {user_input}\n")
                return False  # Let the main loop process the recalled query
            else:
                print(f"\n❌ Query #{query_id} not found.\n")
                return True
        except (ValueError, IndexError):
            print("\n❌ Usage: /recall <id>\n")
            return True

    if query_history and user_input.lower().startswith("/favorite "):
        try:
            query_id = int(user_input.split()[1])
            query_history.mark_favorite(query_id, True)
            print(f"\n⭐️ Query #{query_id} marked as favorite.\n")
            return True
        except (ValueError, IndexError):
            print("\n❌ Usage: /favorite <id>\n")
            return True

    return False


def process_query(
    user_input: str,
    conn: duckdb.DuckDBPyConnection,
    base_url: str,
    model: str,
    conversation_history: List[Dict[str, str]],
    schema: Mapping[str, Sequence[str]],
    logger: logging.Logger,
    mongo_db: Optional[Database],
    skip_freshness: bool,
    debug: bool,
    query_history: Optional[QueryHistory],
):
    """
    Process a user's query.
    """
    conversation_history.append({"role": "user", "content": user_input})

    result = execute_query_with_retry(
        conn,
        base_url,
        model,
        conversation_history,
        schema,
        logger,
        mongo_db,
        skip_freshness,
        user_input,
        debug,
    )

    if result:
        columns, rows, sql, df = result

        print(f"\n📊 Generated SQL: {sql}")
        print(f"\n✅ Results ({len(rows)} rows):\n")
        pretty_print_formatted(columns, rows, use_formatting=True)

        chart_type = detect_visualization_intent(user_input, df)
        if chart_type and not df.empty:
            chart_path = create_chart(df, chart_type, f"Query Result - {chart_type.title()} Chart", user_input)
            if chart_path:
                print(f"\n📈 Chart saved: {chart_path}")

        if query_history:
            try:
                query_history.save_query(
                    user_query=user_input,
                    generated_sql=sql,
                    row_count=len(rows),
                    execution_time_ms=None,
                )
            except Exception:
                pass

        conversation_history.append({
            "role": "assistant",
            "content": f"Query executed successfully. SQL: {sql}\nReturned {len(rows)} rows."
        })

        if rows:
            conversation_history.append({
                "role": "system",
                "content": f"Query returned {len(rows)} rows with columns: {', '.join(columns)}"
            })
    else:
        print("💡 Tip: Try rephrasing your question or being more specific.\n")
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

    if len(conversation_history) > MAX_HISTORY_LENGTH + 1:
        system_prompt = [conversation_history[0]] if conversation_history and conversation_history[0]["role"] == "system" else []
        recent_messages = conversation_history[-MAX_HISTORY_LENGTH:]
        conversation_history = system_prompt + recent_messages

    print()


def run_chat_loop(
    conn: duckdb.DuckDBPyConnection,
    base_url: str,
    model: str,
    schema: Mapping[str, Sequence[str]],
    logger: logging.Logger,
    mongo_db: Optional[Database],
    skip_freshness: bool,
    debug: bool = False,
) -> None:
    """Main chat loop with conversation history."""
    system_prompt = build_system_prompt(schema)
    conversation_history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    query_history = None
    if QUERY_INTELLIGENCE_AVAILABLE:
        try:
            query_history = QueryHistory()
        except Exception as e:
            print(f"Warning: Could not initialize query history: {e}")

    print_welcome_message(schema)

    while True:
        try:
            user_input = input("💬 Query> ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if handle_command(user_input, query_history, conversation_history, system_prompt):
                    if user_input.lower() in {"/exit", "/quit"}:
                        break
                    continue

            process_query(
                user_input,
                conn,
                base_url,
                model,
                conversation_history,
                schema,
                logger,
                mongo_db,
                skip_freshness,
                debug,
                query_history,
            )

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as exc:
            print(f"\n❌ Unexpected error: {exc}")
            log_event(logger, phase="chat.error", error=str(exc))
            print()


def main() -> None:
    """Main entry point for interactive chat interface."""
    parser = argparse.ArgumentParser(
        description="Interactive conversational query interface for FinanGPT."
    )
    parser.add_argument(
        "--skip-freshness-check",
        action="store_true",
        help="Skip the data freshness check before querying.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable comprehensive debug logging (Phase 6).",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    model = os.getenv("MODEL_NAME", "gpt-oss:latest")
    base_url = os.getenv("OLLAMA_URL")
    mongo_uri = os.getenv("MONGO_URI", "")

    if not base_url:
        raise SystemExit("OLLAMA_URL is not set.")

    # Connect to DuckDB
    conn = duckdb.connect("financial_data.duckdb")
    schema = introspect_schema(conn, ALLOWED_TABLES)
    if not schema:
        conn.close()
        raise SystemExit("No DuckDB tables found. Run transform.py first.")

    # Setup logging
    logger = configure_logger("chat")
    log_event(logger, phase="chat.start", model=model)

    # Load MongoDB for freshness checking
    mongo_db = None
    if mongo_uri and not args.skip_freshness_check:
        mongo_db = load_mongo_database(mongo_uri)

    # Check Ollama health before starting chat loop
    if not check_ollama_health(base_url):
        print("☢\u2005 Warning: Ollama service health check failed.")
        print("   The service may be unreachable. Will attempt connection with retries.")
        print("   If issues persist, check that Ollama is running at:", base_url)
        print()

    try:
        run_chat_loop(
            conn,
            base_url,
            model,
            schema,
            logger,
            mongo_db,
            args.skip_freshness_check,
            args.debug,
        )
    finally:
        conn.close()
        log_event(logger, phase="chat.end")


if __name__ == "__main__":
    main()