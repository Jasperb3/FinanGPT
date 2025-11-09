#!/usr/bin/env python3
"""Interactive conversational query interface for FinanGPT with multi-turn context."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import duckdb
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

# Import shared functions from query.py
from query import (
    ALLOWED_TABLES,
    DEFAULT_STALENESS_THRESHOLD_DAYS,
    build_system_prompt,
    check_data_freshness,
    extract_sql,
    extract_tickers_from_sql,
    introspect_schema,
    load_mongo_database,
    pretty_print,
    validate_sql,
)

LOGS_DIR = Path("logs")
MAX_RETRIES = 3
MAX_HISTORY_LENGTH = 20  # Limit conversation history to avoid token overflow

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


def configure_logger() -> logging.Logger:
    """Initialize a JSON logger for chat sessions."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("chat")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(LOGS_DIR / f"chat_{datetime.now(UTC):%Y%m%d}.log")
    stream = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    stream.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(stream)
    return logger


def log_event(logger: logging.Logger, **payload: Any) -> None:
    """Emit a structured JSON log entry."""
    entry = {"ts": datetime.now(UTC).isoformat(), **payload}
    logger.info(json.dumps(entry))


def call_ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 60,
) -> str:
    """Call Ollama chat API with conversation history."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError("Ollama returned an empty response.")
    return content


def print_welcome_message(schema: Mapping[str, Sequence[str]]) -> None:
    """Display welcome message with available data and example queries."""
    print("\n" + "=" * 70)
    print("🤖 FinanGPT Interactive Query Interface (Phase 3)")
    print("=" * 70)
    print("\nConversational mode: Ask follow-up questions and refine your queries!")
    print(f"\n📊 Available tables: {len(schema)}")
    for table in schema.keys():
        print(f"   • {table}")

    print("\n💡 Try asking:")
    for i, example in enumerate(EXAMPLE_QUERIES[:5], 1):
        print(f"   {i}. {example}")

    print("\n📝 Commands:")
    print("   /help    - Show this help message")
    print("   /clear   - Clear conversation history")
    print("   /exit    - Exit the chat")
    print("   /quit    - Exit the chat")

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
    print("   /help    - Show this help message")
    print("   /clear   - Clear conversation history and start fresh")
    print("   /exit    - Exit the chat interface")
    print("   /quit    - Exit the chat interface")

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
) -> Optional[tuple]:
    """Execute query with intelligent error recovery and retry logic.

    Returns: (columns, rows, sql) tuple or None if all retries failed.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Call LLM
            response_text = call_ollama_chat(base_url, model, conversation_history)

            # Extract and validate SQL
            sql = extract_sql(response_text)
            sanitised_sql = validate_sql(sql, schema)

            # Check freshness before executing
            if mongo_db and not skip_freshness:
                tickers = extract_tickers_from_sql(sanitised_sql)
                if tickers:
                    freshness = check_data_freshness(mongo_db, tickers)
                    if freshness["is_stale"]:
                        print(f"\n⚠️  Warning: Data for {', '.join(freshness['stale_tickers'])} may be stale")
                        print(f"   Run: python ingest.py --refresh --tickers {','.join(tickers)}")

            # Execute query
            result = conn.execute(sanitised_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            # Log success
            log_event(
                logger,
                phase="query.success",
                attempt=attempt + 1,
                sql=sanitised_sql,
                rows=len(rows),
            )

            return (columns, rows, sanitised_sql)

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
            print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
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


def trim_conversation_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Trim conversation history to prevent token overflow.

    Keeps the system prompt and the most recent MAX_HISTORY_LENGTH messages.
    """
    if len(history) <= MAX_HISTORY_LENGTH + 1:  # +1 for system prompt
        return history

    # Keep system prompt (first message) and recent messages
    system_prompt = [history[0]] if history and history[0]["role"] == "system" else []
    recent_messages = history[-(MAX_HISTORY_LENGTH):]

    return system_prompt + recent_messages


def run_chat_loop(
    conn: duckdb.DuckDBPyConnection,
    base_url: str,
    model: str,
    schema: Mapping[str, Sequence[str]],
    logger: logging.Logger,
    mongo_db: Optional[Database],
    skip_freshness: bool,
) -> None:
    """Main chat loop with conversation history."""
    # Initialize conversation with system prompt
    system_prompt = build_system_prompt(schema)
    conversation_history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    print_welcome_message(schema)

    while True:
        try:
            # Get user input
            user_input = input("💬 Query> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in {"/exit", "/quit"}:
                print("\n👋 Goodbye! Your session has been logged.")
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() == "/clear":
                conversation_history = [
                    {"role": "system", "content": system_prompt}
                ]
                print("\n🔄 Conversation history cleared.\n")
                continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})

            # Execute query with retry logic
            result = execute_query_with_retry(
                conn,
                base_url,
                model,
                conversation_history,
                schema,
                logger,
                mongo_db,
                skip_freshness,
            )

            if result:
                columns, rows, sql = result

                # Show results
                print(f"\n📊 Generated SQL: {sql}")
                print(f"\n✅ Results ({len(rows)} rows):\n")
                pretty_print(columns, rows)

                # Add successful query to history
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Query executed successfully. SQL: {sql}\nReturned {len(rows)} rows."
                })

                # Add result summary to context
                if rows:
                    conversation_history.append({
                        "role": "system",
                        "content": f"Query returned {len(rows)} rows with columns: {', '.join(columns)}"
                    })
            else:
                # Query failed after retries
                print("💡 Tip: Try rephrasing your question or being more specific.\n")
                # Remove failed user query from history
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()

            # Trim history to prevent token overflow
            conversation_history = trim_conversation_history(conversation_history)

            print()  # Blank line for readability

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
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    model = os.getenv("MODEL_NAME", "phi4:latest")
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
    logger = configure_logger()
    log_event(logger, phase="chat.start", model=model)

    # Load MongoDB for freshness checking
    mongo_db = None
    if mongo_uri and not args.skip_freshness_check:
        mongo_db = load_mongo_database(mongo_uri)

    try:
        run_chat_loop(
            conn,
            base_url,
            model,
            schema,
            logger,
            mongo_db,
            args.skip_freshness_check,
        )
    finally:
        conn.close()
        log_event(logger, phase="chat.end")


if __name__ == "__main__":
    main()
