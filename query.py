#!/usr/bin/env python3
"""LLM-to-SQL query runner with DuckDB guardrails."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

# Import visualization functions (optional - only used if available)
try:
    from visualize import (
        create_chart,
        detect_visualization_intent,
        pretty_print_formatted,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import Phase 6 resilience features
try:
    from resilience import (
        execute_template,
        handle_ollama_failure,
        list_templates,
        load_query_templates,
        print_debug_info,
        suggest_tickers,
        validate_ticker,
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

LOGS_DIR = Path("logs")
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
DEFAULT_STALENESS_THRESHOLD_DAYS = 7
ALLOWED_TABLES = (
    "financials.annual",
    "financials.quarterly",
    "prices.daily",
    "dividends.history",
    "splits.history",
    "company.metadata",
    "company.peers",
    "ratios.financial",
    "growth.annual",
    "user.portfolios",
    # Phase 8: Valuation & Earnings
    "valuation.metrics",
    "earnings.history",
    "earnings.calendar",
    "earnings.calendar_upcoming",
)


def configure_logger() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("query")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(LOGS_DIR / f"query_{datetime.now(UTC):%Y%m%d}.log")
    stream = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    stream.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(stream)
    return logger


def load_mongo_database(mongo_uri: str) -> Optional[Database]:
    """Load MongoDB database for freshness checking."""
    try:
        client = MongoClient(mongo_uri)
        db = client.get_default_database()
        if db:
            return db
        path = mongo_uri.rsplit("/", 1)[-1]
        if path:
            return client[path]
    except Exception:
        pass
    return None


def extract_tickers_from_sql(sql: str) -> List[str]:
    """Extract ticker symbols from SQL WHERE clauses."""
    tickers = []
    # Pattern: ticker = 'AAPL' or ticker IN ('AAPL', 'MSFT')
    single_ticker = re.findall(r"ticker\s*=\s*['\"]([A-Z]+)['\"]", sql, re.IGNORECASE)
    tickers.extend(single_ticker)

    # Pattern: ticker IN (...)
    in_clause = re.search(r"ticker\s+IN\s*\(([^)]+)\)", sql, re.IGNORECASE)
    if in_clause:
        in_tickers = re.findall(r"['\"]([A-Z]+)['\"]", in_clause.group(1))
        tickers.extend(in_tickers)

    return list(set(tickers))  # Deduplicate


def check_data_freshness(
    mongo_db: Optional[Database],
    tickers: List[str],
    threshold_days: int = DEFAULT_STALENESS_THRESHOLD_DAYS,
) -> Dict[str, Any]:
    """Check if data for the given tickers is stale.

    Returns a dict with:
    - is_stale: bool
    - stale_tickers: list of tickers with stale data
    - freshness_info: dict mapping ticker to days since last fetch
    """
    if not mongo_db or not tickers:
        return {"is_stale": False, "stale_tickers": [], "freshness_info": {}}

    try:
        metadata_collection = mongo_db["ingestion_metadata"]
        stale_tickers = []
        freshness_info = {}

        for ticker in tickers:
            # Check the most recent fetch across all data types
            most_recent = metadata_collection.find_one(
                {"ticker": ticker},
                sort=[("last_fetched", -1)]
            )

            if not most_recent:
                stale_tickers.append(ticker)
                freshness_info[ticker] = "never fetched"
                continue

            last_fetched_str = most_recent.get("last_fetched")
            if not last_fetched_str:
                stale_tickers.append(ticker)
                freshness_info[ticker] = "unknown"
                continue

            last_fetched = datetime.fromisoformat(last_fetched_str.replace("Z", "+00:00"))
            age = datetime.now(UTC) - last_fetched
            days_old = age.days

            if days_old >= threshold_days:
                stale_tickers.append(ticker)

            freshness_info[ticker] = f"{days_old} days old"

        return {
            "is_stale": len(stale_tickers) > 0,
            "stale_tickers": stale_tickers,
            "freshness_info": freshness_info,
        }
    except Exception:
        # If we can't check freshness, don't block the query
        return {"is_stale": False, "stale_tickers": [], "freshness_info": {}}


def log_event(logger: logging.Logger, **payload: Any) -> None:
    entry = {"ts": datetime.now(UTC).isoformat(), **payload}
    logger.info(json.dumps(entry))


def introspect_schema(conn: duckdb.DuckDBPyConnection, tables: Sequence[str]) -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    for table in tables:
        try:
            info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        except duckdb.CatalogException:
            continue
        if not info:
            continue
        columns = [row[1] for row in info]
        ordered = []
        for prefix in ("ticker", "date"):
            if prefix in columns:
                ordered.append(prefix)
        remaining = sorted(col for col in columns if col not in ordered)
        schema[table] = [*ordered, *remaining]
    return schema


def build_system_prompt(schema: Mapping[str, Sequence[str]]) -> str:
    from datetime import date, timedelta

    schema_lines = []
    for table, columns in schema.items():
        column_text = ", ".join(columns)
        schema_lines.append(f"- {table}: {column_text}")
    schema_block = "\n".join(schema_lines)

    # Date context for natural language parsing
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    five_years_ago = today - timedelta(days=365*5)

    date_context = f"""
Date Context (Today: {today.isoformat()}):
- "last year" or "past year" → WHERE date >= '{one_year_ago.isoformat()}'
- "last 5 years" → WHERE date >= '{five_years_ago.isoformat()}'
- "recent" or "latest" → ORDER BY date DESC LIMIT 1
- "2023" → WHERE YEAR(date) = 2023
- "YTD" or "year to date" → WHERE YEAR(date) = {today.year}
"""

    # Peer groups information
    peer_groups_info = """
Peer Groups (company.peers table):
Available peer groups: FAANG, Magnificent Seven, Semiconductors, Cloud Computing,
Social Media, Streaming, E-commerce, Payment Processors, Electric Vehicles, Airlines,
Banks, Oil & Gas, Defense, Retail, Pharma, Telecom.

Examples:
- "Compare FAANG revenue" → JOIN company.peers WHERE peer_group = 'FAANG'
- "Rank semiconductor companies" → JOIN company.peers WHERE peer_group = 'Semiconductors'
"""

    # Phase 8: Valuation & Earnings information
    valuation_earnings_info = """
Valuation Metrics (valuation.metrics table):
Ratios: pe_ratio, pb_ratio, ps_ratio, peg_ratio, dividend_yield, payout_ratio
Classifications: cap_class (Large Cap, Mid Cap, Small Cap)

Earnings Intelligence (earnings.history table):
Fields: eps_estimate, eps_actual, eps_surprise, surprise_pct, revenue_estimate, revenue_actual

Earnings Calendar (earnings.calendar and earnings.calendar_upcoming tables):
Fields: earnings_date, period_ending, estimate

Examples:
- "Find undervalued tech stocks with P/E < 15" → SELECT * FROM valuation.metrics WHERE pe_ratio < 15
- "Show stocks that beat earnings" → SELECT * FROM earnings.history WHERE eps_surprise > 0
- "Upcoming earnings this week" → SELECT * FROM earnings.calendar_upcoming WHERE earnings_date <= CURRENT_DATE + 7
"""

    # Window functions and statistical aggregations
    advanced_sql = """
Advanced SQL Features Allowed:
- Window functions: RANK(), ROW_NUMBER(), DENSE_RANK(), LAG(), LEAD(), NTILE()
- Statistical: AVG(), STDDEV(), MEDIAN(), PERCENTILE_CONT()
- Aggregations: SUM(), COUNT(), MIN(), MAX()
- Use PARTITION BY and ORDER BY with window functions
"""

    rules = [
        "Return a single SELECT statement that targets the DuckDB tables listed above.",
        "Do not mutate data. DDL/DML, temporary tables, and multi-statement SQL are forbidden.",
        "Always project the date column and cap the LIMIT at 100 rows.",
        "Default to LIMIT 25 when the user does not specify a limit.",
        "Prefer explicit column lists over SELECT * and keep SQL readable.",
        "Use window functions for rankings, running calculations, and peer comparisons.",
        "Reference peer groups table for comparative analysis across predefined groups.",
    ]
    rules_block = "\n".join(f"- {rule}" for rule in rules)

    return (
        "You are FinanGPT, a disciplined financial data analyst that writes safe DuckDB SQL.\n"
        f"Schema snapshot:\n{schema_block}\n\n"
        f"{date_context}\n"
        f"{peer_groups_info}\n"
        f"{valuation_earnings_info}\n"
        f"{advanced_sql}\n"
        f"Rules:\n{rules_block}\n"
        "Output only SQL, optionally wrapped in ```sql``` fences."
    )


def call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_query: str,
    timeout: int = 60,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
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


def extract_sql(text: str) -> str:
    if not text:
        raise ValueError("LLM response is empty.")
    code_block = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    select_match = re.search(r"(select\b.*)", text, re.IGNORECASE | re.DOTALL)
    if not select_match:
        raise ValueError("Could not locate SQL in the LLM response.")
    return select_match.group(1).strip()


def validate_sql(
    sql: str,
    schema: Mapping[str, Sequence[str]],
    default_limit: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
) -> str:
    if not sql:
        raise ValueError("SQL cannot be empty.")
    cleaned = re.sub(r"\s+", " ", sql.strip())
    cleaned_lower = cleaned.lower()
    if cleaned_lower.count(";") > 1 or (";" in cleaned[:-1] and not cleaned.endswith(";")):
        raise ValueError("Only single-statement SQL is allowed.")
    statement_start = cleaned_lower.lstrip()
    if not statement_start.startswith(("select", "with")):
        raise ValueError("Only SELECT statements are permitted.")
    main_select_idx = find_main_select_index(cleaned)
    if main_select_idx == -1:
        raise ValueError("Only SELECT statements are permitted.")
    cte_names = extract_cte_names(cleaned, main_select_idx)
    disallowed = ("insert", "update", "delete", "drop", "alter", "create", "replace", "grant", "revoke", "truncate")
    if any(re.search(rf"\b{word}\b", cleaned_lower) for word in disallowed):
        raise ValueError("Detected non-read-only SQL.")
    table_refs = extract_table_identifiers(cleaned)
    allowed_tables = {name.lower() for name in schema.keys()}
    cte_allow = {name.lower() for name in cte_names}
    if not table_refs:
        raise ValueError("SQL must reference one of the financials tables.")
    for table in table_refs:
        table_lower = table.lower()
        if table_lower not in allowed_tables and table_lower not in cte_allow:
            raise ValueError(f"Table {table} is not on the allow-list.")
    known_columns = {col.lower() for cols in schema.values() for col in cols}
    ensure_select_columns_are_known(cleaned, known_columns, main_select_idx)
    limit_match = re.search(r"\blimit\s+(\d+)\b", cleaned_lower)
    if limit_match:
        value = int(limit_match.group(1))
        if value > max_limit:
            raise ValueError(f"LIMIT {value} exceeds the maximum of {max_limit}.")
    else:
        cleaned = f"{cleaned} LIMIT {default_limit}"
    return cleaned.rstrip(";")


def extract_table_identifiers(sql: str) -> List[str]:
    pattern = re.compile(r"\b(from|join)\s+([a-zA-Z_][\w\.]*)", re.IGNORECASE)
    tables = []
    for _, table in pattern.findall(sql):
        cleaned = table.rstrip(",")
        if cleaned.startswith("("):
            continue
        tables.append(cleaned)
    return tables


def ensure_select_columns_are_known(sql: str, known_columns: Iterable[str], select_idx: int | None = None) -> None:
    clause = extract_select_clause(sql, select_idx)
    if not clause:
        raise ValueError("Unable to parse SELECT clause.")
    if clause.lower().startswith("distinct "):
        clause = clause[9:].strip()
    expressions = split_select_expressions(clause)
    if not expressions:
        raise ValueError("No columns selected.")
    unknown = []
    for expression in expressions:
        if not expression:
            continue
        if not expression_has_known_column(expression, known_columns):
            unknown.append(expression.strip())
    if unknown:
        raise ValueError(f"Unknown columns in SELECT: {', '.join(unknown[:3])}")


def extract_select_clause(sql: str, select_idx: int | None = None) -> str:
    lower_sql = sql.lower()
    if select_idx is None:
        select_idx = find_main_select_index(sql)
    if select_idx == -1:
        return ""
    from_match = re.search(r"\bfrom\b", lower_sql[select_idx + 6 :])
    if not from_match:
        return ""
    start = select_idx + 6
    end = start + from_match.start()
    return sql[start:end].strip()


def split_select_expressions(clause: str) -> List[str]:
    expressions = []
    current = []
    depth = 0
    for char in clause:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            expr = "".join(current).strip()
            if expr:
                expressions.append(expr)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        expressions.append(tail)
    return expressions


def expression_has_known_column(expression: str, known_columns: Iterable[str]) -> bool:
    cleaned = re.sub(r"(\".*?\"|'.*?')", "", expression)
    if "*" in cleaned:
        return True
    lowered = cleaned.lower()
    return any(re.search(rf"\b{re.escape(col)}\b", lowered) for col in known_columns)


def find_main_select_index(sql: str) -> int:
    lower = sql.lower()
    idx = 0
    depth = 0
    in_single = False
    in_double = False
    while idx < len(sql):
        char = sql[idx]
        if char == "'" and not in_double:
            if idx == 0 or sql[idx - 1] != "\\":
                in_single = not in_single
        elif char == '"' and not in_single:
            if idx == 0 or sql[idx - 1] != "\\":
                in_double = not in_double
        if in_single or in_double:
            idx += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif is_token_match(lower, idx, "select") and depth == 0:
            return idx
        idx += 1
    return -1


def is_token_match(lower_sql: str, idx: int, token: str) -> bool:
    token_len = len(token)
    if lower_sql.startswith(token, idx):
        before = lower_sql[idx - 1] if idx > 0 else " "
        after_idx = idx + token_len
        after = lower_sql[after_idx] if after_idx < len(lower_sql) else " "
        return not before.isalnum() and before != "_" and not after.isalnum() and after != "_"
    return False


def extract_cte_names(sql: str, select_idx: int) -> Set[str]:
    header = sql[:select_idx]
    pattern = re.compile(r"(?i)(?:with|,)\s*([a-zA-Z_][\w]*)\s*(?:\([^)]*\))?\s+as\s*\(")
    return {match.group(1) for match in pattern.finditer(header)}


def pretty_print(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    if not rows:
        print("No rows returned.")
        return
    widths = [len(str(col)) for col in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    header = " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns))
    divider = "-+-".join("-" * width for width in widths)
    print(header)
    print(divider)
    for row in rows:
        line = " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query DuckDB through an LLM→SQL layer.")
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural language question (leave empty to enter interactively).",
    )
    parser.add_argument(
        "--skip-freshness-check",
        action="store_true",
        help="Skip the data freshness check before querying.",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Disable automatic chart generation.",
    )
    parser.add_argument(
        "--no-formatting",
        action="store_true",
        help="Disable enhanced financial formatting.",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Use a saved query template (Phase 6).",
    )
    parser.add_argument(
        "--template-params",
        type=str,
        help="Template parameters as key=value pairs (comma-separated).",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List all available query templates.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable comprehensive debug logging.",
    )
    args = parser.parse_args()
    load_dotenv()
    model = os.getenv("MODEL_NAME", "phi4:latest")
    base_url = os.getenv("OLLAMA_URL")
    mongo_uri = os.getenv("MONGO_URI", "")

    # Handle --list-templates flag
    if args.list_templates:
        if not RESILIENCE_AVAILABLE:
            raise SystemExit("Resilience module not available. Install required dependencies.")
        templates = list_templates()
        if not templates:
            print("No query templates found.")
        else:
            print("\n📚 Available Query Templates:\n")
            for tpl in templates:
                print(f"  • {tpl['name']}: {tpl['description']}")
        return

    conn = duckdb.connect("financial_data.duckdb")
    schema = introspect_schema(conn, ALLOWED_TABLES)
    if not schema:
        conn.close()
        raise SystemExit("No DuckDB tables found. Run transform.py first.")

    logger = configure_logger()

    # Handle template execution
    if args.template:
        if not RESILIENCE_AVAILABLE:
            conn.close()
            raise SystemExit("Resilience module not available. Install pyyaml: pip install pyyaml")

        # Parse template parameters
        params = {}
        if args.template_params:
            for pair in args.template_params.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    params[key.strip()] = value.strip()

        try:
            columns, rows, sql = execute_template(args.template, params, conn)
            if args.debug:
                print(f"\n[DEBUG] Template: {args.template}")
                print(f"[DEBUG] Parameters: {params}")
                print(f"[DEBUG] Generated SQL: {sql}\n")

            print(f"📊 Executed template '{args.template}':\n")
            if VISUALIZATION_AVAILABLE and not args.no_formatting:
                pretty_print_formatted(columns, rows, use_formatting=True)
            else:
                pretty_print(columns, rows)

            log_event(logger, phase="query.template", template=args.template, rows=len(rows))
        except Exception as exc:
            log_event(logger, phase="query.template_error", template=args.template, error=str(exc))
            conn.close()
            raise SystemExit(f"Template execution failed: {exc}") from exc
        finally:
            conn.close()
        return

    # Normal query flow
    if not base_url:
        conn.close()
        raise SystemExit("OLLAMA_URL is not set.")

    question = args.question or input("Query> ").strip()
    if not question:
        conn.close()
        raise SystemExit("A natural language query is required.")

    system_prompt = build_system_prompt(schema)

    # Load MongoDB for freshness checking
    mongo_db = None
    if mongo_uri and not args.skip_freshness_check:
        mongo_db = load_mongo_database(mongo_uri)

    try:
        # Try to call Ollama
        try:
            response_text = call_ollama(base_url, model, system_prompt, question)
            sql = extract_sql(response_text)

            if args.debug and RESILIENCE_AVAILABLE:
                print(f"\n[DEBUG] LLM Response:\n{response_text}\n")
                print(f"[DEBUG] Extracted SQL:\n{sql}\n")

        except requests.ConnectionError as conn_err:
            # Graceful degradation when Ollama is down
            if RESILIENCE_AVAILABLE:
                sql = handle_ollama_failure(conn_err)
                if not sql:
                    conn.close()
                    raise SystemExit("Exiting due to Ollama connection failure.") from conn_err
            else:
                conn.close()
                raise SystemExit(f"Ollama connection failed: {conn_err}") from conn_err

        # Validate SQL
        sanitised_sql = validate_sql(sql, schema)

        if args.debug:
            print(f"[DEBUG] Validated SQL:\n{sanitised_sql}\n")

        # Check data freshness before executing
        if mongo_db and not args.skip_freshness_check:
            tickers = extract_tickers_from_sql(sanitised_sql)
            if tickers:
                freshness = check_data_freshness(mongo_db, tickers)
                if freshness["is_stale"]:
                    print("\n⚠️  Warning: Data may be stale")
                    print(f"Stale tickers: {', '.join(freshness['stale_tickers'])}")
                    print("\nFreshness details:")
                    for ticker, info in freshness["freshness_info"].items():
                        print(f"  {ticker}: {info}")
                    print(f"\nTo update data, run: python ingest.py --refresh --tickers {','.join(tickers)}")

                    user_input = input("\nContinue with stale data? [y/N]: ").strip().lower()
                    if user_input not in ("y", "yes"):
                        raise SystemExit("Query cancelled. Please refresh data and try again.")

        # Execute query with timing
        import time
        start_time = time.time()
        result = conn.execute(sanitised_sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        query_time = time.time() - start_time

        # Print debug info if enabled
        if args.debug and RESILIENCE_AVAILABLE:
            print_debug_info(
                system_prompt,
                question,
                response_text if 'response_text' in locals() else "(Direct SQL)",
                sanitised_sql,
                query_time,
                len(rows),
                enabled=True,
            )

        # Use enhanced formatting if available
        if VISUALIZATION_AVAILABLE and not args.no_formatting:
            pretty_print_formatted(columns, rows, use_formatting=True)
        else:
            pretty_print(columns, rows)

        # Create visualization if enabled and available
        if VISUALIZATION_AVAILABLE and not args.no_chart and rows:
            df = pd.DataFrame(rows, columns=columns)
            chart_type = detect_visualization_intent(question, df)
            if chart_type:
                chart_path = create_chart(df, chart_type, "Query Result", question)
                if chart_path:
                    print(f"\n📈 Chart saved: {chart_path}")

        log_event(logger, phase="query.success", sql=sanitised_sql, rows=len(rows))

    except (requests.RequestException, ValueError, duckdb.Error) as exc:
        log_event(logger, phase="query.error", error=str(exc))
        raise SystemExit(f"Query failed: {exc}") from exc
    finally:
        conn.close()


if __name__ == "__main__":
    main()
