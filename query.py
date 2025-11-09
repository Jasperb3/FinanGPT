#!/usr/bin/env python3
"""LLM-to-SQL query runner with DuckDB guardrails."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import duckdb
import requests
from dotenv import load_dotenv

LOGS_DIR = Path("logs")
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
ALLOWED_TABLES = ("financials.annual", "financials.quarterly")


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
    schema_lines = []
    for table, columns in schema.items():
        column_text = ", ".join(columns)
        schema_lines.append(f"- {table}: {column_text}")
    schema_block = "\n".join(schema_lines)
    rules = [
        "Return a single SELECT statement that targets the DuckDB tables listed above.",
        "Do not mutate data. DDL/DML, temporary tables, and multi-statement SQL are forbidden.",
        "Always project the date column and cap the LIMIT at 100 rows.",
        "Default to LIMIT 25 when the user does not specify a limit.",
        "Prefer explicit column lists over SELECT * and keep SQL readable.",
    ]
    rules_block = "\n".join(f"- {rule}" for rule in rules)
    return (
        "You are FinanGPT, a disciplined financial data analyst that writes safe DuckDB SQL.\n"
        f"Schema snapshot:\n{schema_block}\n\n"
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
    args = parser.parse_args()
    load_dotenv()
    model = os.getenv("MODEL_NAME", "phi4:latest")
    base_url = os.getenv("OLLAMA_URL")
    if not base_url:
        raise SystemExit("OLLAMA_URL is not set.")
    conn = duckdb.connect("financial_data.duckdb")
    schema = introspect_schema(conn, ALLOWED_TABLES)
    if not schema:
        conn.close()
        raise SystemExit("No DuckDB tables found. Run transform.py first.")
    question = args.question or input("Query> ").strip()
    if not question:
        raise SystemExit("A natural language query is required.")
    system_prompt = build_system_prompt(schema)
    logger = configure_logger()
    try:
        response_text = call_ollama(base_url, model, system_prompt, question)
        sql = extract_sql(response_text)
        sanitised_sql = validate_sql(sql, schema)
        result = conn.execute(sanitised_sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        pretty_print(columns, rows)
    except (requests.RequestException, ValueError, duckdb.Error) as exc:
        log_event(logger, phase="query.error", error=str(exc))
        raise SystemExit(f"Query failed: {exc}") from exc
    finally:
        conn.close()


if __name__ == "__main__":
    main()
