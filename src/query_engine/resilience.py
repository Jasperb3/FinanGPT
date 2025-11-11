#!/usr/bin/env python3
"""Phase 6: Error Resilience & UX Polish utilities.

This module provides:
- Graceful degradation when Ollama is unavailable
- Query templates library
- Ticker validation and autocomplete
- Comprehensive debug logging
"""

from __future__ import annotations

import sys
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Callable, Sequence, Tuple

import duckdb
import yaml


def load_query_templates(templates_file: str = "templates/queries.yaml") -> Dict[str, Any]:
    """Load query templates from YAML file.

    Args:
        templates_file: Path to templates YAML file

    Returns:
        Dictionary of template definitions

    Raises:
        FileNotFoundError: If templates file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    templates_path = Path(templates_file)
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates file not found: {templates_file}")

    with templates_path.open("r") as f:
        templates = yaml.safe_load(f)

    return templates or {}


def execute_template(
    template_name: str,
    params: Dict[str, Any],
    conn: duckdb.DuckDBPyConnection,
    templates: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], List[tuple], str]:
    """Execute a query template with given parameters.

    Args:
        template_name: Name of the template to execute
        params: Dictionary of parameter values
        conn: DuckDB connection
        templates: Pre-loaded templates dict (optional)

    Returns:
        Tuple of (columns, rows, sql)

    Raises:
        KeyError: If template not found or required param missing
        ValueError: If SQL generation fails
    """
    if templates is None:
        templates = load_query_templates()

    if template_name not in templates:
        raise KeyError(f"Template '{template_name}' not found. Available: {list(templates.keys())}")

    template = templates[template_name]

    # Merge defaults with provided params
    final_params = template.get("defaults", {}).copy()
    final_params.update(params)

    # Check for missing required params
    required_params = template.get("params", [])
    missing = [p for p in required_params if p not in final_params]
    if missing:
        raise KeyError(f"Missing required parameters for template '{template_name}': {missing}")

    # Generate SQL from template
    sql_template = template["sql"]
    try:
        sql = sql_template.format(**final_params)
    except KeyError as e:
        raise ValueError(f"Failed to generate SQL: missing parameter {e}")

    # Execute query
    result = conn.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()

    return (columns, rows, sql)


def list_templates(templates: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """List all available query templates.

    Args:
        templates: Pre-loaded templates dict (optional)

    Returns:
        List of dicts with template name and description
    """
    if templates is None:
        try:
            templates = load_query_templates()
        except FileNotFoundError:
            return []

    return [
        {"name": name, "description": tpl.get("description", "No description")}
        for name, tpl in templates.items()
    ]


def validate_ticker(ticker: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if ticker exists in database.

    Args:
        ticker: Stock ticker symbol
        conn: DuckDB connection

    Returns:
        True if ticker exists, False otherwise
    """
    try:
        result = conn.execute(
            "SELECT COUNT(*) FROM company.metadata WHERE ticker = ?",
            [ticker.upper()]
        ).fetchone()
        return result[0] > 0 if result else False
    except duckdb.Error:
        # Table might not exist
        return False


def suggest_tickers(partial: str, conn: duckdb.DuckDBPyConnection, limit: int = 10) -> List[str]:
    """Autocomplete ticker symbols.

    Args:
        partial: Partial ticker string
        conn: DuckDB connection
        limit: Maximum number of suggestions

    Returns:
        List of matching ticker symbols
    """
    try:
        results = conn.execute(
            "SELECT ticker FROM company.metadata WHERE ticker LIKE ? ORDER BY ticker LIMIT ?",
            [f"{partial.upper()}%", limit]
        ).fetchall()
        return [r[0] for r in results]
    except duckdb.Error:
        return []


def jittered_backoff_delays(base_seconds: float, factor: float, max_retries: int) -> List[float]:
    """Return a list of backoff delays with jitter."""

    delays = []
    for attempt in range(max_retries):
        delay = base_seconds * (factor ** attempt)
        jitter = random.uniform(0, delay * 0.25)
        delays.append(delay + jitter)
    return delays


def bounded_parallel_map(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: int,
    timeout: float,
) -> Tuple[List[Any], List[Any]]:
    """Execute func over items with bounded parallelism and timeout.

    Returns (results, errors) where each result is a tuple (item, value) and
    each error is (item, exception).
    """

    results: List[Any] = []
    errors: List[Any] = []
    if not items:
        return results, errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(func, item): item for item in items}
        start = time.time()
        try:
            for future in as_completed(future_map, timeout=timeout):
                item = future_map[future]
                try:
                    value = future.result()
                    results.append((item, value))
                except Exception as exc:  # noqa: BLE001
                    errors.append((item, exc))
        except FuturesTimeoutError:
            pass

        elapsed = time.time() - start
        if elapsed >= timeout:
            for future, item in future_map.items():
                if not future.done():
                    future.cancel()
                    errors.append((item, TimeoutError("freshness batch timed out")))

    return results, errors


def get_all_tickers(conn: duckdb.DuckDBPyConnection) -> List[str]:
    """Get all available tickers from database.

    Args:
        conn: DuckDB connection

    Returns:
        List of all ticker symbols
    """
    try:
        results = conn.execute("SELECT DISTINCT ticker FROM company.metadata ORDER BY ticker").fetchall()
        return [r[0] for r in results]
    except duckdb.Error:
        return []


def handle_ollama_failure(error: Exception) -> Optional[str]:
    """Handle Ollama connection failures with graceful degradation.

    Args:
        error: The exception that was raised

    Returns:
        SQL string if user provides direct SQL, None to exit
    """
    print("\n⚠️  Ollama is not reachable. Connection error:")
    print(f"   {error}")
    print("\n💡 Fallback options:")
    print("   1. Enter SQL directly (expert mode)")
    print("   2. Use saved query templates")
    print("   3. Exit and fix connection")

    choice = input("\nSelect [1/2/3]: ").strip()

    if choice == "1":
        print("\n📝 Expert Mode: Enter your SQL query directly")
        print("   (Must be a SELECT statement with allowed tables)")
        sql = input("\nSQL> ").strip()
        if sql:
            return sql
        else:
            print("No SQL provided.")
            return None

    elif choice == "2":
        try:
            templates = load_query_templates()
            print("\n📚 Available query templates:")
            for name, tpl in templates.items():
                print(f"   • {name}: {tpl.get('description', 'No description')}")

            template_name = input("\nTemplate name> ").strip()
            if template_name not in templates:
                print(f"Template '{template_name}' not found.")
                return None

            template = templates[template_name]
            print(f"\n📋 Template: {template.get('description')}")
            print(f"   Required parameters: {', '.join(template.get('params', []))}")

            # Collect parameters
            params = {}
            for param in template.get("params", []):
                default = template.get("defaults", {}).get(param)
                prompt = f"{param}" + (f" (default: {default})" if default else "") + "> "
                value = input(prompt).strip()
                if value:
                    params[param] = value
                elif default is not None:
                    params[param] = default

            # Generate SQL
            sql_template = template["sql"]
            # Merge defaults
            final_params = template.get("defaults", {}).copy()
            final_params.update(params)
            sql = sql_template.format(**final_params)
            print(f"\n📊 Generated SQL: {sql}")
            return sql

        except Exception as e:
            print(f"Error loading templates: {e}")
            return None

    else:
        print("Exiting. Please check Ollama connection and try again.")
        return None


def debug_log(message: str, enabled: bool = False) -> None:
    """Print debug message if debug mode is enabled.

    Args:
        message: Debug message to print
        enabled: Whether debug mode is active
    """
    if enabled:
        print(f"[DEBUG] {message}")


def print_debug_info(
    system_prompt: str,
    user_query: str,
    llm_response: str,
    validated_sql: str,
    query_time: float,
    rows: int,
    enabled: bool = False,
) -> None:
    """Print comprehensive debug information.

    Args:
        system_prompt: System prompt sent to LLM
        user_query: User's natural language query
        llm_response: LLM's raw response
        validated_sql: SQL after validation
        query_time: Query execution time in seconds
        rows: Number of rows returned
        enabled: Whether debug mode is active
    """
    if not enabled:
        return

    print("\n" + "=" * 70)
    print("[DEBUG] System Prompt ({} chars):".format(len(system_prompt)))
    print("=" * 70)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)

    print("\n" + "=" * 70)
    print(f"[DEBUG] User Query: {user_query}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("[DEBUG] LLM Response:")
    print("=" * 70)
    print(llm_response)

    print("\n" + "=" * 70)
    print("[DEBUG] Validated SQL:")
    print("=" * 70)
    print(validated_sql)

    print("\n" + "=" * 70)
    print(f"[DEBUG] Query Time: {query_time:.3f}s")
    print(f"[DEBUG] Rows Returned: {rows}")
    print("=" * 70 + "\n")
