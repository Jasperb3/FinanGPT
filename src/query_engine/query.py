#!/usr/bin/env python3
"""LLM-to-SQL query runner with DuckDB guardrails."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
import threading
from collections import deque
import hashlib

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from src.core.time_utils import parse_utc_timestamp
from src.core.config_loader import load_config
from src.utils.paths import get_duckdb_path

# Import centralized logging
from src.utils.logging import configure_logger, log_event

# ============================================================================
# Security: Input Sanitization
# ============================================================================

def sanitize_ticker(ticker: str) -> str:
    """
    Validate and sanitize ticker symbol to prevent command injection.

    Args:
        ticker: Raw ticker symbol input

    Returns:
        Sanitized ticker symbol (uppercase)

    Raises:
        ValueError: If ticker format is invalid
    """
    if not ticker:
        raise ValueError("Ticker cannot be empty")

    # Allow only alphanumeric, dots, hyphens (standard ticker formats)
    if not re.match(r'^[A-Z0-9.\-]+$', ticker, re.IGNORECASE):
        raise ValueError(f"Invalid ticker format: {ticker}. Only alphanumeric, dots, and hyphens allowed.")

    # Max length check (typical tickers are 1-10 chars)
    if len(ticker) > 10:
        raise ValueError(f"Ticker too long: {ticker} (max 10 characters)")

    return ticker.upper()


def sanitize_tickers_input(tickers_str: str) -> List[str]:
    """
    Sanitize comma-separated tickers string.

    Args:
        tickers_str: Comma-separated ticker symbols

    Returns:
        List of sanitized ticker symbols
    """
    if not tickers_str:
        return []

    tickers = [t.strip() for t in tickers_str.split(',')]
    return [sanitize_ticker(t) for t in tickers if t]


# ============================================================================
# Security: Path Traversal Prevention
# ============================================================================

def validate_file_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> Path:
    """
    Validate file path to prevent path traversal attacks.

    Args:
        file_path: File path to validate
        allowed_dirs: List of allowed directory paths (optional)

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or outside allowed directories
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Convert to Path object and resolve
    try:
        path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/etc') or file_path.startswith('/sys'):
        raise ValueError(f"Path traversal detected: {file_path}")

    # Validate against allowed directories
    if allowed_dirs:
        allowed_paths = [Path(d).resolve() for d in allowed_dirs]
        if not any(str(path).startswith(str(allowed)) for allowed in allowed_paths):
            raise ValueError(f"Path not in allowed directories: {file_path}")

    # Ensure it's a file (not directory)
    if path.exists() and not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return path


def safe_read_file(file_path: str, allowed_dirs: Optional[List[str]] = None) -> str:
    """
    Safely read file contents with path validation.

    Args:
        file_path: File path to read
        allowed_dirs: List of allowed directories

    Returns:
        File contents as string

    Raises:
        ValueError: If path validation fails
        FileNotFoundError: If file doesn't exist
    """
    validated_path = validate_file_path(file_path, allowed_dirs)

    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(validated_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# Security: Error Message Sanitization
# ============================================================================

def sanitize_error_message(error: Exception, debug_mode: bool = False) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Args:
        error: Exception to sanitize
        debug_mode: If True, return full error details

    Returns:
        Sanitized error message
    """
    if debug_mode:
        # In debug mode, return full details
        return f"{type(error).__name__}: {str(error)}"

    # Production mode: generic messages
    error_type = type(error).__name__

    # Map specific errors to safe generic messages
    generic_messages = {
        'FileNotFoundError': "File not found. Please check the file path.",
        'PermissionError': "Permission denied. Please check file permissions.",
        'ValueError': "Invalid input provided. Please check your query.",
        'ConnectionError': "Database connection error. Please try again later.",
        'TimeoutError': "Request timed out. Please try again.",
        'OllamaConnectionError': "AI service is temporarily unavailable.",
        'OllamaTimeoutError': "AI service request timed out.",
        'OllamaResponseError': "AI service returned an invalid response.",
        'SQLExtractionError': "Could not generate SQL query. Please rephrase your question.",
        'SemanticValidationError': "Query doesn't match question intent. Please clarify.",
    }

    return generic_messages.get(error_type, "An error occurred. Please try again or contact support.")


def safe_error_response(error: Exception, debug_mode: bool = False) -> Dict[str, Any]:
    """
    Create a safe error response dictionary.

    Args:
        error: Exception to format
        debug_mode: If True, include full error details

    Returns:
        Error response dictionary
    """
    response = {
        'success': False,
        'error': sanitize_error_message(error, debug_mode),
        'error_type': type(error).__name__
    }

    if debug_mode:
        import traceback
        response['traceback'] = traceback.format_exc()

    return response


# ============================================================================
# Security: Credential Management
# ============================================================================

def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
    """
    Mask sensitive values for logging/display.

    Args:
        value: Sensitive value to mask
        show_chars: Number of characters to show at end

    Returns:
        Masked value (e.g., "***abc123")
    """
    if not value or len(value) <= show_chars:
        return "***"

    return "*" * (len(value) - show_chars) + value[-show_chars:]


def validate_env_var(var_name: str, required: bool = True) -> str:
    """
    Safely load environment variable with validation.

    Args:
        var_name: Environment variable name
        required: If True, raise error if not found

    Returns:
        Environment variable value

    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(var_name)

    if required and not value:
        raise ValueError(f"Required environment variable not set: {var_name}")

    return value or ""


def sanitize_connection_string(conn_str: str) -> str:
    """
    Mask password in connection string for safe logging.

    Args:
        conn_str: Database connection string

    Returns:
        Connection string with password masked
    """
    if not conn_str:
        return ""

    # Pattern for MongoDB connection strings
    # mongodb://user:password@host:port/db -> mongodb://user:***@host:port/db
    pattern = r'(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@.+)'

    def mask_password(match):
        return f"{match.group(1)}***{match.group(3)}"

    return re.sub(pattern, mask_password, conn_str)


# ============================================================================
# Exception Hierarchy
# ============================================================================

class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Ollama service unreachable."""
    pass


class OllamaTimeoutError(OllamaError):
    """Ollama request timed out."""
    pass


class OllamaResponseError(OllamaError):
    """Ollama returned invalid response."""
    pass


class SQLExtractionError(OllamaError):
    """Could not extract SQL from LLM response."""
    pass


class SemanticValidationError(Exception):
    """SQL doesn't semantically match the question."""
    pass

# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiter:
    """Token bucket rate limiter for Ollama requests."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()

    def acquire(self, block: bool = True) -> bool:
        """Acquire permission to make a request."""
        with self.lock:
            now = time.time()

            # Remove expired requests
            while self.requests and self.requests[0] < now - self.window:
                self.requests.popleft()

            # Check if we can proceed
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            if not block:
                return False

            # Block until slot available
            sleep_time = self.requests[0] + self.window - now
            if sleep_time > 0:
                logger = logging.getLogger("query")
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                return self.acquire(block=True)

            return True


# Global rate limiter instance
_ollama_rate_limiter = None


def get_rate_limiter(config: Dict[str, Any]) -> RateLimiter:
    """Get or create rate limiter instance based on configuration."""
    global _ollama_rate_limiter
    if _ollama_rate_limiter is None:
        max_requests = config.get('ollama', {}).get('rate_limit_requests', 10)
        window_seconds = config.get('ollama', {}).get('rate_limit_window', 60)
        _ollama_rate_limiter = RateLimiter(max_requests, window_seconds)
    return _ollama_rate_limiter

# Import visualization functions (optional - only used if available)
try:
    from src.ui.visualize import (
        create_chart,
        detect_visualization_intent,
        pretty_print_formatted,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import Phase 6 resilience features
try:
    from src.query_engine.resilience import (
        execute_template,
        handle_ollama_failure,
        list_templates,
        load_query_templates,
        print_debug_info,
        suggest_tickers,
        validate_ticker,
        bounded_parallel_map,
        jittered_backoff_delays,
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

# Import query caching module
try:
    from src.query.cache import QueryCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

LEGACY_DEFAULT_LIMIT = 25
LEGACY_MAX_LIMIT = 100

try:
    _CONFIG = load_config()
except Exception:
    _CONFIG = None

_QUERY_DEFAULT_LIMIT = _CONFIG.default_limit if _CONFIG else LEGACY_DEFAULT_LIMIT
_QUERY_MAX_LIMIT = _CONFIG.max_limit if _CONFIG else LEGACY_MAX_LIMIT
_QUERY_SETTINGS = _CONFIG.get('query', {}) if _CONFIG else {}

_SEMANTIC_VALIDATION_TOGGLE = (
    _QUERY_SETTINGS.get('semantic_validation_enabled')
    if isinstance(_QUERY_SETTINGS, dict)
    else None
)

DEFAULT_LIMIT = _QUERY_DEFAULT_LIMIT
MAX_LIMIT = _QUERY_MAX_LIMIT
DEFAULT_STALENESS_THRESHOLD_DAYS = 7

_SECURITY_SETTINGS = None


def _get_security_settings() -> Dict[str, Any]:
    """Return cached security settings from config."""
    global _SECURITY_SETTINGS
    if _SECURITY_SETTINGS is None:
        security_cfg = _CONFIG.get('security', {}) if _CONFIG else {}
        _SECURITY_SETTINGS = {
            'allow_comments': security_cfg.get('allow_comments', True),
            'allow_union': security_cfg.get('allow_union', True),
            'compat_legacy_validator': security_cfg.get('compat_legacy_validator', False),
        }
    return _SECURITY_SETTINGS


def _semantic_validation_enabled() -> Optional[bool]:
    if _SEMANTIC_VALIDATION_TOGGLE is None:
        return None
    return bool(_SEMANTIC_VALIDATION_TOGGLE)

# Global cache instance
_query_cache = None

def get_query_cache(config: Dict[str, Any]) -> Optional['QueryCache']:
    """Get or create query cache instance based on configuration."""
    global _query_cache
    if _query_cache is None and CACHE_AVAILABLE:
        cache_enabled = config.get('query', {}).get('cache_enabled', False)
        if cache_enabled:
            ttl = config.get('query', {}).get('cache_ttl_seconds', 300)
            max_entries = config.get('query', {}).get('cache_max_entries', 100)
            _query_cache = QueryCache(ttl_seconds=ttl, max_entries=max_entries)
    return _query_cache


def build_cache_metadata(
    tickers: Sequence[str],
    freshness_info: Optional[Mapping[str, Any]] = None,
    freshness_status: str = "ok",
) -> Dict[str, Any]:
    """Build metadata describing which tickers a cached entry depends on."""

    metadata: Dict[str, Any] = {
        'tickers': list(tickers) if tickers else [],
    }

    timestamp_map: Dict[str, Any] = {}
    if freshness_info:
        for ticker in tickers:
            info = freshness_info.get(ticker)
            if not info:
                continue
            ts = info.get('last_fetched') or info.get('timestamp')
            if ts:
                timestamp_map[ticker] = ts

    if timestamp_map:
        metadata['last_ingest_ts'] = timestamp_map
    else:
        metadata['last_ingest_ts'] = datetime.now(UTC).isoformat()

    metadata['freshness_status'] = freshness_status

    return metadata


def invalidate_cache_for_tickers(tickers: Sequence[str]) -> int:
    """Invalidate cached queries referencing the provided tickers."""

    if not tickers or not CACHE_AVAILABLE:
        return 0

    config = _CONFIG or load_config()
    cache = get_query_cache(config) if config else None
    if not cache:
        return 0
    return cache.invalidate(tickers=tickers)


# Schema caching with refresh detection

_schema_cache = None
_schema_hash = None


def get_schema_hash(schema: Mapping[str, Sequence[str]]) -> str:
    """Generate hash of current DuckDB schema."""
    schema_str = json.dumps({k: list(v) for k, v in schema.items()}, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()


def get_cached_schema(conn: duckdb.DuckDBPyConnection, allowed_tables: Sequence[str], force_refresh: bool = False) -> Mapping[str, Sequence[str]]:
    """Get schema with automatic refresh detection."""
    global _schema_cache, _schema_hash

    # Introspect current schema
    current_schema = introspect_schema(conn, allowed_tables)
    current_hash = get_schema_hash(current_schema)

    # Refresh if hash changed or forced
    if force_refresh or _schema_hash != current_hash:
        if _schema_hash and _schema_hash != current_hash:
            logger = logging.getLogger("query")
            logger.info("Schema changed, refreshing cache...")
        _schema_cache = current_schema
        _schema_hash = current_hash

    return _schema_cache if _schema_cache else current_schema


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
    # Phase 9: Analyst Intelligence
    "analyst.recommendations",
    "analyst.price_targets",
    "analyst.consensus",
    "analyst.growth_estimates",
    # Phase 10: Technical Analysis
    "technical.indicators",
)
SUMMARY_SAMPLE_LIMIT = 25
SUMMARY_SAMPLE_LIMIT = 25


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


class FreshnessConfig:
    def __init__(self) -> None:
        config = load_config()
        raw = config.get('freshness', {}) if hasattr(config, 'get') else {}
        self.max_batch_size = int(raw.get('max_batch_size', 10))
        self.timeout_ms = int(raw.get('timeout_ms', 2000))
        self.max_retries = int(raw.get('max_retries', 1))
        self.threshold_days = raw.get('staleness_threshold_days', DEFAULT_STALENESS_THRESHOLD_DAYS)


FRESHNESS_CFG = FreshnessConfig()


def check_data_freshness(
    mongo_db: Optional[Database],
    tickers: List[str],
    threshold_days: int = DEFAULT_STALENESS_THRESHOLD_DAYS,
) -> Dict[str, Any]:
    """Check if data for the given tickers is stale with bounded batching."""
    if not mongo_db or not tickers:
        return {"is_stale": False, "stale_tickers": [], "freshness_info": {}, "status": "skipped"}

    metadata_collection = mongo_db["ingestion_metadata"]
    stale_tickers: List[str] = []
    freshness_info: Dict[str, str] = {}
    status = "ok"

    def fetch_single(ticker: str) -> Dict[str, Any]:
        return metadata_collection.find_one({"ticker": ticker}, sort=[("last_fetched", -1)])

    batches = [tickers[i:i + FRESHNESS_CFG.max_batch_size] for i in range(0, len(tickers), FRESHNESS_CFG.max_batch_size)]

    for batch in batches:
        attempts = 0
        delays = jittered_backoff_delays(0.25, 2.0, FRESHNESS_CFG.max_retries)
        while attempts <= FRESHNESS_CFG.max_retries:
            timeout = FRESHNESS_CFG.timeout_ms / 1000
            results, errors = bounded_parallel_map(fetch_single, batch, max_workers=min(len(batch), 4), timeout=timeout)
            if errors and attempts < FRESHNESS_CFG.max_retries:
                time.sleep(delays[attempts] if attempts < len(delays) else 0)
                attempts += 1
                continue

            if errors:
                status = "unknown"
                for ticker, _ in errors:
                    freshness_info[ticker] = "unknown"
                break

            for ticker, doc in results:
                if not doc:
                    stale_tickers.append(ticker)
                    freshness_info[ticker] = "never fetched"
                    continue
                last_fetched_str = doc.get("last_fetched")
                if not last_fetched_str:
                    stale_tickers.append(ticker)
                    freshness_info[ticker] = "unknown"
                    continue
                last_fetched = parse_utc_timestamp(last_fetched_str)
                age = datetime.now(UTC) - last_fetched
                days_old = age.days
                if days_old >= threshold_days:
                    stale_tickers.append(ticker)
                freshness_info[ticker] = f"{days_old} days old"
            break

    return {
        "is_stale": len(stale_tickers) > 0,
        "stale_tickers": stale_tickers,
        "freshness_info": freshness_info,
        "status": status,
    }



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
    """Builds an enhanced system prompt with detailed schema and query guidance."""
    
    schema_lines = []
    for table, columns in schema.items():
        column_text = ", ".join(columns)
        schema_lines.append(f"- {table}: {column_text}")
    schema_block = "\n".join(schema_lines)

    # Enhanced table and column descriptions
    detailed_descriptions = """
**Detailed Table Descriptions:**

*   **`company.metadata`**: Core company information.
    *   `longName`: The full, official name of the company. **Use this for company names in results.**
    *   `sector`: Broad economic sector (e.g., 'Technology', 'Healthcare').
    *   `industry`: Specific industry within a sector (e.g., 'Software - Infrastructure').
    *   `marketCap`: Total market value of the company's outstanding shares.
    *   `country`: The country where the company is headquartered.

*   **`financials.annual` / `financials.quarterly`**: Income statement, balance sheet, and cash flow data.
    *   `totalRevenue`: The company's total sales over a period.
    *   `netIncome`: The company's profit after all expenses and taxes.
    *   `totalAssets`: Everything the company owns.
    *   `totalDebt`: All of the company's short and long-term debt.

*   **`prices.daily`**: Daily stock price data.
    *   `close`: The closing price of the stock for the day.
    *   `volume`: The number of shares traded during the day.

*   **`ratios.financial`**: Pre-calculated financial ratios.
    *   `roe`: Return on Equity (Net Income / Shareholder Equity). Measures profitability relative to equity.
    *   `roa`: Return on Assets (Net Income / Total Assets). Measures profitability relative to assets.
    *   `net_margin`: Net Profit Margin (Net Income / Revenue).
    *   `pe_ratio`: Price-to-Earnings ratio.

*   **`analyst.recommendations`**: Analyst ratings changes.
    *   `action`: The action taken by the analyst (e.g., 'up', 'down', 'init').
    *   `firm`: The name of the analyst's firm.

*   **`technical.indicators`**: Technical analysis data.
    *   `sma_50` / `sma_200`: 50-day and 200-day Simple Moving Averages. A "golden cross" is `sma_50 > sma_200`.
    *   `rsi_14`: 14-day Relative Strength Index. < 30 is often considered oversold, > 70 is overbought.
"""

    # Date context for natural language parsing
    today = date.today()
    one_year_ago = today - timedelta(days=365)
    five_years_ago = today - timedelta(days=365*5)

    date_context = f"""
**Date Context (Today is {today.isoformat()}):**

*   "last year" or "past year" → `WHERE date >= '{one_year_ago.isoformat()}'`
*   "last 5 years" → `WHERE date >= '{five_years_ago.isoformat()}'`
*   "recent" or "latest" → `ORDER BY date DESC LIMIT 1`
*   "in 2023" → `WHERE YEAR(date) = 2023`
*   "year to date" or "YTD" → `WHERE YEAR(date) = {today.year}`
"""

    # Common query patterns and examples
    query_patterns = """
**Common Query Patterns & Examples:**

1.  **Find companies in a specific sector with certain criteria:**
    *   *User Query:* "Find tech stocks with P/E under 20"
    *   *SQL:*
        ```sql
        SELECT
          m.ticker,
          m.longName,
          v.pe_ratio
        FROM valuation.metrics AS v
        JOIN company.metadata AS m
          ON v.ticker = m.ticker
        WHERE
          m.sector = 'Technology' AND v.pe_ratio < 20
        ORDER BY
          v.pe_ratio ASC
        LIMIT 25
        ```

2.  **Compare a metric across a peer group:**
    *   *User Query:* "Compare revenue growth for FAANG stocks"
    *   *SQL:*
        ```sql
        SELECT
          g.ticker,
          g.date,
          g.revenue_growth_annual
        FROM growth.annual AS g
        JOIN company.peers AS p
          ON g.ticker = p.ticker
        WHERE
          p.peer_group = 'FAANG' AND g.date >= (current_date - INTERVAL '5' YEAR)
        ORDER BY
          g.ticker,
          g.date
        ```

3.  **Rank companies by a metric:**
    *   *User Query:* "What are the top 10 companies by market cap?"
    *   *SQL:*
        ```sql
        SELECT
          ticker,
          longName,
          marketCap
        FROM company.metadata
        ORDER BY
          marketCap DESC
        LIMIT 10
        ```
"""

    # Rules and common pitfalls
    rules_and_pitfalls = """
**SQL Generation Rules & Common Pitfalls:**

*   **CRITICAL:** Your response MUST ONLY contain the SQL query wrapped in ```sql``` fences. No explanations, no commentary, no example output.
*   **Always use table aliases** for clarity (e.g., `FROM company.metadata AS m`).
*   **NEVER use `ILIKE`**. For case-insensitive matching, use `lower(column_name) = 'value'`.
*   **Column `longName` in `company.metadata` contains the company's name.** Do not use `name` or `company_name`.
*   When a user asks for "growth", look for pre-calculated growth fields like `revenue_growth_annual` in the `growth.annual` table first.
*   For rankings, use `ORDER BY` and `LIMIT`. For more complex rankings, use window functions like `RANK()`.
*   Default to a `LIMIT` of 25 if the user doesn't specify a number. The maximum `LIMIT` is 100.
"""

    return (
        "You are an expert DuckDB SQL query generator for financial data analysis.\n\n"
        "Your task is to convert a user's natural language question into a single, valid DuckDB SQL query.\n"
        "You must adhere to all the rules and guidelines provided.\n\n"
        "════════════════════════════════════════════════════════════════\n"
        f"**DATABASE SCHEMA (DuckDB):**\n{schema_block}\n\n"
        f"{detailed_descriptions}\n"
        f"{date_context}\n"
        f"{query_patterns}\n"
        f"{rules_and_pitfalls}"
        "════════════════════════════════════════════════════════════════\n\n"
        "REMINDER: Output ONLY the SQL query in ```sql``` fences. Nothing else.\n"
    )


def augment_question_with_hints(question: str) -> str:
    """Augment user question with helpful hints AND enforce SQL-only response."""
    hints: List[str] = []
    lowered = question.lower()

    # More specific hints based on keywords
    if any(keyword in lowered for keyword in ("analyst", "price target", "recommendation")):
        hints.append("For analyst ratings or price targets, join `analyst.price_targets` with `analyst.consensus` on the ticker.")
    if "golden cross" in lowered:
        hints.append("A 'golden cross' occurs when `sma_50` is greater than `sma_200` in the `technical.indicators` table.")
    if "oversold" in lowered or "rsi" in lowered:
        hints.append("Check for `rsi_14 < 30` in the `technical.indicators` table for oversold conditions.")
    if "undervalued" in lowered or "p/e" in lowered:
        hints.append("For valuation questions, use `valuation.metrics` and look at `pe_ratio`, `pb_ratio`, etc. A low P/E ratio can indicate an undervalued stock.")

    # Always add critical reminders to user message
    reminder = (
        "\n\n"
        "IMPORTANT REMINDERS:\n"
        "1. Output ONLY SQL code wrapped in ```sql``` fences.\n"
        "2. Use `company.metadata` for company info (the column is `longName`).\n"
        "3. Join tables on the `ticker` column.\n"
        "4. Use table aliases for readability.\n"
        "5. NO explanations, tables, or prose - ONLY SQL code."
    )

    if not hints:
        return f"{question}{reminder}"

    helper_block = "\n".join(f"- {hint}" for hint in hints)
    return f"{question}\n\n**Helpful Hints:**\n{helper_block}{reminder}"


def check_ollama_health(base_url: str, timeout: int = 5) -> bool:
    """Check if Ollama service is reachable."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


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


def call_ollama_chat_with_retry(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
    backoff: list = None,
    timeout: int = 60
) -> str:
    """Call Ollama chat API with exponential backoff on transient failures."""
    if backoff is None:
        backoff = [1, 2, 4]

    logger = logging.getLogger("chat")

    for attempt, delay in enumerate(backoff[:max_retries], start=1):
        try:
            return call_ollama_chat(base_url, model, messages, timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries:
                raise
            logger.warning(f"Ollama call failed (attempt {attempt}/{max_retries}), "
                          f"retrying in {delay}s: {e}")
            time.sleep(delay)


def call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_query: str,
    timeout: int = 60,
) -> str:
    """Call Ollama API with comprehensive error handling."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        # Validate response structure
        data = response.json()
        if not data.get("message"):
            raise OllamaResponseError("Missing 'message' field in response")

        message = data.get("message") or {}
        content = message.get("content")

        if not content or not content.strip():
            raise OllamaResponseError("Empty content in response")

        return content

    except requests.ConnectionError as e:
        raise OllamaConnectionError(f"Cannot reach Ollama at {base_url}: {e}") from e
    except requests.Timeout as e:
        raise OllamaTimeoutError(f"Ollama timeout after {timeout}s: {e}") from e
    except requests.HTTPError as e:
        raise OllamaResponseError(f"Ollama HTTP error: {e}") from e
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise OllamaResponseError(f"Malformed Ollama response: {e}") from e


def call_ollama_with_retry(
    base_url: str,
    model: str,
    system_prompt: str,
    user_query: str,
    max_retries: int = 3,
    backoff: Optional[List[int]] = None,
    timeout: int = 60
) -> str:
    """Call Ollama with exponential backoff on transient failures."""
    if backoff is None:
        backoff = [1, 2, 4]

    logger = logging.getLogger("query")

    for attempt, delay in enumerate(backoff[:max_retries], start=1):
        try:
            return call_ollama(base_url, model, system_prompt, user_query, timeout)
        except (OllamaConnectionError, OllamaTimeoutError) as e:
            if attempt == max_retries:
                raise
            logger.warning(f"Ollama call failed (attempt {attempt}/{max_retries}), "
                          f"retrying in {delay}s: {e}")
            time.sleep(delay)
        except OllamaResponseError:
            # Non-retriable errors - fail immediately
            raise


def extract_sql(text: str) -> str:
    """Extract SQL from LLM response with multiple fallback strategies."""
    if not text or not text.strip():
        raise SQLExtractionError("LLM response is empty")

    # Strategy 1: Code block with sql marker
    code_block = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        sql = code_block.group(1).strip()
        if sql:
            return sql

    # Strategy 2: Generic code block with SELECT/WITH
    code_block = re.search(r"```\s*((SELECT|WITH)\b.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        sql = code_block.group(1).strip()
        if sql:
            return sql

    # Strategy 3: SELECT statement anywhere in text (greedy match until semicolon, LIMIT, or end)
    select_match = re.search(
        r"(SELECT\b.*?)(?:;|\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if select_match:
        sql = select_match.group(1).strip()
        # Make sure we got actual SQL, not just the word SELECT in prose
        if sql and len(sql) > 20 and 'FROM' in sql.upper():
            return sql

    # Strategy 4: Look for WITH clause (CTE)
    with_match = re.search(
        r"(WITH\b.*?SELECT\b.*?)(?:;|\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if with_match:
        sql = with_match.group(1).strip()
        if sql and 'FROM' in sql.upper():
            return sql

    # Strategy 5: Look for SQL keywords in sequence (last resort)
    # Match: SELECT ... FROM ... (optional WHERE/ORDER BY/LIMIT)
    sql_pattern = re.search(
        r"(SELECT\s+.+?\s+FROM\s+[\w.]+(?:\s+(?:JOIN|WHERE|GROUP BY|ORDER BY|LIMIT|HAVING).+?)*?)(?:;|\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if sql_pattern:
        sql = sql_pattern.group(1).strip()
        if sql:
            return sql

    # All strategies failed
    raise SQLExtractionError(
        f"Could not extract SQL from LLM response. "
        f"The LLM may have returned formatted results instead of SQL. "
        f"Response preview: {text[:300]}..."
    )


def validate_sql_semantics(sql: str, question: str) -> tuple[bool, str]:
    """
    Semantic validation: Does the SQL answer the question?

    Returns: (is_valid, reason)
    """
    sql_lower = sql.lower()
    question_lower = question.lower()

    # Check 1: Aggregation mismatch
    if any(word in question_lower for word in ['average', 'mean', 'avg']):
        if 'avg(' not in sql_lower:
            return False, "Question asks for average but SQL doesn't use AVG()"

    if any(word in question_lower for word in ['total', 'sum']):
        if 'sum(' not in sql_lower and 'count(' not in sql_lower:
            return False, "Question asks for total/sum but SQL doesn't aggregate"

    # Check 2: Time range mismatch
    if any(word in question_lower for word in ['last', 'recent', 'latest']):
        if 'order by' not in sql_lower or 'desc' not in sql_lower:
            return False, "Question asks for latest but SQL doesn't order DESC"

    # Check 3: Comparison mismatch
    if 'compare' in question_lower or ' vs ' in question_lower or ' versus ' in question_lower:
        # Should have multiple tickers or entities
        ticker_mentions = sql_lower.count('ticker')
        if ticker_mentions < 2 and 'in (' not in sql_lower:
            return False, "Question asks for comparison but SQL only queries one entity"

    # Check 4: Ranking mismatch
    if any(word in question_lower for word in ['top', 'highest', 'best', 'rank']):
        if 'order by' not in sql_lower or 'limit' not in sql_lower:
            return False, "Question asks for top/ranking but SQL doesn't order or limit"

    # Check 5: Count mismatch
    if any(word in question_lower for word in ['how many', 'count', 'number of']):
        if 'count(' not in sql_lower:
            return False, "Question asks for count but SQL doesn't use COUNT()"

    # Check 6: Growth/change mismatch
    if any(word in question_lower for word in ['growth', 'change', 'increase', 'decrease']):
        if 'lag(' not in sql_lower and 'lead(' not in sql_lower and 'yoy' not in sql_lower:
            return False, "Question asks for growth/change but SQL doesn't calculate it"

    aggregate_funcs = ("sum(", "avg(", "count(", "min(", "max(")
    uses_aggregate = any(func in sql_lower for func in aggregate_funcs)
    has_group_by = "group by" in sql_lower

    if uses_aggregate and not has_group_by:
        return False, "Aggregations require a GROUP BY clause for non-aggregated columns."

    if " join " in sql_lower:
        select_clause = extract_select_clause(sql)
        expressions = split_select_expressions(select_clause)
        ambiguous = []
        for expr in expressions:
            cleaned = expr.strip()
            if not cleaned or cleaned == "*":
                continue
            lowered = cleaned.lower()
            if any(func in lowered for func in aggregate_funcs):
                continue
            if " as " in lowered:
                cleaned = cleaned.split(" as ", 1)[0].strip()
            if "." not in cleaned and "(" not in cleaned:
                ambiguous.append(cleaned)
                break
        if ambiguous:
            return False, f"Column '{ambiguous[0]}' should be qualified with a table alias when joining tables."

    cast_matches = re.findall(r"cast\s*\([^)]*?as\s+([a-zA-Z0-9_]+)", sql_lower)
    if cast_matches:
        allowed_casts = {
            "double", "float", "decimal", "numeric", "bigint", "int", "integer",
            "smallint", "real", "varchar", "text", "date", "timestamp", "boolean"
        }
        for target in cast_matches:
            if target not in allowed_casts:
                return False, f"Casting to '{target}' is not supported in safe mode."

    return True, "OK"


def validate_sql(
    sql: str,
    schema: Mapping[str, Sequence[str]],
    default_limit: Optional[int] = None,
    max_limit: Optional[int] = None,
    question: Optional[str] = None,
) -> str:
    if not sql:
        raise ValueError("SQL cannot be empty.")

    security_settings = _get_security_settings()
    allow_comments = security_settings['allow_comments']
    allow_union = security_settings['allow_union']
    compat_legacy = security_settings['compat_legacy_validator']

    effective_default_limit = default_limit if default_limit is not None else (
        LEGACY_DEFAULT_LIMIT if compat_legacy else DEFAULT_LIMIT
    )
    effective_max_limit = max_limit if max_limit is not None else (
        LEGACY_MAX_LIMIT if compat_legacy else MAX_LIMIT
    )

    if compat_legacy:
        allow_comments = True
        allow_union = True

    raw_sql = sql.strip()
    if not allow_comments:
        if re.search(r'(--|/\*)', raw_sql):
            raise ValueError("SQL comments are disabled by policy.")

    if not allow_union and re.search(r'\bunion\b', raw_sql, re.IGNORECASE):
        raise ValueError("UNION queries are not permitted.")

    # Strip SQL comments when allowed (LLM-generated comments are safe after extraction)
    sql_no_comments = re.sub(r'--[^\n]*', '', raw_sql)
    sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)

    cleaned = re.sub(r"\s+", " ", sql_no_comments.strip())
    cleaned_lower = cleaned.lower()
    if cleaned_lower.count(";") > 1 or (";" in cleaned[:-1] and not cleaned.endswith(";")):
        raise ValueError("Only single-statement SQL is allowed (semicolon detected).")
    statement_start = cleaned_lower.lstrip()
    if not statement_start.startswith(("select", "with")):
        raise ValueError("Only SELECT/CTE statements are permitted.")
    main_select_idx = find_main_select_index(cleaned)
    if main_select_idx == -1:
        raise ValueError("Only SELECT statements are permitted.")
    cte_names = extract_cte_names(cleaned, main_select_idx)
    # Phase 3: Enhanced SQL injection prevention
    # Block dangerous keywords (case-insensitive, accounts for obfuscation)
    dangerous_patterns = [
        r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b',
        r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b', r'\bGRANT\b',
        r'\bREVOKE\b', r'\bEXECUTE\b', r'\bEXEC\b', r'\bREPLACE\b',
        r'\bXP_\b', r'\bSP_\b',  # SQL Server stored procedures
        r'\bUNION\b.*\bSELECT\b',  # UNION injection
        r'\bINTO\s+OUTFILE\b',  # File operations
        r'\bLOAD_FILE\b',  # MySQL file operations
        r'@@', r'CHAR\(', r'CHR\(',  # Obfuscation techniques
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValueError(f"Disallowed SQL pattern detected: {pattern}")
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
        if value > effective_max_limit:
            cleaned = re.sub(
                r"(?i)\blimit\s+\d+\b",
                f"LIMIT {effective_max_limit}",
                cleaned,
                count=1,
            )
    else:
        cleaned = f"{cleaned} LIMIT {effective_default_limit}"

    semantic_toggle = _semantic_validation_enabled()
    should_run_semantics = (
        question
        and not compat_legacy
        and (semantic_toggle if semantic_toggle is not None else True)
    )

    if should_run_semantics:
        is_valid, reason = validate_sql_semantics(cleaned, question)
        if not is_valid:
            raise SemanticValidationError(reason)

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
    divider = "- + -".join("-" * width for width in widths)
    print(header)
    print(divider)
    for row in rows:
        line = " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        print(line)


def serialise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        try:
            return float(value)
        except Exception:
            return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def prepare_summary_rows(columns: Sequence[str], rows: Sequence[Sequence[Any]], limit: int = SUMMARY_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    sample = rows[:limit]
    formatted: List[Dict[str, Any]] = []
    for row in sample:
        record = {}
        for column, value in zip(columns, row):
            record[column] = serialise_value(value)
        formatted.append(record)
    return formatted


def generate_result_summary(
    base_url: str,
    model: str,
    question: str,
    sql: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    logger: logging.Logger,
) -> Optional[str]:
    total_rows = len(rows)
    if total_rows == 0:
        return (
            f"No matching records were found for '{question}'. "
            "Consider widening the date range or ensuring the ticker exists in the dataset."
        )
    sample_rows = prepare_summary_rows(columns, rows)
    payload = {
        "question": question,
        "sql": sql,
        "row_count": total_rows,
        "columns": columns,
        "sample_rows": sample_rows,
    }
    summary_system_prompt = (
        "You are FinanGPT's reporting analyst. Craft a concise natural-language answer (<=120 words) that directly "
        "addresses the user's finance question using the provided SQL results. Highlight the most relevant metrics, "
        "call out notable leaders/laggards, and mention the time coverage. If the sample is partial, state that the "
        "insights are based on the returned subset."
    )
    try:
        response = call_ollama(base_url, model, summary_system_prompt, json.dumps(payload, default=str))
        stripped = response.strip()
        if stripped:
            return stripped
    except Exception as exc:
        log_event(logger, phase="query.summary_error", error=str(exc))
    return _fallback_summary(question, columns, rows)


def _fallback_summary(question: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    total_rows = len(rows)
    if total_rows == 0:
        return f"No matching rows were found for '{question}'."
    sample = rows[0]
    details = ", ".join(
        f"{col}={serialise_value(val)}"
        for col, val in zip(columns, sample)
        if col.lower() in {"ticker", "date", "netincome", "totalrevenue", "pe_ratio", "eps_actual", "consensus_label", "upside_pct"}
    )
    if not details:
        details = ", ".join(f"{col}={serialise_value(val)}" for col, val in zip(columns, sample)[:4])
    return (
        f"Returned {total_rows} rows. Latest row snapshot: {details}. "
        "Review the table above for full details."
    )


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

    conn = duckdb.connect(str(get_duckdb_path()))
    schema = introspect_schema(conn, ALLOWED_TABLES)
    if not schema:
        conn.close()
        raise SystemExit("No DuckDB tables found. Run transform.py first.")

    logger = configure_logger("query")

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

            summary_text = generate_result_summary(base_url, model, f"Template: {args.template}", sql or "", columns, rows, logger)
            if summary_text:
                print("\nSummary:")
                print(summary_text.strip())

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
        # Check Ollama health before attempting
        if not check_ollama_health(base_url):
            logger.warning("Ollama health check failed, attempting graceful degradation")
            if RESILIENCE_AVAILABLE:
                # Try graceful degradation
                conn_err = requests.ConnectionError("Ollama health check failed")
                sql = handle_ollama_failure(conn_err)
                if not sql:
                    conn.close()
                    raise SystemExit("Ollama service is not available and no fallback template matched.")
            else:
                conn.close()
                raise SystemExit("Ollama service is not available.")

        # Try to call Ollama with retry logic
        hinted_question = augment_question_with_hints(question)
        try:
            response_text = call_ollama_with_retry(base_url, model, system_prompt, hinted_question)
            sql = extract_sql(response_text)

            if args.debug and RESILIENCE_AVAILABLE:
                print(f"\n[DEBUG] Hint-augmented question:\n{hinted_question}\n")
                print(f"[DEBUG] LLM Response:\n{response_text}\n")
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

        try:
            sanitised_sql = validate_sql(sql, schema, question=question)
        except SemanticValidationError as exc:
            log_event(logger, phase="query.semantic_validation_failed", error=str(exc))
            conn.close()
            raise SystemExit(f"Semantic validation failed: {exc}") from exc

        tickers = extract_tickers_from_sql(sanitised_sql)
        freshness_details: Dict[str, Any] = {}

        if args.debug:
            print(f"[DEBUG] Validated SQL:\n{sanitised_sql}\n")

        # Check data freshness before executing
        freshness_status = "skipped"
        if mongo_db and not args.skip_freshness_check and tickers:
            freshness = check_data_freshness(mongo_db, tickers)
            freshness_details = freshness.get("freshness_info", {})
            freshness_status = freshness.get("status", "ok")
            if freshness_status == "unknown":
                logger.warning("Freshness check timed out; proceeding with query.")
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
        else:
            freshness_details = {}
            freshness_status = "skipped"

        # Try to load config for cache
        config = {}
        if CACHE_AVAILABLE:
            try:
                config = load_config()
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        # Check cache first
        cache = get_query_cache(config)
        cached_df = None
        if cache:
            cached_df = cache.get(sanitised_sql)

        # Execute query with timing
        start_time = time.time()

        if cached_df is not None:
            # Cache hit - convert DataFrame back to columns and rows
            df = cached_df
            columns = list(df.columns)
            rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
            query_time = time.time() - start_time
            if args.debug:
                print("🚀 Cache hit! Query returned instantly from cache.")
            if cache:
                cache.record_query_latency(query_time)
        else:
            # Cache miss - execute query
            result = conn.execute(sanitised_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            query_time = time.time() - start_time
            df = pd.DataFrame(rows, columns=columns)

            # Cache the result
            if cache and not df.empty:
                metadata = build_cache_metadata(tickers, freshness_details, freshness_status)
                cache.set(sanitised_sql, df, metadata=metadata)
                if args.debug:
                    print("💾 Query result cached for future use.")
            if cache:
                cache.record_query_latency(query_time)

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

        summary_text = generate_result_summary(base_url, model, question, sanitised_sql, columns, rows, logger)
        if summary_text:
            print("\nSummary:")
            print(summary_text.strip())

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
