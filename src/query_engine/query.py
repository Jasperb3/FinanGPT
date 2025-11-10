#!/usr/bin/env python3
"""LLM-to-SQL query runner with DuckDB guardrails."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime, timedelta, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from src.core.time_utils import parse_utc_timestamp

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

import threading
from collections import deque


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
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

# Import query caching module
try:
    from src.query.cache import QueryCache
    from src.core.config_loader import load_config
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

LOGS_DIR = Path("logs")
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
DEFAULT_STALENESS_THRESHOLD_DAYS = 7

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


# Schema caching with refresh detection
import hashlib

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

            last_fetched = parse_utc_timestamp(last_fetched_str)
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
    schema_lines = []
    for table, columns in schema.items():
        column_text = ", ".join(columns)
        schema_lines.append(f"- {table}: {column_text}")
    schema_block = "\n".join(schema_lines)
    guidance: list[str] = []
    ratios_columns = schema.get("ratios.financial")
    if ratios_columns:
        key_ratios = [col for col in ratios_columns if col not in {"ticker", "date"}]
        if key_ratios:
            guidance.append(
                "Use `ratios.financial` for capital-efficiency metrics such as roe, roa, net_margin, gross_margin, "
                "ebitda_margin, debt_ratio, asset_turnover, fcf_margin, and cash_conversion."
            )
    if "company.peers" in schema:
        guidance.append("Join `company.peers` when the query references industries or peer groups (FAANG, SEMICONDUCTORS, etc.).")
    if any(table.startswith("analyst.") for table in schema.keys()):
        guidance.append("Only use `analyst.*` tables when the user explicitly requests analyst sentiment, targets, or forecasts.")
    table_guidance = ""
    if guidance:
        guidance = [f"- {line}" for line in guidance]
        table_guidance = "Table guidance:\n" + "\n".join(guidance) + "\n"

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

    # Phase 9: Analyst Intelligence information
    analyst_intelligence_info = """
Analyst Intelligence (Phase 9):

Analyst Recommendations (analyst.recommendations table):
Fields: firm, from_grade, to_grade, action, action_score
Actions: up (upgrade), down (downgrade), maintain, init (initiated), reit (reiterated)

Price Targets (analyst.price_targets table):
Fields: current_price, target_low, target_mean, target_high, upside_pct, downside_pct, max_upside_pct, num_analysts

Analyst Consensus (analyst.consensus table):
Fields: strong_buy, buy, hold, sell, strong_sell, total_analysts, consensus_rating, consensus_label
Consensus Labels: Strong Buy, Buy, Hold, Sell, Strong Sell

Growth Estimates (analyst.growth_estimates table):
Fields: current_qtr_growth, next_qtr_growth, current_year_growth, next_year_growth, next_5yr_growth, peg_forward

Examples:
- "Show me stocks with recent analyst upgrades" → SELECT * FROM analyst.recommendations WHERE action = 'up' ORDER BY date DESC
- "Find stocks with highest upside to price targets" → SELECT * FROM analyst.price_targets ORDER BY upside_pct DESC
- "Stocks rated Strong Buy with upside > 15%" → SELECT * FROM analyst.consensus c JOIN analyst.price_targets p ON c.ticker = p.ticker WHERE consensus_label = 'Strong Buy' AND upside_pct > 15
- "Companies with 5-year growth estimates > 20%" → SELECT * FROM analyst.growth_estimates WHERE next_5yr_growth > 20
"""

    # Phase 10: Technical Analysis information
    technical_analysis_info = """
Technical Analysis (Phase 10 - technical.indicators table):

Moving Averages:
- sma_20, sma_50, sma_200: Simple moving averages
- ema_12, ema_26: Exponential moving averages

Momentum Indicators:
- rsi_14: Relative Strength Index (0-100, <30=oversold, >70=overbought)
- macd, macd_signal, macd_histogram: MACD indicator components

Volatility:
- bb_upper, bb_middle, bb_lower: Bollinger Bands (20-day, 2 std dev)

Volume:
- volume_avg_20: 20-day average volume
- volume_ratio: Current volume / 20-day average

Price Momentum:
- pct_change_1d, pct_change_5d, pct_change_20d, pct_change_60d, pct_change_252d: % price changes

52-Week Analysis:
- week_52_high, week_52_low: 52-week high and low prices
- pct_from_52w_high, pct_from_52w_low: Distance from 52-week extremes (%)

Examples:
- "Find stocks with golden cross (SMA50 > SMA200)" → SELECT * FROM technical.indicators WHERE sma_50 > sma_200
- "Show oversold stocks (RSI < 30)" → SELECT * FROM technical.indicators WHERE rsi_14 < 30
- "Stocks breaking above Bollinger upper band" → SELECT * FROM technical.indicators WHERE close > bb_upper
- "Positive MACD crossover" → SELECT * FROM technical.indicators WHERE macd > macd_signal AND macd_histogram > 0
- "Stocks near 52-week lows with high volume" → SELECT * FROM technical.indicators WHERE pct_from_52w_low < 5 AND volume_ratio > 2
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
        "Use ANSI interval arithmetic such as `current_date + INTERVAL '1' YEAR`; avoid `date_add` unless you pass (date, interval).",
        "Wrap the SQL inside ```sql``` fences and do not add commentary or explanations.",
    ]
    rules_block = "\n".join(f"- {rule}" for rule in rules)

    return (
        "You are FinanGPT, a disciplined financial data analyst that writes safe DuckDB SQL.\n"
        f"Schema snapshot:\n{schema_block}\n\n"
        f"{table_guidance}"
        f"{date_context}\n"
        f"{peer_groups_info}\n"
        f"{valuation_earnings_info}\n"
        f"{analyst_intelligence_info}\n"
        f"{technical_analysis_info}\n"
        f"{advanced_sql}\n"
        f"Rules:\n{rules_block}\n"
        "Output only SQL, optionally wrapped in ```sql``` fences."
    )


def check_ollama_health(base_url: str, timeout: int = 5) -> bool:
    """Check if Ollama service is reachable."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


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

    # Strategy 2: Generic code block
    code_block = re.search(r"```\s*(SELECT\b.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        sql = code_block.group(1).strip()
        if sql:
            return sql

    # Strategy 3: SELECT statement anywhere in text
    select_match = re.search(r"(SELECT\b.*?)(?:\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if select_match:
        sql = select_match.group(1).strip()
        if sql:
            return sql

    # Strategy 4: Look for WITH clause (CTE)
    with_match = re.search(r"(WITH\b.*?SELECT\b.*?)(?:\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if with_match:
        sql = with_match.group(1).strip()
        if sql:
            return sql

    # All strategies failed
    raise SQLExtractionError(
        f"Could not extract SQL from LLM response. "
        f"Response preview: {text[:200]}..."
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

    return True, "OK"


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
    # Phase 3: Enhanced SQL injection prevention
    # Block dangerous keywords (case-insensitive, accounts for obfuscation)
    dangerous_patterns = [
        r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b',
        r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b', r'\bGRANT\b',
        r'\bREVOKE\b', r'\bEXECUTE\b', r'\bEXEC\b', r'\bREPLACE\b',
        r'\bXP_\b', r'\bSP_\b',  # SQL Server stored procedures
        r'--',  # SQL comments (injection vector)
        r'/\*', r'\*/',  # Multi-line comments
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


def augment_question_with_hints(question: str) -> str:
    hints: List[str] = []
    lowered = question.lower()
    if any(keyword in lowered for keyword in ("analyst", "price target", "price targets", "recommendation", "consensus")):
        hints.append(
            "Use `analyst.price_targets` (current_price, target_low/mean/high, upside_pct) and `analyst.consensus` "
            "(strong_buy..strong_sell, consensus_label) filtered by the requested ticker. Limit to the next 12 months "
            "using `WHERE date BETWEEN current_date AND current_date + INTERVAL '12' MONTH`."
        )
    if any(keyword in lowered for keyword in ("roe", "return on equity", "roa", "margin")):
        hints.append(
            "For ROE/ROA/margin questions, pull metrics from `ratios.financial` instead of analyst tables."
        )
    if "peer" in lowered or "similar companies" in lowered:
        hints.append(
            "Use `company.peers` joined to `company.metadata` or valuation tables to compare companies within the same peer_group."
        )
    if any(keyword in lowered for keyword in ("p/e", "pe ratio", "price-to-earnings", "valuation")):
        hints.append(
            "Retrieve valuation ratios (pe_ratio, pb_ratio, ps_ratio, dividend_yield, peg_ratio) from `valuation.metrics`."
        )
    if any(keyword in lowered for keyword in ("net income", "revenue", "fiscal year", "annual results")):
        hints.append(
            "For net income or revenue questions, select `netIncome` and `totalRevenue` from `financials.annual` "
            "filtered by the specific ticker and order by date DESC LIMIT 1 to capture the latest fiscal year."
        )
    if any(keyword in lowered for keyword in ("eps", "earnings per share")):
        hints.append(
            "For EPS trends, query `earnings.history` with `eps_actual`, `report_date`, and window functions such as "
            "LAG() over the past N quarters (`WHERE ticker = '<TICKER>' AND report_date >= current_date - INTERVAL '2' YEAR`)."
        )
    if not hints:
        return question
    helper_block = "\n".join(f"- {hint}" for hint in hints)
    return f"{question}\n\nHelper Notes:\n{helper_block}"


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

        # Try to load config for cache
        config = {}
        if CACHE_AVAILABLE:
            try:
                config = load_config()
            except:
                pass

        # Check cache first
        cache = get_query_cache(config)
        cached_df = None
        if cache:
            cached_df = cache.get(sanitised_sql)

        # Execute query with timing
        start_time = time.time()

        if cached_df is not None:
            # Cache hit - convert DataFrame back to columns and rows
            columns = list(cached_df.columns)
            rows = [tuple(row) for row in cached_df.itertuples(index=False, name=None)]
            query_time = time.time() - start_time
            if args.debug:
                print("🚀 Cache hit! Query returned instantly from cache.")
        else:
            # Cache miss - execute query
            result = conn.execute(sanitised_sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            query_time = time.time() - start_time

            # Cache the result
            if cache and rows:
                df = pd.DataFrame(rows, columns=columns)
                cache.set(sanitised_sql, df)
                if args.debug:
                    print("💾 Query result cached for future use.")

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
