"""
Application-wide constants for FinanGPT.

This module centralizes magic numbers, default values, and configuration
constants to improve code maintainability and reduce duplication.

Author: FinanGPT Enhancement Plan 4 Phase 4
Created: 2025-11-10
"""

# ============================================================================
# Database Configuration
# ============================================================================

# DuckDB
DEFAULT_DUCKDB_PATH = "financial_data.duckdb"

# MongoDB Collections
COLLECTION_RAW_ANNUAL = "raw_annual"
COLLECTION_RAW_QUARTERLY = "raw_quarterly"
COLLECTION_PRICES_DAILY = "stock_prices_daily"
COLLECTION_DIVIDENDS = "dividends_history"
COLLECTION_SPLITS = "splits_history"
COLLECTION_METADATA = "company_metadata"
COLLECTION_INGESTION_METADATA = "ingestion_metadata"
COLLECTION_EARNINGS_HISTORY = "earnings_history"
COLLECTION_EARNINGS_CALENDAR = "earnings_calendar"
COLLECTION_ANALYST_RECOMMENDATIONS = "analyst_recommendations"
COLLECTION_PRICE_TARGETS = "price_targets"
COLLECTION_ANALYST_CONSENSUS = "analyst_consensus"
COLLECTION_GROWTH_ESTIMATES = "growth_estimates"

# DuckDB Schemas
SCHEMA_FINANCIALS = "financials"
SCHEMA_PRICES = "prices"
SCHEMA_DIVIDENDS = "dividends"
SCHEMA_SPLITS = "splits"
SCHEMA_COMPANY = "company"
SCHEMA_RATIOS = "ratios"
SCHEMA_GROWTH = "growth"
SCHEMA_USER = "user"
SCHEMA_VALUATION = "valuation"
SCHEMA_EARNINGS = "earnings"
SCHEMA_ANALYST = "analyst"
SCHEMA_TECHNICAL = "technical"

# DuckDB Tables (with schema prefixes)
TABLE_FINANCIALS_ANNUAL = "financials.annual"
TABLE_FINANCIALS_QUARTERLY = "financials.quarterly"
TABLE_PRICES_DAILY = "prices.daily"
TABLE_DIVIDENDS_HISTORY = "dividends.history"
TABLE_SPLITS_HISTORY = "splits.history"
TABLE_COMPANY_METADATA = "company.metadata"
TABLE_COMPANY_PEERS = "company.peers"
TABLE_RATIOS_FINANCIAL = "ratios.financial"
TABLE_GROWTH_ANNUAL = "growth.annual"
TABLE_USER_PORTFOLIOS = "user.portfolios"
TABLE_VALUATION_METRICS = "valuation.metrics"
TABLE_EARNINGS_HISTORY = "earnings.history"
TABLE_EARNINGS_CALENDAR = "earnings.calendar"
TABLE_ANALYST_RECOMMENDATIONS = "analyst.recommendations"
TABLE_ANALYST_PRICE_TARGETS = "analyst.price_targets"
TABLE_ANALYST_CONSENSUS = "analyst.consensus"
TABLE_ANALYST_GROWTH_ESTIMATES = "analyst.growth_estimates"
TABLE_TECHNICAL_INDICATORS = "technical.indicators"

# ============================================================================
# Query Limits & Thresholds
# ============================================================================

DEFAULT_QUERY_LIMIT = 50
MAX_QUERY_LIMIT = 1000
DEFAULT_STALENESS_THRESHOLD_DAYS = 7

# ============================================================================
# Ingestion Configuration
# ============================================================================

MAX_TICKERS_PER_RUN = 50
MAX_INGESTION_ATTEMPTS = 3
PRICE_LOOKBACK_DAYS = 365
DEFAULT_RETRY_BACKOFF = [1, 2, 4]  # Seconds

# Supported equity types
ALLOWED_EQUITY_TYPES = {
    "EQUITY",
    "COMMON STOCK",
    "COMMONSTOCK",
    "STOCK",
    "PREFERRED STOCK",
    "ETF",  # For detection/rejection
}

# ============================================================================
# Currency Support
# ============================================================================

BASE_CURRENCY = "USD"

SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CNY",
    "CAD", "AUD", "CHF", "HKD", "SGD",
    "KRW", "INR"
]

# ============================================================================
# Ollama Configuration
# ============================================================================

DEFAULT_OLLAMA_TIMEOUT = 60  # Seconds
DEFAULT_OLLAMA_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_REQUESTS = 10
DEFAULT_RATE_LIMIT_WINDOW = 60  # Seconds
DEFAULT_MAX_CONTEXT_TOKENS = 4000

# ============================================================================
# Cache Configuration
# ============================================================================

DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_CACHE_MAX_ENTRIES = 100

# ============================================================================
# Security Configuration
# ============================================================================

MAX_TICKER_LENGTH = 10
ALLOWED_TICKER_PATTERN = r'^[A-Z0-9.\-]+$'

# Default allowed directories for file operations
DEFAULT_ALLOWED_DIRECTORIES = [".", "charts", "logs"]

# ============================================================================
# Performance Configuration
# ============================================================================

DEFAULT_CONCURRENT_WORKERS = 10
DEFAULT_WORKER_TIMEOUT = 120  # Seconds
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MAX_MEMORY_MB = 2048

# ============================================================================
# Logging Configuration
# ============================================================================

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"

# ============================================================================
# Visualization Configuration
# ============================================================================

DEFAULT_CHART_OUTPUT_DIR = "charts"
SUPPORTED_EXPORT_FORMATS = ["csv", "json", "excel", "parquet"]

# ============================================================================
# Validation Patterns (SQL Injection Prevention)
# ============================================================================

DANGEROUS_SQL_PATTERNS = [
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

# ============================================================================
# Chat Configuration
# ============================================================================

MAX_CHAT_HISTORY_LENGTH = 20
DEFAULT_PRESERVE_RECENT_MESSAGES = 5

# Example queries for help
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

# ============================================================================
# Error Messages (Production-Safe)
# ============================================================================

GENERIC_ERROR_MESSAGES = {
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

# ============================================================================
# Application Metadata
# ============================================================================

APP_NAME = "FinanGPT"
APP_VERSION = "2.8"
APP_DESCRIPTION = "AI-Powered Financial Data Analysis Platform"
