"""
Smart Error Handler Module

Transforms technical errors into user-friendly messages with helpful suggestions.

Author: FinanGPT Team
"""

import re
from typing import Optional, List, Tuple
from difflib import get_close_matches
import duckdb
from src.query_engine.query import SemanticValidationError


class SmartErrorHandler:
    """
    Enhanced error handler that provides user-friendly error messages
    with actionable suggestions.

    Features:
    - Table not found → Suggest similar table names
    - Column missing → Suggest similar column names
    - No data for ticker → Suggest similar tickers
    - Date range empty → Show available date range
    - Query too broad → Suggest filters
    """

    def __init__(self, db_connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize SmartErrorHandler.

        Args:
            db_connection: DuckDB connection for schema introspection
        """
        self.conn = db_connection
        self.allowed_tables = [
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
            "valuation.metrics",
            "earnings.history",
            "earnings.calendar",
            "analyst.recommendations",
            "analyst.price_targets",
            "analyst.consensus",
            "analyst.growth_estimates",
            "technical.indicators"
        ]

    def enhance_error(
        self,
        error: Exception,
        user_query: str,
        sql: Optional[str] = None
    ) -> str:
        """
        Transform technical error into helpful message.

        Args:
            error: Original exception
            user_query: User's natural language query
            sql: Generated SQL (if available)

        Returns:
            Enhanced error message
        """
        error_str = str(error)

        if isinstance(error, SemanticValidationError):
            return self._handle_semantic_error(str(error))

        # Table not found
        if "not on the allow-list" in error_str or "Table" in error_str and "not found" in error_str:
            return self._handle_table_not_found(error_str, sql)

        # Column not found
        if "column" in error_str.lower() and "not found" in error_str.lower():
            return self._handle_column_not_found(error_str, sql)

        # Ticker not found
        if "ticker" in user_query.lower() and ("no data" in error_str.lower() or "empty" in error_str.lower()):
            return self._handle_ticker_not_found(user_query)

        # Date range issues
        if "date" in error_str.lower() and ("empty" in error_str.lower() or "no data" in error_str.lower()):
            return self._handle_date_range_error(user_query)

        # Query too broad
        if "limit" in error_str.lower() or "too many" in error_str.lower():
            return self._handle_query_too_broad(user_query)

        # SQL syntax error
        if "syntax" in error_str.lower() or "parser" in error_str.lower():
            return self._handle_sql_syntax_error(error_str, user_query)

        # Division by zero
        if "division by zero" in error_str.lower():
            return self._handle_division_by_zero(user_query)

        # Default fallback
        return self._default_error_message(error_str, user_query)

    def _handle_table_not_found(self, error_str: str, sql: Optional[str]) -> str:
        """Handle table not found errors."""
        # Extract table name from error or SQL
        table_mentioned = self._extract_table_name(error_str, sql)

        if table_mentioned:
            similar = get_close_matches(table_mentioned, self.allowed_tables, n=3, cutoff=0.5)
        else:
            similar = []

        message = f"""
❌ I don't have a table called '{table_mentioned}'.

💡 Did you mean one of these?
"""
        if similar:
            for table in similar:
                desc = self._get_table_description(table)
                message += f"   • {table} - {desc}\n"
        else:
            message += "   • company.metadata - Company information\n"
            message += "   • financials.annual - Financial statements\n"
            message += "   • prices.daily - Stock prices\n"

        message += "\n💡 Try asking: 'Show me all companies' or 'List available tickers'"
        return message

    def _handle_column_not_found(self, error_str: str, sql: Optional[str]) -> str:
        """Handle column not found errors."""
        # Extract column name from error
        column_match = re.search(r"column ['\"]?(\w+)['\"]?", error_str, re.IGNORECASE)
        column_name = column_match.group(1) if column_match else "unknown"

        # Extract table name
        table_match = re.search(r"table ['\"]?([\w.]+)['\"]?", error_str, re.IGNORECASE)
        table_name = table_match.group(1) if table_match else None

        message = f"""
❌ Column '{column_name}' not found"""

        if table_name and self.conn:
            try:
                # Get available columns
                columns = self._get_table_columns(table_name)
                similar = get_close_matches(column_name, columns, n=3, cutoff=0.5)

                if similar:
                    message += f" in table '{table_name}'.\n\n💡 Did you mean:\n"
                    for col in similar:
                        message += f"   • {col}\n"
                else:
                    message += f" in table '{table_name}'.\n\n💡 Available columns:\n"
                    for col in columns[:10]:  # Show first 10
                        message += f"   • {col}\n"
                    if len(columns) > 10:
                        message += f"   ... and {len(columns) - 10} more\n"
            except:
                pass

        return message

    def _handle_ticker_not_found(self, user_query: str) -> str:
        """Handle ticker not found errors."""
        # Extract ticker from query
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', user_query)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"

        message = f"""
❌ Ticker '{ticker}' not found in database.

💡 Possible reasons:
   • Ticker might be misspelled
   • Company not yet ingested
   • Ticker might have changed (merger/acquisition)

💡 Try:
   • Check spelling: Use 'Search for companies with name like Apple'
   • Ingest ticker: Run 'python ingest.py --tickers {ticker}'
   • List available tickers: Ask 'Show me all available tickers'
"""
        return message

    def _handle_date_range_error(self, user_query: str) -> str:
        """Handle date range errors."""
        message = """
❌ No data found for the specified date range.

💡 Possible reasons:
   • Data not available for that period
   • Date format not recognized
   • Company not public during that time

💡 Try:
   • Check available date range: 'Show oldest and newest dates for <ticker>'
   • Use broader range: 'last 5 years' instead of specific dates
   • Ensure ticker was public: Check IPO date
"""
        return message

    def _handle_query_too_broad(self, user_query: str) -> str:
        """Handle query too broad errors."""
        message = """
❌ Query would return too many rows (exceeds LIMIT).

💡 Suggestions to narrow your query:
   • Add filters: Specify sector, industry, or ticker
   • Limit date range: 'last year' instead of 'all time'
   • Add TOP/LIMIT: 'top 10 companies by revenue'
   • Filter by metric: 'companies with P/E < 15'

💡 Examples:
   • 'Show top 10 tech stocks by market cap'
   • 'Show AAPL revenue for last 3 years'
   • 'Find companies in Technology sector with ROE > 20%'
"""
        return message

    def _handle_sql_syntax_error(self, error_str: str, user_query: str) -> str:
        """Handle SQL syntax errors."""
        message = f"""
❌ I generated invalid SQL for your query.

This is usually because:
   • Complex query structure
   • Ambiguous phrasing
   • Multiple requests in one query

💡 Try:
   • Rephrase more simply: Break into smaller questions
   • Be more specific: Specify exact columns/tables you need
   • Use templates: Try /list-templates for predefined queries

💡 Original error: {error_str[:100]}
"""
        return message

    def _handle_division_by_zero(self, user_query: str) -> str:
        """Handle division by zero errors."""
        message = """
❌ Division by zero encountered in calculation.

This often happens when:
   • Dividing by metrics that can be zero (e.g., revenue for some tickers)
   • Unprofitable companies (negative earnings)

💡 Try:
   • Add filter: 'WHERE <denominator> != 0'
   • Use NULLIF: Handles zero gracefully
   • Filter profitable only: 'WHERE netIncome > 0'
"""
        return message

    def _default_error_message(self, error_str: str, user_query: str) -> str:
        """Default error message when no specific handler matches."""
        message = f"""
❌ Query failed: {error_str[:200]}

💡 Suggestions:
   • Try rephrasing your question more simply
   • Use /help to see example queries
   • Use /list-templates to see predefined queries
   • Enable debug mode: --debug flag shows generated SQL

💡 Common fixes:
   • Check ticker spelling
   • Ensure date ranges are valid
   • Verify column names in error message
"""
        return message

    def _extract_table_name(self, error_str: str, sql: Optional[str]) -> Optional[str]:
        """Extract table name from error message or SQL."""
        # Try error string first
        match = re.search(r"table ['\"]?([\w.]+)['\"]?", error_str, re.IGNORECASE)
        if match:
            return match.group(1)

        # Try SQL
        if sql:
            match = re.search(r"FROM\s+([\w.]+)", sql, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _get_table_description(self, table: str) -> str:
        """Get human-readable description of table."""
        descriptions = {
            "financials.annual": "Annual financial statements",
            "financials.quarterly": "Quarterly financial statements",
            "prices.daily": "Daily stock prices (OHLCV)",
            "dividends.history": "Dividend payments",
            "splits.history": "Stock splits",
            "company.metadata": "Company information (sector, industry, employees)",
            "company.peers": "Peer group mappings",
            "ratios.financial": "Calculated financial ratios (ROE, ROA, margins)",
            "growth.annual": "Year-over-year growth metrics",
            "user.portfolios": "Portfolio holdings",
            "valuation.metrics": "Valuation ratios (P/E, P/B, dividend yield)",
            "earnings.history": "Historical earnings (estimates vs actuals)",
            "earnings.calendar": "Upcoming earnings dates",
            "analyst.recommendations": "Analyst upgrades/downgrades",
            "analyst.price_targets": "Price target consensus",
            "analyst.consensus": "Rating consensus (buy/hold/sell)",
            "analyst.growth_estimates": "Growth forecasts",
            "technical.indicators": "Technical analysis indicators"
        }
        return descriptions.get(table, "Unknown table")

    def _get_table_columns(self, table: str) -> List[str]:
        """Get list of columns for a table."""
        if not self.conn:
            return []

        try:
            result = self.conn.execute(f"DESCRIBE {table}").fetchall()
            return [row[0] for row in result]
        except:
            return []


def suggest_alternative_query(user_query: str, error: Exception) -> Optional[str]:
    """
    Suggest an alternative query based on the failed query.

    Args:
        user_query: Original query
        error: Error that occurred

    Returns:
        Suggested alternative query, or None
    """
    error_str = str(error).lower()

    # If table not found, suggest metadata query
    if "table" in error_str and "not found" in error_str:
        return "Show me all available companies"

    # If ticker issue, suggest search
    if "ticker" in user_query.lower():
        # Extract ticker
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', user_query)
        if ticker_match:
            ticker = ticker_match.group(1)
            return f"Search for companies with ticker like {ticker}"

    # If date issue, suggest broader range
    if "date" in error_str or "last" in user_query.lower():
        # Suggest "last 5 years" as safe default
        modified = re.sub(r'\b\d+\s+(year|month|day)s?\b', '5 years', user_query, flags=re.IGNORECASE)
        if modified != user_query:
            return modified

    return None
    def _handle_semantic_error(self, message: str) -> str:
        """Provide user-friendly semantic validation guidance."""

        return (
            f"❌ Semantic validation failed: {message}\n\n"
            "💡 Fix tips:\n"
            "   • Verify GROUP BY clauses cover all non-aggregated columns\n"
            "   • Qualify columns when joining tables (e.g., prices.ticker)\n"
            "   • Avoid casting to unsupported types\n"
        )
