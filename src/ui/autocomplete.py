"""
Autocomplete & Suggestion Engine Module

Provides contextual suggestions for tickers, query templates, and follow-up queries.

Author: FinanGPT Team
"""

from typing import List, Dict, Optional, Tuple
import re
import duckdb


class AutocompleteEngine:
    """
    Provides intelligent suggestions for user queries.

    Features:
    - Ticker autocomplete with company names
    - Query template suggestions
    - Follow-up query recommendations
    - Contextual help
    """

    def __init__(self, db_connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize AutocompleteEngine.

        Args:
            db_connection: DuckDB connection for ticker lookup
        """
        self.conn = db_connection
        self._init_query_templates()

    def _init_query_templates(self):
        """Initialize common query templates."""
        self.query_templates = {
            "compare": [
                "Compare revenue across multiple tickers",
                "Compare P/E ratios for peer group",
                "Compare sector performance",
                "Compare historical vs estimated earnings",
                "Compare technical indicators for stocks"
            ],
            "find": [
                "Find undervalued stocks with low P/E",
                "Find stocks with recent analyst upgrades",
                "Find high dividend yield stocks",
                "Find stocks with strong momentum (RSI)",
                "Find companies beating earnings estimates"
            ],
            "show": [
                "Show revenue trends over time",
                "Show profit margins by company",
                "Show analyst consensus ratings",
                "Show technical indicators (SMA, MACD)",
                "Show earnings surprise history"
            ],
            "top": [
                "Top 10 companies by revenue",
                "Top stocks by ROE (return on equity)",
                "Top gainers by price momentum",
                "Top rated stocks by analysts",
                "Top dividend paying stocks"
            ],
            "analyze": [
                "Analyze FAANG companies performance",
                "Analyze sector trends",
                "Analyze correlation between stocks",
                "Analyze portfolio performance",
                "Analyze earnings trends"
            ]
        }

        self.follow_up_templates = {
            "revenue": [
                "Show profit margins for same period",
                "Compare to sector average",
                "Show year-over-year growth",
                "Plot as chart"
            ],
            "price": [
                "Show technical indicators",
                "Compare to 52-week high/low",
                "Show analyst price targets",
                "Calculate returns over period"
            ],
            "earnings": [
                "Show earnings surprise percentage",
                "Compare estimate vs actual trend",
                "Show next earnings date",
                "Show analyst revisions"
            ],
            "valuation": [
                "Compare to industry average",
                "Show historical P/E trend",
                "Include dividend yield",
                "Show PEG ratio for growth context"
            ]
        }

    def suggest_tickers(
        self,
        partial: str,
        limit: int = 10,
        include_company_name: bool = True
    ) -> List[Dict[str, str]]:
        """
        Suggest ticker symbols based on partial input.

        Args:
            partial: Partial ticker or company name
            limit: Maximum suggestions
            include_company_name: Include company name in results

        Returns:
            List of suggestions with ticker and company name
        """
        if not self.conn or not partial:
            return []

        partial = partial.upper().strip()

        try:
            if include_company_name:
                query = """
                    SELECT DISTINCT ticker, longName, sector
                    FROM company.metadata
                    WHERE ticker LIKE ? OR UPPER(longName) LIKE ?
                    ORDER BY ticker
                    LIMIT ?
                """
                results = self.conn.execute(query, (f"{partial}%", f"%{partial}%", limit)).fetchall()

                suggestions = []
                for ticker, name, sector in results:
                    suggestions.append({
                        "ticker": ticker,
                        "name": name or "Unknown",
                        "sector": sector or "Unknown",
                        "display": f"{ticker} - {name or 'Unknown'}"
                    })
                return suggestions
            else:
                query = """
                    SELECT DISTINCT ticker
                    FROM company.metadata
                    WHERE ticker LIKE ?
                    ORDER BY ticker
                    LIMIT ?
                """
                results = self.conn.execute(query, (f"{partial}%", limit)).fetchall()
                return [{"ticker": row[0], "display": row[0]} for row in results]

        except Exception as e:
            print(f"Error in ticker autocomplete: {e}")
            return []

    def suggest_query_templates(self, partial_query: str) -> List[str]:
        """
        Suggest query templates based on partial input.

        Args:
            partial_query: User's partial query

        Returns:
            List of suggested query templates
        """
        partial_lower = partial_query.lower().strip()

        # Match first word
        first_word = partial_lower.split()[0] if partial_lower else ""

        # Direct match
        if first_word in self.query_templates:
            return self.query_templates[first_word]

        # Fuzzy match
        suggestions = []
        for keyword, templates in self.query_templates.items():
            if keyword.startswith(first_word) or first_word in keyword:
                suggestions.extend(templates)

        # If no match, return popular templates
        if not suggestions:
            suggestions = [
                "Show AAPL revenue for last 5 years",
                "Find stocks with P/E < 15",
                "Compare FAANG companies by market cap",
                "Show top 10 tech stocks by revenue",
                "Find stocks with RSI < 30 (oversold)"
            ]

        return suggestions[:5]  # Limit to 5 suggestions

    def suggest_follow_up_queries(
        self,
        last_query: str,
        last_result_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Suggest follow-up queries based on last query and results.

        Args:
            last_query: User's last query
            last_result_columns: Columns from last result

        Returns:
            List of suggested follow-up queries
        """
        last_query_lower = last_query.lower()
        suggestions = []

        # Check for keywords in last query
        for keyword, templates in self.follow_up_templates.items():
            if keyword in last_query_lower:
                suggestions.extend(templates)

        # Column-based suggestions
        if last_result_columns:
            if "revenue" in str(last_result_columns).lower():
                suggestions.append("Show profit margins")
                suggestions.append("Calculate revenue growth rate")

            if "price" in str(last_result_columns).lower() or "close" in str(last_result_columns).lower():
                suggestions.append("Add moving averages (SMA)")
                suggestions.append("Show RSI indicator")

            if "ticker" in str(last_result_columns).lower():
                suggestions.append("Add company names")
                suggestions.append("Add sector information")

        # Generic suggestions
        if not suggestions:
            suggestions = [
                "Plot results as chart",
                "Export to CSV",
                "Show more details",
                "Compare to sector average",
                "Filter by date range"
            ]

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return unique_suggestions[:5]  # Limit to 5

    def suggest_peer_groups(self, partial: str = "") -> List[Dict[str, any]]:
        """
        Suggest peer groups for comparison queries.

        Args:
            partial: Partial peer group name

        Returns:
            List of peer group suggestions
        """
        peer_groups = {
            "FAANG": {
                "name": "FAANG",
                "description": "Facebook, Apple, Amazon, Netflix, Google",
                "example": "Compare FAANG companies by revenue"
            },
            "Magnificent Seven": {
                "name": "Magnificent Seven",
                "description": "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
                "example": "Show Magnificent Seven P/E ratios"
            },
            "Semiconductors": {
                "name": "Semiconductors",
                "description": "Major chip manufacturers",
                "example": "Rank Semiconductors by profit margin"
            },
            "Cloud Computing": {
                "name": "Cloud Computing",
                "description": "Cloud infrastructure providers",
                "example": "Compare Cloud Computing revenue growth"
            },
            "Banks": {
                "name": "Banks",
                "description": "Major banking institutions",
                "example": "Show Banks ROE and debt ratios"
            }
        }

        if not partial:
            return list(peer_groups.values())

        partial_lower = partial.lower()
        matches = []
        for key, group in peer_groups.items():
            if partial_lower in key.lower() or partial_lower in group["description"].lower():
                matches.append(group)

        return matches if matches else list(peer_groups.values())[:5]

    def suggest_columns_for_table(self, table_name: str) -> List[Dict[str, str]]:
        """
        Suggest common columns for a specific table.

        Args:
            table_name: Table name

        Returns:
            List of column suggestions with descriptions
        """
        if not self.conn:
            return []

        try:
            # Get all columns
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = []

            for col_name, col_type in result:
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "display": f"{col_name} ({col_type})"
                })

            return columns

        except Exception as e:
            print(f"Error getting columns for {table_name}: {e}")
            return []

    def suggest_filters(self, table_name: str, column_name: str) -> List[str]:
        """
        Suggest common filter values for a column.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            List of suggested filter expressions
        """
        # Common filter patterns
        numeric_filters = [
            f"{column_name} > 0",
            f"{column_name} < 100",
            f"{column_name} BETWEEN 10 AND 100",
            f"{column_name} IS NOT NULL"
        ]

        text_filters = [
            f"{column_name} LIKE '%search%'",
            f"{column_name} IN ('value1', 'value2')",
            f"{column_name} IS NOT NULL"
        ]

        date_filters = [
            f"{column_name} > '2023-01-01'",
            f"{column_name} >= CURRENT_DATE - INTERVAL 1 YEAR",
            f"{column_name} BETWEEN '2023-01-01' AND '2023-12-31'"
        ]

        # Try to determine column type
        if any(kw in column_name.lower() for kw in ["date", "time", "timestamp"]):
            return date_filters
        elif any(kw in column_name.lower() for kw in ["name", "ticker", "sector", "industry"]):
            return text_filters
        else:
            return numeric_filters


def format_ticker_suggestions(suggestions: List[Dict[str, str]]) -> str:
    """
    Format ticker suggestions for display.

    Args:
        suggestions: List of ticker suggestion dicts

    Returns:
        Formatted string
    """
    if not suggestions:
        return "No matching tickers found."

    lines = []
    lines.append("\n💡 Ticker Suggestions:")
    for s in suggestions:
        ticker = s.get("ticker", "")
        name = s.get("name", "")
        sector = s.get("sector", "")
        lines.append(f"  • {ticker:6s} - {name[:40]:40s} ({sector})")

    return "\n".join(lines)


def format_query_suggestions(suggestions: List[str], title: str = "Query Suggestions") -> str:
    """
    Format query suggestions for display.

    Args:
        suggestions: List of query suggestions
        title: Title for suggestions section

    Returns:
        Formatted string
    """
    if not suggestions:
        return "No suggestions available."

    lines = []
    lines.append(f"\n💡 {title}:")
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)


def extract_ticker_from_query(query: str) -> Optional[str]:
    """
    Extract ticker symbol from query text.

    Args:
        query: User query

    Returns:
        Ticker symbol if found, else None
    """
    # Look for 2-5 uppercase letters (ticker pattern)
    match = re.search(r'\b([A-Z]{2,5})\b', query)
    if match:
        return match.group(1)
    return None


def suggest_date_range(query: str) -> Optional[str]:
    """
    Suggest date range based on query context.

    Args:
        query: User query

    Returns:
        Suggested date range string
    """
    query_lower = query.lower()

    if "ytd" in query_lower or "year to date" in query_lower:
        return "CURRENT_DATE >= DATE_TRUNC('year', CURRENT_DATE)"

    if "last year" in query_lower:
        return "CURRENT_DATE >= CURRENT_DATE - INTERVAL 1 YEAR"

    if "last quarter" in query_lower:
        return "CURRENT_DATE >= CURRENT_DATE - INTERVAL 3 MONTHS"

    if "last month" in query_lower:
        return "CURRENT_DATE >= CURRENT_DATE - INTERVAL 1 MONTH"

    # Extract number of years/months
    match = re.search(r'last (\d+) (year|month)s?', query_lower)
    if match:
        num = match.group(1)
        unit = match.group(2)
        return f"CURRENT_DATE >= CURRENT_DATE - INTERVAL {num} {unit.upper()}S"

    return None
