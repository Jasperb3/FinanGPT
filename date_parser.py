"""
Enhanced Date Parser Module

Parses complex and natural language date expressions into SQL-compatible date ranges.

Author: FinanGPT Team
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from dateutil import relativedelta


class EnhancedDateParser:
    """
    Parse natural language date expressions into SQL date constraints.

    Supported patterns:
    - Relative: "last year", "past 6 months", "YTD"
    - Quarters: "Q4 2024", "last quarter"
    - Specific: "2023-01-01", "January 2023"
    - Event-based: "since IPO", "after stock split"
    - Financial: "fiscal year 2023", "last earnings season"
    """

    def __init__(self):
        """Initialize EnhancedDateParser."""
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for date expressions."""
        self.patterns = {
            # Relative dates
            "ytd": r'\b(ytd|year to date)\b',
            "last_n_years": r'\blast\s+(\d+)\s+years?\b',
            "last_n_months": r'\blast\s+(\d+)\s+months?\b',
            "last_n_days": r'\blast\s+(\d+)\s+days?\b',
            "last_year": r'\blast\s+year\b',
            "last_month": r'\blast\s+month\b',
            "last_week": r'\blast\s+week\b',
            "this_year": r'\bthis\s+year\b',
            "this_quarter": r'\bthis\s+quarter\b',

            # Quarters
            "quarter": r'\bQ([1-4])\s+(\d{4})\b',
            "last_quarter": r'\blast\s+quarter\b',
            "next_quarter": r'\bnext\s+quarter\b',

            # Fiscal years
            "fiscal_year": r'\bfiscal\s+year\s+(\d{4})\b',

            # Specific dates
            "iso_date": r'\b(\d{4})-(\d{2})-(\d{2})\b',
            "year_only": r'\b(\d{4})\b',

            # Event-based
            "since_ipo": r'\bsince\s+ipo\b',
            "after_split": r'\bafter\s+(the\s+)?(stock\s+)?split\b',
            "before_covid": r'\bbefore\s+covid\b',
            "past_n_years": r'\bpast\s+(\d+)\s+years?\b',
            "last_earnings": r'\blast\s+earnings(\s+season)?\b'
        }

    def parse(self, query: str, reference_date: Optional[datetime] = None) -> Optional[Tuple[str, str]]:
        """
        Parse date expression from query.

        Args:
            query: User query containing date expression
            reference_date: Reference date (defaults to today)

        Returns:
            Tuple of (start_date, end_date) as SQL-compatible strings, or None
        """
        if reference_date is None:
            reference_date = datetime.now()

        query_lower = query.lower()

        # Try each pattern
        for pattern_name, pattern in self.patterns.items():
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._parse_pattern(pattern_name, match, reference_date)

        return None

    def _parse_pattern(
        self,
        pattern_name: str,
        match: re.Match,
        reference_date: datetime
    ) -> Optional[Tuple[str, str]]:
        """Parse specific pattern match into date range."""

        # YTD
        if pattern_name == "ytd":
            start = datetime(reference_date.year, 1, 1)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last N years
        if pattern_name == "last_n_years" or pattern_name == "past_n_years":
            n = int(match.group(1))
            start = reference_date - relativedelta.relativedelta(years=n)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last N months
        if pattern_name == "last_n_months":
            n = int(match.group(1))
            start = reference_date - relativedelta.relativedelta(months=n)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last N days
        if pattern_name == "last_n_days":
            n = int(match.group(1))
            start = reference_date - timedelta(days=n)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last year
        if pattern_name == "last_year":
            start = reference_date - relativedelta.relativedelta(years=1)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last month
        if pattern_name == "last_month":
            start = reference_date - relativedelta.relativedelta(months=1)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # Last week
        if pattern_name == "last_week":
            start = reference_date - timedelta(days=7)
            return (start.strftime("%Y-%m-%d"), reference_date.strftime("%Y-%m-%d"))

        # This year
        if pattern_name == "this_year":
            start = datetime(reference_date.year, 1, 1)
            end = datetime(reference_date.year, 12, 31)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # This quarter
        if pattern_name == "this_quarter":
            quarter = (reference_date.month - 1) // 3 + 1
            start_month = (quarter - 1) * 3 + 1
            start = datetime(reference_date.year, start_month, 1)
            end_month = start_month + 2
            if end_month > 12:
                end = datetime(reference_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(reference_date.year, end_month + 1, 1) - timedelta(days=1)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Specific quarter (Q1 2024)
        if pattern_name == "quarter":
            quarter = int(match.group(1))
            year = int(match.group(2))
            start_month = (quarter - 1) * 3 + 1
            start = datetime(year, start_month, 1)

            # End of quarter
            end_month = start_month + 2
            if end_month == 12:
                end = datetime(year, 12, 31)
            else:
                end = datetime(year, end_month + 1, 1) - timedelta(days=1)

            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Last quarter
        if pattern_name == "last_quarter":
            current_quarter = (reference_date.month - 1) // 3 + 1
            if current_quarter == 1:
                year = reference_date.year - 1
                quarter = 4
            else:
                year = reference_date.year
                quarter = current_quarter - 1

            start_month = (quarter - 1) * 3 + 1
            start = datetime(year, start_month, 1)
            end_month = start_month + 2
            if end_month == 12:
                end = datetime(year, 12, 31)
            else:
                end = datetime(year, end_month + 1, 1) - timedelta(days=1)

            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Fiscal year
        if pattern_name == "fiscal_year":
            year = int(match.group(1))
            # Assume fiscal year = calendar year for simplicity
            # In production, this would look up company-specific fiscal year
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # ISO date
        if pattern_name == "iso_date":
            date_str = match.group(0)
            return (date_str, date_str)

        # Year only
        if pattern_name == "year_only":
            year = int(match.group(1))
            # Only treat as date if it's a reasonable year (1900-2100)
            if 1900 <= year <= 2100:
                start = datetime(year, 1, 1)
                end = datetime(year, 12, 31)
                return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        # Before COVID
        if pattern_name == "before_covid":
            return ("1900-01-01", "2020-03-01")

        # Event-based dates (return None, needs context)
        if pattern_name in ["since_ipo", "after_split", "last_earnings"]:
            # These need company-specific data
            # Return None to signal special handling needed
            return None

        return None

    def generate_sql_date_filter(
        self,
        query: str,
        date_column: str = "date",
        reference_date: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Generate SQL date filter from natural language query.

        Args:
            query: User query
            date_column: Name of date column in SQL
            reference_date: Reference date (defaults to today)

        Returns:
            SQL WHERE clause for date filtering, or None
        """
        date_range = self.parse(query, reference_date)

        if not date_range:
            return None

        start_date, end_date = date_range

        if start_date == end_date:
            return f"{date_column} = '{start_date}'"
        else:
            return f"{date_column} BETWEEN '{start_date}' AND '{end_date}'"


def parse_date_expression(query: str) -> Optional[Dict[str, str]]:
    """
    Parse date expression from query (convenience function).

    Args:
        query: User query

    Returns:
        Dictionary with start_date, end_date, and sql_filter
    """
    parser = EnhancedDateParser()
    date_range = parser.parse(query)

    if date_range:
        start_date, end_date = date_range
        sql_filter = parser.generate_sql_date_filter(query)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "sql_filter": sql_filter
        }

    return None


def detect_date_references(query: str) -> List[str]:
    """
    Detect all date references in query.

    Args:
        query: User query

    Returns:
        List of detected date expressions
    """
    patterns = [
        r'\bytd\b|\byear to date\b',
        r'\blast\s+\d+\s+(year|month|day)s?\b',
        r'\bQ[1-4]\s+\d{4}\b',
        r'\bfiscal\s+year\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\bsince\s+ipo\b',
        r'\bbefore\s+covid\b'
    ]

    references = []
    query_lower = query.lower()

    for pattern in patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        references.extend(matches)

    return references
