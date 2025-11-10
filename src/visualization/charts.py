#!/usr/bin/env python3
"""Visualization module for FinanGPT - Chart generation and data formatting."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

# Use non-interactive backend for server environments
matplotlib.use('Agg')

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Chart type keywords for intent detection
CHART_KEYWORDS = {
    'line': ['plot', 'trend', 'over time', 'timeline', 'time series'],
    'bar': ['compare', 'comparison', 'versus', 'vs', 'bar chart', 'column'],
    'scatter': ['correlation', 'scatter', 'relationship', 'vs'],
    'candlestick': ['candlestick', 'ohlc', 'candle', 'stock chart'],
}

# Financial column patterns for formatting
REVENUE_PATTERNS = ['revenue', 'income', 'cash', 'assets', 'liabilities', 'equity', 'flow', 'profit', 'ebitda', 'marketcap']
RATIO_PATTERNS = ['margin', 'ratio', 'roe', 'roa', 'growth', 'return']
PRICE_PATTERNS = ['price', 'open', 'high', 'low', 'close', 'adj_close']
VOLUME_PATTERNS = ['volume', 'shares', 'count']


def detect_visualization_intent(query: str, df: pd.DataFrame) -> Optional[str]:
    """Determine if query requests a visualization and what type.

    Returns chart type ('line', 'bar', 'scatter', 'candlestick') or None.
    """
    if df.empty:
        return None

    query_lower = query.lower()

    # Check for explicit chart keywords
    for chart_type, keywords in CHART_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            # Special case: candlestick requires OHLC columns
            if chart_type == 'candlestick':
                required_cols = {'open', 'high', 'low', 'close'}
                if required_cols.issubset(set(col.lower() for col in df.columns)):
                    return 'candlestick'
            else:
                return chart_type

    # Auto-detect based on data shape
    return infer_chart_type(df)


def infer_chart_type(df: pd.DataFrame) -> Optional[str]:
    """Infer appropriate chart type based on DataFrame structure."""
    if df.empty:
        return None

    # Check for OHLC candlestick data
    ohlc_cols = {'open', 'high', 'low', 'close'}
    if ohlc_cols.issubset(set(col.lower() for col in df.columns)):
        return 'candlestick'

    # Time-series data with date column → line chart
    if 'date' in df.columns.str.lower():
        if len(df) >= 5:  # Meaningful time series
            return 'line'

    # Multiple tickers with single metric → bar chart
    if 'ticker' in df.columns.str.lower() and len(df) >= 2:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 1:
            return 'bar'

    # Two numeric columns → scatter plot
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2 and len(df) >= 3:
        return 'scatter'

    return None


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    title: str = "Financial Data Visualization",
    query: str = "",
) -> Optional[str]:
    """Create and save a chart based on the data and type.

    Returns the path to the saved chart file, or None if creation failed.
    """
    if df.empty:
        return None

    try:
        if chart_type == 'line':
            return create_line_chart(df, title)
        elif chart_type == 'bar':
            return create_bar_chart(df, title)
        elif chart_type == 'scatter':
            return create_scatter_plot(df, title)
        elif chart_type == 'candlestick':
            return create_candlestick_chart(df, title)
        else:
            return None
    except Exception as e:
        print(f"⚠️  Chart creation failed: {e}")
        return None


def create_line_chart(df: pd.DataFrame, title: str) -> str:
    """Create a line chart for time-series data."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find date column (case-insensitive)
    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if not date_col:
        raise ValueError("No 'date' column found for line chart")

    # Convert dates
    df[date_col] = pd.to_datetime(df[date_col])

    # Find ticker column
    ticker_col = next((col for col in df.columns if col.lower() == 'ticker'), None)

    # Get numeric columns (excluding date and ticker)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric data to plot")

    # Choose which metric to plot (first numeric column)
    metric_col = numeric_cols[0]

    if ticker_col and ticker_col in df.columns:
        # Plot multiple tickers
        for ticker in df[ticker_col].unique():
            subset = df[df[ticker_col] == ticker].sort_values(date_col)
            ax.plot(subset[date_col], subset[metric_col], label=ticker, marker='o', linewidth=2)
        ax.legend(loc='best', frameon=True, shadow=True)
    else:
        # Single series
        df_sorted = df.sort_values(date_col)
        ax.plot(df_sorted[date_col], df_sorted[metric_col], marker='o', linewidth=2, color='#2E86AB')

    # Format axes
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(format_column_name(metric_col), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')

    # Format y-axis with financial units
    if any(pattern in metric_col.lower() for pattern in REVENUE_PATTERNS):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format_large_number(x)))

    plt.tight_layout()

    # Save chart
    filename = sanitize_filename(f"{title}_{datetime.now():%Y%m%d_%H%M%S}.png")
    filepath = CHARTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(filepath)


def create_bar_chart(df: pd.DataFrame, title: str) -> str:
    """Create a bar chart for comparing metrics across tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find ticker column
    ticker_col = next((col for col in df.columns if col.lower() == 'ticker'), None)
    if not ticker_col:
        # Use first column as labels
        ticker_col = df.columns[0]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric data to plot")

    # Choose metric (first numeric column)
    metric_col = numeric_cols[0]

    # Create bar chart
    tickers = df[ticker_col].tolist()
    values = df[metric_col].tolist()

    bars = ax.bar(range(len(tickers)), values, color='#2E86AB', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                format_large_number(value) if abs(value) >= 1e6 else f'{value:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Format axes
    ax.set_xlabel('Company', fontsize=12, fontweight='bold')
    ax.set_ylabel(format_column_name(metric_col), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Format y-axis
    if any(pattern in metric_col.lower() for pattern in REVENUE_PATTERNS):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format_large_number(x)))

    plt.tight_layout()

    # Save chart
    filename = sanitize_filename(f"{title}_{datetime.now():%Y%m%d_%H%M%S}.png")
    filepath = CHARTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(filepath)


def create_scatter_plot(df: pd.DataFrame, title: str) -> str:
    """Create a scatter plot for correlation analysis."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for scatter plot")

    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    # Find ticker column for labels
    ticker_col = next((col for col in df.columns if col.lower() == 'ticker'), None)

    # Create scatter plot
    ax.scatter(df[x_col], df[y_col], s=100, alpha=0.6, color='#2E86AB', edgecolors='black')

    # Add ticker labels if available
    if ticker_col and ticker_col in df.columns:
        for idx, row in df.iterrows():
            ax.annotate(row[ticker_col], (row[x_col], row[y_col]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Format axes
    ax.set_xlabel(format_column_name(x_col), fontsize=12, fontweight='bold')
    ax.set_ylabel(format_column_name(y_col), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save chart
    filename = sanitize_filename(f"{title}_{datetime.now():%Y%m%d_%H%M%S}.png")
    filepath = CHARTS_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(filepath)


def create_candlestick_chart(df: pd.DataFrame, title: str) -> str:
    """Create a candlestick chart for OHLC stock price data."""
    try:
        import mplfinance as mpf
    except ImportError:
        print("⚠️  mplfinance not installed. Install with: pip install mplfinance")
        return create_line_chart(df, title)  # Fallback to line chart

    # Find date column
    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if not date_col:
        raise ValueError("No 'date' column found")

    # Prepare data for mplfinance (requires specific column names)
    df_candle = df.copy()
    df_candle[date_col] = pd.to_datetime(df_candle[date_col])
    df_candle = df_candle.set_index(date_col)

    # Rename columns to mplfinance format (Open, High, Low, Close, Volume)
    column_mapping = {}
    for col in df_candle.columns:
        col_lower = col.lower()
        if col_lower == 'open':
            column_mapping[col] = 'Open'
        elif col_lower == 'high':
            column_mapping[col] = 'High'
        elif col_lower == 'low':
            column_mapping[col] = 'Low'
        elif col_lower == 'close' or col_lower == 'adj_close':
            column_mapping[col] = 'Close'
        elif col_lower == 'volume':
            column_mapping[col] = 'Volume'

    df_candle = df_candle.rename(columns=column_mapping)

    # Sort by date
    df_candle = df_candle.sort_index()

    # Create candlestick chart
    filename = sanitize_filename(f"{title}_{datetime.now():%Y%m%d_%H%M%S}.png")
    filepath = CHARTS_DIR / filename

    # Configure style
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='#E0E0E0')

    # Plot
    mpf.plot(
        df_candle,
        type='candle',
        style=s,
        title=title,
        volume='Volume' in df_candle.columns,
        savefig=filepath,
        figsize=(12, 8),
        ylabel='Price ($)',
        ylabel_lower='Volume',
    )

    return str(filepath)


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B/T suffixes."""
    abs_val = abs(value)

    if abs_val >= 1e12:
        return f"${value/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_financial_value(value: Any, column_name: str) -> str:
    """Format value based on column type (revenue, ratio, price, etc.)."""
    if value is None or pd.isna(value):
        return "N/A"

    if not isinstance(value, (int, float)):
        return str(value)

    col_lower = column_name.lower()

    # Revenue, income, assets, etc. - use K/M/B suffixes
    if any(pattern in col_lower for pattern in REVENUE_PATTERNS):
        return format_large_number(value)

    # Ratios and margins - show as percentages
    if any(pattern in col_lower for pattern in RATIO_PATTERNS):
        return f"{value * 100:.2f}%"

    # Prices - show with $ and 2 decimals
    if any(pattern in col_lower for pattern in PRICE_PATTERNS):
        return f"${value:.2f}"

    # Volume - show with commas
    if any(pattern in col_lower for pattern in VOLUME_PATTERNS):
        return f"{int(value):,}"

    # Default - 2 decimal places
    return f"{value:.2f}"


def format_column_name(column: str) -> str:
    """Format column name for display (human-readable)."""
    # Convert camelCase or snake_case to Title Case
    formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', column)
    formatted = formatted.replace('_', ' ')
    return formatted.title()


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length
    return sanitized[:200]


def pretty_print_formatted(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    use_formatting: bool = True
) -> None:
    """Enhanced pretty print with financial formatting."""
    if not rows:
        print("No rows returned.")
        return

    # Format values
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for idx, value in enumerate(row):
            if use_formatting and idx < len(columns):
                formatted_row.append(format_financial_value(value, columns[idx]))
            else:
                formatted_row.append(str(value))
        formatted_rows.append(formatted_row)

    # Calculate column widths
    widths = [len(str(col)) for col in columns]
    for row in formatted_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    # Print header
    header = " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns))
    divider = "-+-".join("-" * width for width in widths)

    print(header)
    print(divider)

    # Print rows
    for row in formatted_rows:
        line = " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        print(line)


def export_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Export DataFrame to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"📄 Data exported to: {filepath}")


def export_to_json(df: pd.DataFrame, filepath: str) -> None:
    """Export DataFrame to JSON file."""
    df.to_json(filepath, orient='records', indent=2, date_format='iso')
    print(f"📄 Data exported to: {filepath}")


def export_to_excel(df: pd.DataFrame, filepath: str) -> None:
    """Export DataFrame to Excel file."""
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"📄 Data exported to: {filepath}")
    except ImportError:
        print("⚠️  openpyxl not installed. Install with: pip install openpyxl")
