"""Compatibility wrapper for visualization utilities."""

from src.ui import visualize as _visualize_module
from src.ui.visualize import (
    detect_visualization_intent,
    format_financial_value,
    format_large_number,
    format_column_name,
    sanitize_filename,
    infer_chart_type,
    create_chart,
    pretty_print_formatted,
    create_line_chart,
    create_bar_chart,
    create_scatter_plot,
    create_candlestick_chart,
    export_to_csv,
    export_to_json,
    export_to_excel,
)

plt = _visualize_module.plt

__all__ = [
    "detect_visualization_intent",
    "format_financial_value",
    "format_large_number",
    "format_column_name",
    "sanitize_filename",
    "infer_chart_type",
    "create_chart",
    "create_line_chart",
    "create_bar_chart",
    "create_scatter_plot",
    "create_candlestick_chart",
    "export_to_csv",
    "export_to_json",
    "export_to_excel",
    "pretty_print_formatted",
    "plt",
]
