"""
Progress indicator utilities using tqdm.

This module provides progress bars and trackers for long-running operations,
significantly improving user experience by showing real-time progress.

Features:
- Simple progress bars for iterables
- Multi-stage progress tracking
- Custom formatting
- Automatic time estimation

Author: FinanGPT Enhancement Plan 3
Created: 2025-11-09
"""

from typing import Iterable, Optional, Any
from tqdm import tqdm


def with_progress(
    iterable: Iterable,
    description: str = "Processing",
    unit: str = "item",
    total: Optional[int] = None,
    disable: bool = False
):
    """
    Wrap iterable with progress bar.

    This provides visual feedback during long operations, preventing users
    from thinking the process has frozen.

    Args:
        iterable: Iterable to wrap (list, generator, etc.)
        description: Description shown before progress bar
        unit: Unit name (e.g., "ticker", "row", "file")
        total: Total count (optional, auto-detected for lists)
        disable: Set True to disable progress bar (for quiet mode)

    Returns:
        tqdm-wrapped iterable

    Example:
        >>> tickers = ["AAPL", "MSFT", "GOOGL"]
        >>> for ticker in with_progress(tickers, "Ingesting", "ticker"):
        ...     ingest_symbol(ticker)
        Ingesting: |████████████████████| 3/3 [00:15<00:00, 0.20 ticker/s]
    """
    return tqdm(
        iterable,
        desc=description,
        unit=unit,
        total=total,
        ncols=80,
        disable=disable,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )


class ProgressTracker:
    """
    Multi-stage progress tracking for complex operations.

    This is useful for operations with multiple distinct stages (e.g.,
    transformation pipeline with 7 tables to process).

    Example:
        >>> tracker = ProgressTracker(total=7, description="Transforming")
        >>> tracker.update("annual financials", rows=1234)
        >>> tracker.update("quarterly financials", rows=5678)
        >>> # ... process remaining stages
        >>> tracker.close()

        Transforming: |████████----------| 2/7 [00:15<00:45, rows=6912]
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        disable: bool = False
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of stages/steps
            description: Description shown before progress bar
            disable: Set True to disable progress bar
        """
        self.pbar = tqdm(
            total=total,
            desc=description,
            unit="step",
            ncols=80,
            disable=disable,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.total = total
        self.current = 0
        self.description = description  # Store original description

    def update(self, step_name: str, increment: int = 1, **metadata):
        """
        Update progress with step name and optional metadata.

        Args:
            step_name: Name of current step (e.g., "annual", "quarterly")
            increment: Number of steps completed (default: 1)
            **metadata: Additional info to display (e.g., rows=1234)

        Example:
            >>> tracker.update("prices", rows=50000, tickers=50)
            Transforming: |████████| 3/7 [00:20<00:15, rows=50000, tickers=50]
        """
        self.current += increment
        self.pbar.update(increment)

        # Format metadata for display
        if metadata:
            postfix = ", ".join(f"{k}={v}" for k, v in metadata.items())
            self.pbar.set_postfix_str(postfix)

        # Update description with current step
        self.pbar.set_description(f"{self.description}: {step_name}")

    def close(self):
        """Close progress bar."""
        self.pbar.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        self.close()


def print_stage(stage_name: str, description: str = ""):
    """
    Print formatted stage header (alternative to progress bar).

    Useful for operations where progress percentage isn't meaningful.

    Args:
        stage_name: Name of current stage
        description: Optional description

    Example:
        >>> print_stage("Ingestion", "Fetching data from yfinance")
        ========================================
        🔄 Ingestion
        Fetching data from yfinance
        ========================================
    """
    print(f"\n{'=' * 60}")
    print(f"🔄 {stage_name}")
    if description:
        print(f"   {description}")
    print(f"{'=' * 60}\n")


def estimate_remaining_time(
    completed: int,
    total: int,
    elapsed_seconds: float
) -> str:
    """
    Estimate remaining time based on progress.

    Args:
        completed: Number of items completed
        total: Total number of items
        elapsed_seconds: Time elapsed so far

    Returns:
        Human-readable time estimate (e.g., "2m 30s")

    Example:
        >>> time_left = estimate_remaining_time(25, 100, 30.0)
        >>> print(f"Estimated time remaining: {time_left}")
        Estimated time remaining: 1m 30s
    """
    if completed == 0:
        return "unknown"

    rate = elapsed_seconds / completed
    remaining_seconds = rate * (total - completed)

    if remaining_seconds < 60:
        return f"{int(remaining_seconds)}s"
    elif remaining_seconds < 3600:
        minutes = int(remaining_seconds / 60)
        seconds = int(remaining_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining_seconds / 3600)
        minutes = int((remaining_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
