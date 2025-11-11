"""
Centralized logging configuration for FinanGPT.

This module provides consistent logging setup across all modules,
eliminating duplicated logger configuration code.

Features:
- JSON and text format support
- File and console handlers
- Automatic log directory creation
- Configurable log levels

Author: FinanGPT Enhancement Plan 4 Phase 4
Created: 2025-11-10
"""

import json
import logging
import sys
from datetime import datetime, UTC
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.paths import get_logs_dir


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            'timestamp': datetime.now(UTC).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value

        return json.dumps(log_data)


def configure_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = 'INFO',
    log_format: str = 'json',
    console_output: bool = True,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """
    Configure logger with consistent settings.

    Args:
        name: Logger name (usually module name)
        log_dir: Directory for log files (default: 'logs')
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = configure_logger('ingest', level='DEBUG')
        >>> logger.info("Starting ingestion", ticker="AAPL")
    """
    logger = logging.getLogger(name)

    # Return existing logger if already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Create log directory
    log_path = Path(log_dir) if log_dir else get_logs_dir()
    log_path.mkdir(parents=True, exist_ok=True)

    max_bytes_value = max_bytes if max_bytes and max_bytes > 0 else 10 * 1024 * 1024
    backup_count_value = backup_count if backup_count is not None else 5

    # File handler with rotation
    log_file = log_path / f"{name.replace('.', '_')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes_value,
        backupCount=max(backup_count_value, 1),
        encoding='utf-8',
    )

    if log_format == 'json':
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )

    logger.addHandler(file_handler)

    # Console handler (text only for readability)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        logger.addHandler(console_handler)

    return logger


def log_event(
    logger: logging.Logger,
    level: str = 'info',
    **kwargs
) -> None:
    """
    Log an event with structured data.

    Args:
        logger: Logger instance
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        **kwargs: Key-value pairs to log

    Example:
        >>> logger = configure_logger('ingest')
        >>> log_event(logger, phase='ingest.annual', ticker='AAPL', rows=100)
    """
    message = kwargs.pop('message', None) or kwargs.pop('msg', '')

    # Get logging method
    log_method = getattr(logger, level.lower(), logger.info)

    # Log with extra fields
    log_method(message, extra=kwargs)


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Get or create logger with configuration.

    Args:
        name: Logger name
        config: Optional configuration dictionary

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    if config is None:
        config = {}

    log_config = config.get('logging', {})

    max_bytes = None
    if 'max_file_size_mb' in log_config:
        try:
            max_bytes = int(float(log_config['max_file_size_mb']) * 1024 * 1024)
        except (ValueError, TypeError):
            max_bytes = None

    backup_count = log_config.get('backup_count')
    if backup_count is not None:
        try:
            backup_count = int(backup_count)
        except (ValueError, TypeError):
            backup_count = None

    return configure_logger(
        name=name,
        log_dir=log_config.get('directory', 'logs'),
        level=log_config.get('level', 'INFO'),
        log_format=log_config.get('format', 'json'),
        console_output=log_config.get('console_output', True),
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
