"""Logging configuration for the tiger package."""

import logging
import os
import sys


def get_logger(name: str = "tiger") -> logging.Logger:
    """Return a logger for the given name.

    Uses the package hierarchy so child loggers inherit the root tiger config.

    Args:
        name: Logger name, typically ``__name__`` from the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def setup_logging(level: int | str | None = None) -> None:
    """Configure the tiger package logger.

    Call once at the entry point (notebook cell, training script, etc.).
    Safe to call multiple times — duplicate handlers are not added.

    The log level is resolved in this order:
    1. ``level`` argument (if provided)
    2. ``LOG_LEVEL`` environment variable (e.g. ``LOG_LEVEL=DEBUG``)
    3. ``logging.INFO`` as the default

    Args:
        level: Logging level, e.g. ``logging.DEBUG`` or ``"DEBUG"``.
    """
    logger = logging.getLogger("tiger")
    if logger.handlers:
        return

    resolved = level or os.environ.get("LOG_LEVEL", logging.INFO)
    logger.setLevel(resolved)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolved)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


# Module-level logger for direct use: `from tiger.logger import logger`
logger = get_logger("tiger")
