"""Shared utility helpers for logging and retry logic."""

from __future__ import annotations

import sys
from typing import Sequence


def format_error_message(error: Exception | str, limit: int = 400) -> str:
    """Trim error output to avoid flooding logs with large payloads."""
    message = str(error).strip()
    if len(message) > limit:
        message = message[:limit] + "... (truncated)"
    if not message and isinstance(error, Exception):
        return error.__class__.__name__
    return message


def max_retry_attempts(schedule: Sequence[float]) -> int:
    """Return total attempts including the initial try."""
    return len(schedule) + 1


def color_text(text: str, color: str) -> str:
    """Apply basic ANSI color when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text

    colors = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "bold": "1",
    }
    code = colors.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"
