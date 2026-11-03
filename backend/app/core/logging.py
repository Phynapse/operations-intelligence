"""Structured logging configuration.

Logs are emitted as single-line JSON so they can be shipped to a log store
without a parser. Two safety rules are enforced here:

* the LLM API key is registered as a secret and redacted from every record;
* user-supplied free text (questions) is truncated before it reaches a log
  record, so an oversized request body cannot flood the log pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings

#: Keys whose values are replaced with ``***`` in every log record.
SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "llm_api_key",
        "password",
        "secret",
        "token",
    }
)

#: Maximum number of characters of free-text user input kept in a log record.
MAX_LOGGED_TEXT = 300

_RESERVED_LOG_RECORD_KEYS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively replace secret-looking values with a mask."""
    if key is not None and key.lower() in SECRET_KEYS:
        return "***"
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) > MAX_LOGGED_TEXT:
        return value[:MAX_LOGGED_TEXT] + "...[truncated]"
    return value


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_KEYS and not key.startswith("_")
        }
        if context:
            payload["context"] = redact(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(settings: Settings) -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn's own loggers would otherwise emit unstructured access logs.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger that uses the configured root handlers."""
    return logging.getLogger(name)
