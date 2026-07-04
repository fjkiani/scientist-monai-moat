"""Structured JSON logging.

One log record per line: {ts, level, msg, ...extras}. Extras attached via
`logger.info("...", extra={"request_id": rid, "tenant_id": tid})` flow into
the JSON payload verbatim.

Design invariants:
  * No PHI, no secrets — the caller scrubs before logging.
  * Logs go to stderr as newline-delimited JSON so any log aggregator
    (Render's built-in tail, Datadog, Loki, etc.) can index them.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


LOGGER_NAME = "oncology_arbiter"


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON object.

    We deliberately do NOT use logging's default `%(...)s` templating — that
    hides fields from log aggregators. Extras attached via `extra=` flow
    through `LogRecord.__dict__` untouched.
    """

    RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Attach a single JSON handler to the root logger. Idempotent."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)
    # Uvicorn's default access log is text; silence it so we get pure JSON.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or LOGGER_NAME)
