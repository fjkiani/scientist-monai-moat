"""Observability: request-id middleware, structured JSON logging, /metrics.

Every request gets an X-Request-Id header (uuid4 hex if the client didn't
provide one). That id flows into the audit ledger AND into a structured
log line so a support engineer can pivot from a user-visible request id
to the exact server-side event.
"""
from .request_id import RequestIdMiddleware
from .logging_config import configure_logging, get_logger

__all__ = ["RequestIdMiddleware", "configure_logging", "get_logger"]
