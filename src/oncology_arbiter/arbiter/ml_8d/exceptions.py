"""Bilinear decoder input validation errors."""
from __future__ import annotations

class BilinearDecoderInputError(Exception):
    """Unsafe bilinear request. Callers must not relabel this as PATH A."""

    def __init__(self, message: str, *, code: str, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.missing = list(missing or [])
