"""Request-id middleware.

Reads an incoming X-Request-Id (accepts client-supplied ids for distributed
tracing continuity), or mints a new uuid4 hex. Attaches it to
`request.state.request_id` for downstream handlers and echoes it as a header
on the response so the client can quote it in a support ticket.
"""
from __future__ import annotations

import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_ID_HEADER = "X-Request-Id"
# Accept hex-uuid, dashed uuid, or short reasonable ids from callers
_ALLOWED = re.compile(r"^[A-Za-z0-9._\-]{4,64}$")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach `request.state.request_id` + echo `X-Request-Id` on response.

    If the caller sends a valid X-Request-Id header, we reuse it — this is
    how tracing IDs propagate across services. If missing or malformed, we
    mint a fresh uuid4 hex.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        if incoming and _ALLOWED.match(incoming):
            rid = incoming
        else:
            rid = uuid.uuid4().hex
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response
