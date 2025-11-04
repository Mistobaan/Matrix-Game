"""Middleware and helpers for structured logging."""

from __future__ import annotations

import logging
import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs HTTP requests without capturing payload contents."""

    def __init__(self, app, logger: Optional[logging.Logger] = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("matrix_game.web")

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        body = await request.body()
        request._body = body  # allow downstream handlers to re-read  # pylint: disable=protected-access
        request_size = len(body)

        response = await call_next(request)

        process_ms = (time.perf_counter() - start_time) * 1000
        response_size = _response_size(response)
        self.logger.info(
            "HTTP %s %s -> %s %.2fms req=%sB resp=%sB",
            request.method,
            request.url.path,
            response.status_code,
            process_ms,
            request_size,
            response_size,
        )
        return response


def _response_size(response: Response) -> int:
    header_size = response.headers.get("content-length")
    if header_size is not None:
        try:
            return int(header_size)
        except ValueError:
            pass

    body = getattr(response, "body", None)
    if body is None:
        return 0
    if isinstance(body, bytes):
        return len(body)
    if isinstance(body, str):
        return len(body.encode("utf-8"))
    return 0
