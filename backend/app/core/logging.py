"""Centralized logging configuration and request correlation middleware."""

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Add the current request ID to every formatted application log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


def configure_logging(log_level: str) -> None:
    """Configure a readable standard-library logger exactly once."""

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"
        )
    )
    root_logger.addHandler(handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log safe request metadata only, while assigning a correlation ID."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.logger = logging.getLogger("codefrog.api")

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_context.set(request_id)
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            self.logger.error(
                "request failed method=%s path=%s status_code=500",
                request.method,
                request.url.path,
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            if response is not None:
                response.headers[REQUEST_ID_HEADER] = request_id
                self._log_response(
                    response.status_code,
                    request.method,
                    request.url.path,
                    duration_ms,
                )
            request_id_context.reset(token)

    def _log_response(self, status_code: int, method: str, path: str, duration_ms: int) -> None:
        log_method = self.logger.info
        if status_code >= 500:
            log_method = self.logger.error
        elif status_code >= 400:
            log_method = self.logger.warning
        log_method(
            "request completed method=%s path=%s status_code=%s duration_ms=%s",
            method,
            path,
            status_code,
            duration_ms,
        )
