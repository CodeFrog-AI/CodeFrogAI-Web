"""Centralized logging configuration and request correlation middleware."""

import logging
import time
import uuid
from contextvars import ContextVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")

REDACTED_VALUE = "REDACTED"
SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "code",
        "state",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "client_id",
        "secret",
        "password",
        "api_key",
        "apikey",
        "key",
        "authorization",
    }
)

# Loggers owned by third-party libraries that record full request URLs (including
# query strings) rather than the sanitized, path-only messages this application's
# own RequestLoggingMiddleware emits. OAuth codes, state, and secrets travel as
# query parameters, so these loggers need the same redaction applied.
SENSITIVE_URL_LOGGER_NAMES = ("httpx", "uvicorn.access")


def _redact_url_like(value: str) -> str:
    """Redact sensitive query parameter values within a URL or path string."""

    if "?" not in value:
        return value
    split = urlsplit(value)
    if not split.query:
        return value
    redacted_pairs = [
        (key, REDACTED_VALUE if key.lower() in SENSITIVE_QUERY_PARAMS else val)
        for key, val in parse_qsl(split.query, keep_blank_values=True)
    ]
    redacted_query = urlencode(redacted_pairs)
    return urlunsplit((split.scheme, split.netloc, split.path, redacted_query, split.fragment))


def _redact_arg(arg: object) -> object:
    """Redact an individual log-record argument if it looks like a URL or path."""

    if isinstance(arg, (str, bytes)):
        text = arg.decode() if isinstance(arg, bytes) else arg
    elif hasattr(arg, "query"):
        # httpx.URL and similar URL objects: stringify to redact, then return as str.
        text = str(arg)
    else:
        return arg
    if not (text.startswith("/") or "://" in text):
        return arg
    return _redact_url_like(text)


class SensitiveQueryStringFilter(logging.Filter):
    """Strip OAuth codes, tokens, and secrets from third-party URL/access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_arg(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: _redact_arg(value) for key, value in record.args.items()}
        elif not record.args and isinstance(record.msg, str):
            record.msg = _redact_url_like(record.msg)
        return True


class RequestIdFilter(logging.Filter):
    """Add the current request ID to every formatted application log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


def _install_sensitive_url_filters() -> None:
    """Attach URL redaction to loggers that print raw request URLs or paths.

    Attached to the logger (not a handler) so redaction runs during
    ``Logger.handle`` before any handler — including test-only ones such as
    pytest's ``caplog`` — ever sees the record.
    """

    for logger_name in SENSITIVE_URL_LOGGER_NAMES:
        target_logger = logging.getLogger(logger_name)
        if not any(isinstance(f, SensitiveQueryStringFilter) for f in target_logger.filters):
            target_logger.addFilter(SensitiveQueryStringFilter())


def configure_logging(log_level: str) -> None:
    """Configure a readable standard-library logger exactly once."""

    _install_sensitive_url_filters()

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
