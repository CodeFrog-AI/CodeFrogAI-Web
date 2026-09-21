"""Small set of safe, application-level API exceptions."""

from typing import Any


class ApplicationError(Exception):
    """Base exception carrying only client-safe error information."""

    status_code = 500
    code = "INTERNAL_SERVER_ERROR"
    message = "An unexpected error occurred"

    def __init__(self, message: str | None = None, details: Any | None = None) -> None:
        self.message = message or self.message
        self.details = details


class BadRequestError(ApplicationError):
    status_code = 400
    code = "BAD_REQUEST"
    message = "The request could not be processed"


class UnauthorizedError(ApplicationError):
    status_code = 401
    code = "UNAUTHORIZED"
    message = "Authentication is required"


class ForbiddenError(ApplicationError):
    status_code = 403
    code = "FORBIDDEN"
    message = "You do not have permission to perform this action"


class NotFoundError(ApplicationError):
    status_code = 404
    code = "RESOURCE_NOT_FOUND"
    message = "The requested resource was not found"


class ConflictError(ApplicationError):
    status_code = 409
    code = "CONFLICT"
    message = "The request conflicts with the current resource state"


class BadGatewayError(ApplicationError):
    status_code = 502
    code = "BAD_GATEWAY"
    message = "An upstream service request failed"


class ServiceUnavailableError(ApplicationError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"
    message = "The service is temporarily unavailable"


class RateLimitedError(ApplicationError):
    status_code = 429
    code = "RATE_LIMITED"
    message = "An upstream rate limit was reached. Try again later."
