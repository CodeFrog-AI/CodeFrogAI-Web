"""Centralized, safe exception-to-response conversion."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import ApplicationError
from app.schemas.errors import ErrorResponse, ValidationErrorDetail

logger = logging.getLogger(__name__)

HTTP_ERROR_CODES = {
    400: ("BAD_REQUEST", "The request could not be processed"),
    401: ("UNAUTHORIZED", "Authentication is required"),
    403: ("FORBIDDEN", "You do not have permission to perform this action"),
    404: ("RESOURCE_NOT_FOUND", "The requested resource was not found"),
    409: ("CONFLICT", "The request conflicts with the current resource state"),
    503: ("SERVICE_UNAVAILABLE", "Service unavailable"),
}


def error_response(
    status_code: int, code: str, message: str, details: Any | None = None
) -> JSONResponse:
    """Build the common public error envelope."""

    payload = ErrorResponse(error={"code": code, "message": message, "details": details})
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def safe_validation_details(error: RequestValidationError) -> list[ValidationErrorDetail]:
    """Expose only validation fields that cannot echo request secrets."""

    return [
        ValidationErrorDetail(
            location=[str(part) for part in item["loc"]],
            message=item["msg"],
            type=item["type"],
        )
        for item in error.errors()
    ]


async def application_error_handler(
    _request: Request, error: ApplicationError
) -> JSONResponse:
    return error_response(error.status_code, error.code, error.message, error.details)


async def http_exception_handler(
    _request: Request, error: StarletteHTTPException
) -> JSONResponse:
    code, message = HTTP_ERROR_CODES.get(
        error.status_code, ("HTTP_ERROR", "The request could not be processed")
    )
    return error_response(error.status_code, code, message)


async def validation_exception_handler(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    return error_response(
        422,
        "VALIDATION_ERROR",
        "Request validation failed",
        [detail.model_dump() for detail in safe_validation_details(error)],
    )


async def database_exception_handler(_request: Request, error: SQLAlchemyError) -> JSONResponse:
    logger.error("Database operation failed (exception type=%s)", type(error).__name__)
    return error_response(500, "DATABASE_ERROR", "Database operation failed")


async def unexpected_exception_handler(_request: Request, error: Exception) -> JSONResponse:
    logger.error("Unexpected server error (exception type=%s)", type(error).__name__)
    return error_response(500, "INTERNAL_SERVER_ERROR", "An unexpected error occurred")


def register_exception_handlers(app: FastAPI) -> None:
    """Register global handlers once during application construction."""

    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
