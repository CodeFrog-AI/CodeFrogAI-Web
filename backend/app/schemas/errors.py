"""Pydantic schemas for the stable API error envelope."""

from typing import Any

from pydantic import BaseModel


class ValidationErrorDetail(BaseModel):
    """A safe subset of a request-validation failure."""

    location: list[str]
    message: str
    type: str


class ErrorBody(BaseModel):
    """The inner, predictable API error payload."""

    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """The public error envelope returned by every global handler."""

    error: ErrorBody
