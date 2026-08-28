"""Schemas for API health responses."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """The public database health response."""

    status: Literal["healthy"] = "healthy"
    database: Literal["healthy"] = "healthy"
