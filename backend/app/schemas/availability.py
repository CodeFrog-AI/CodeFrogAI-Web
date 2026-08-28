"""Schemas for placeholder resource availability endpoints."""

from typing import Literal

from pydantic import BaseModel


class ResourceAvailabilityResponse(BaseModel):
    """Describe a registered API resource without exposing business data."""

    status: Literal["available"] = "available"
    resource: Literal["users", "repositories", "tasks"]
