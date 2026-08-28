"""Initial user resource routes."""

from fastapi import APIRouter

from app.schemas.availability import ResourceAvailabilityResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=ResourceAvailabilityResponse)
def list_users() -> ResourceAvailabilityResponse:
    """Confirm that the versioned users resource is registered."""

    return ResourceAvailabilityResponse(resource="users")
