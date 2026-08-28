"""Initial repository resource routes."""

from fastapi import APIRouter

from app.schemas.availability import ResourceAvailabilityResponse

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.get("", response_model=ResourceAvailabilityResponse)
def list_repositories() -> ResourceAvailabilityResponse:
    """Confirm that the versioned repositories resource is registered."""

    return ResourceAvailabilityResponse(resource="repositories")
