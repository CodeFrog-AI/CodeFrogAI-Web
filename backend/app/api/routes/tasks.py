"""Initial agent task resource routes."""

from fastapi import APIRouter

from app.schemas.availability import ResourceAvailabilityResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=ResourceAvailabilityResponse)
def list_tasks() -> ResourceAvailabilityResponse:
    """Confirm that the versioned tasks resource is registered."""

    return ResourceAvailabilityResponse(resource="tasks")
