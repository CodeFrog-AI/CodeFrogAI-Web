"""Health-check endpoints."""

from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from app.db.database import DatabaseConnectionError, check_database_connection
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


def build_health_response(checker: Callable[[], None]) -> HealthResponse:
    """Return the shared database health response or a safe failure."""

    try:
        checker()
    except DatabaseConnectionError as error:
        raise HTTPException(status_code=503, detail="Database is unavailable") from error

    return HealthResponse()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API and database availability."""

    return build_health_response(check_database_connection)
