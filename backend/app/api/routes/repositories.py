"""Repository resource routes."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.core.exceptions import BadGatewayError, ForbiddenError, NotFoundError
from app.db.database import get_db
from app.db.models import User
from app.integrations.github.contents import (
    GitHubAuthError,
    GitHubContentClient,
    GitHubContentError,
    GitHubNotFoundError,
)
from app.integrations.github.tokens import decrypt_access_token
from app.scanner.service import get_owned_repository, scan_repository
from app.schemas.availability import ResourceAvailabilityResponse
from app.schemas.repositories import RepositoryScanResponse

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.get("", response_model=ResourceAvailabilityResponse)
def list_repositories() -> ResourceAvailabilityResponse:
    """Confirm that the versioned repositories resource is registered."""

    return ResourceAvailabilityResponse(resource="repositories")


@router.post("/{repository_id}/scan", response_model=RepositoryScanResponse)
def scan_repository_files(
    repository_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> RepositoryScanResponse:
    """Index the source files and code chunks of a repository owned by the caller."""

    repository = get_owned_repository(session, repository_id, current_user)
    try:
        token = decrypt_access_token(repository.github_account.access_token_encrypted)
        with GitHubContentClient(token) as client:
            summary = scan_repository(session, repository, client)
    except GitHubAuthError:
        raise ForbiddenError(
            "GitHub access could not be verified. Reconnect GitHub and try again."
        ) from None
    except GitHubNotFoundError:
        raise NotFoundError("Repository was not found on GitHub or is not accessible") from None
    except GitHubContentError:
        raise BadGatewayError("GitHub request failed. Try again later.") from None
    return RepositoryScanResponse(**vars(summary))
