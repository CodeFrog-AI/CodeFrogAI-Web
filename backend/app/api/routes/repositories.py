"""Repository resource routes."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.core.exceptions import BadGatewayError, ForbiddenError, NotFoundError
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, User
from app.integrations.github.connection import (
    connect_repository,
    connected_repository_ids,
    get_user_github_account,
)
from app.integrations.github.contents import (
    GitHubAuthError,
    GitHubContentClient,
    GitHubContentError,
    GitHubNotFoundError,
)
from app.integrations.github.tokens import decrypt_access_token
from app.scanner.search import (
    DEFAULT_RESULT_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_RESULT_LIMIT,
    search_repository_code,
)
from app.scanner.service import get_owned_repository, scan_repository
from app.schemas.availability import ResourceAvailabilityResponse
from app.schemas.repositories import (
    CodeSearchResponse,
    CodeSearchResult,
    ConnectedRepositoryResponse,
    ConnectRepositoryRequest,
    GitHubRepositoryListResponse,
    GitHubRepositoryResponse,
    RepositoryScanResponse,
)

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.get("", response_model=ResourceAvailabilityResponse)
def list_repositories() -> ResourceAvailabilityResponse:
    """Confirm that the versioned repositories resource is registered."""

    return ResourceAvailabilityResponse(resource="repositories")


def _github_client_for(account: GitHubAccount) -> GitHubContentClient:
    """Build a GitHub client from the account's stored token, or require a reconnect."""

    token = decrypt_access_token(account.access_token_encrypted)
    if token is None:
        raise GitHubAuthError("No stored GitHub token")
    return GitHubContentClient(token)


@contextmanager
def _github_errors() -> Iterator[None]:
    """Translate GitHub client failures into safe, client-facing API errors."""

    try:
        yield
    except GitHubAuthError:
        raise ForbiddenError(
            "GitHub access could not be verified. Reconnect GitHub and try again."
        ) from None
    except GitHubNotFoundError:
        raise NotFoundError("Repository was not found on GitHub or is not accessible") from None
    except GitHubContentError:
        raise BadGatewayError("GitHub request failed. Try again later.") from None


@router.get("/github", response_model=GitHubRepositoryListResponse)
def list_github_repositories(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> GitHubRepositoryListResponse:
    """List repositories available to the caller's connected GitHub account."""

    account = get_user_github_account(session, current_user)
    with _github_errors(), _github_client_for(account) as client:
        available = client.list_repositories()
    connected = connected_repository_ids(session, account)
    return GitHubRepositoryListResponse(
        repositories=[
            GitHubRepositoryResponse(
                github_repository_id=item.github_repository_id,
                owner=item.owner,
                name=item.name,
                default_branch=item.default_branch,
                private=item.private,
                connected=item.github_repository_id in connected,
            )
            for item in available
        ]
    )


@router.post("/connect", response_model=ConnectedRepositoryResponse)
def connect_github_repository(
    payload: ConnectRepositoryRequest,
    response: Response,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ConnectedRepositoryResponse:
    """Connect a GitHub repository the caller can access; reconnecting reuses the record."""

    account = get_user_github_account(session, current_user)
    with _github_errors(), _github_client_for(account) as client:
        repository, created = connect_repository(
            session, account, client, payload.github_repository_id
        )
    response.status_code = 201 if created else 200
    return _connected_response(repository)


def _connected_response(repository: Repository) -> ConnectedRepositoryResponse:
    return ConnectedRepositoryResponse(
        id=repository.id,
        github_repository_id=repository.github_repository_id,
        owner=repository.owner,
        name=repository.name,
        default_branch=repository.default_branch,
        private=bool((repository.connection_metadata or {}).get("private", False)),
        connection_status=repository.connection_status,
    )


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


@router.get("/{repository_id}/search", response_model=CodeSearchResponse)
def search_repository(
    repository_id: uuid.UUID,
    query: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH, pattern=r"\S")],
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=MAX_RESULT_LIMIT)] = DEFAULT_RESULT_LIMIT,
) -> CodeSearchResponse:
    """Search the indexed code of a repository owned by the caller."""

    repository = get_owned_repository(session, repository_id, current_user)
    search_text = query.strip()
    hits = search_repository_code(session, repository, search_text, limit)
    return CodeSearchResponse(
        repository_id=repository.id,
        query=search_text,
        results=[CodeSearchResult(**vars(hit)) for hit in hits],
    )
