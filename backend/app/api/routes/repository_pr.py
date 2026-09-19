"""Pull request endpoints for a repository the caller owns: create, read, diff, and AI review.

Every endpoint loads the repository through the existing ownership check first. Creating a
pull request and starting a review each need an explicit `approved: true`, decided by the
server from the authenticated user's request; nothing a model says can trigger them. The
review is read-only: it never changes files, commits, pushes, or opens anything.
"""

import uuid
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from app.agent.llm import get_llm_provider
from app.agent.reviewer import review_pull_request
from app.api.routes.repositories import _agent_errors
from app.auth.dependencies import get_current_user
from app.context.redaction import redact_secrets
from app.core.config import get_settings
from app.core.exceptions import ForbiddenError
from app.db.database import get_db
from app.db.models import User
from app.embeddings.provider import get_embedding_provider
from app.integrations.github.pr_diff import sanitize_diff
from app.pullrequests import service as pr_service
from app.pullrequests.service import pull_request_client, pull_request_errors
from app.scanner.service import get_owned_repository
from app.schemas.git import ApprovalRequest
from app.schemas.pull_request import (
    CreatePullRequestRequest,
    PullRequestDiffResponse,
    PullRequestFileResponse,
    PullRequestResponse,
    PullRequestSummaryResponse,
    ReviewMetadata,
    ReviewRequest,
    ReviewResponse,
)
from app.workspace import exclusive_workspace, get_workspace_root
from app.workspace import service as workspace_service

router = APIRouter(prefix="/repositories", tags=["pull-requests"])

CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]
PullNumber = Annotated[int, Path(ge=1, le=10_000_000)]

APPROVAL_REQUIRED = "This action must be explicitly approved before it runs."
MAX_BODY_CHARS_RETURNED = 10_000


def _require_approval(payload: ApprovalRequest) -> None:
    if payload.approved is not True:
        raise ForbiddenError(APPROVAL_REQUIRED)


@router.post("/{repository_id}/agent/pr", response_model=PullRequestSummaryResponse)
def create_pull_request(
    repository_id: uuid.UUID, payload: CreatePullRequestRequest, current_user: CurrentUser, session: DbSession
) -> PullRequestSummaryResponse:
    """Open a pull request from the workspace's pushed `codefrog/` branch to the default branch (requires `approved: true`).

    If an open pull request for that branch already exists it is returned instead (`created: false`).
    """

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        git = workspace_service.require_git(root, repository, current_user)
        with pull_request_errors("The repository or branch was not found on GitHub."), pull_request_client(repository) as client:
            result = pr_service.create_pull_request(client, repository, git, title=payload.title, body=payload.body)
    pull_request = result.pull_request
    return PullRequestSummaryResponse(
        repository_id=repository.id,
        number=pull_request.number,
        url=pull_request.url,
        title=redact_secrets(pull_request.title)[0],
        head_branch=pull_request.head_branch,
        base_branch=pull_request.base_branch,
        state=pull_request.state,
        created=result.created,
        redactions=result.redactions,
    )


@router.get("/{repository_id}/agent/pr/{number}", response_model=PullRequestResponse)
def get_pull_request(repository_id: uuid.UUID, number: PullNumber, current_user: CurrentUser, session: DbSession) -> PullRequestResponse:
    repository = get_owned_repository(session, repository_id, current_user)
    with pull_request_errors(), pull_request_client(repository) as client:
        pull_request = client.get_pull_request(repository.owner, repository.name, number)
    body, redactions = redact_secrets(pull_request.body[:MAX_BODY_CHARS_RETURNED])
    return PullRequestResponse(
        repository_id=repository.id,
        number=pull_request.number,
        url=pull_request.url,
        title=redact_secrets(pull_request.title)[0],
        body=body,
        body_redactions=redactions,
        state=pull_request.state,
        draft=pull_request.draft,
        head_branch=pull_request.head_branch,
        base_branch=pull_request.base_branch,
        created_at=pull_request.created_at,
        updated_at=pull_request.updated_at,
    )


@router.get("/{repository_id}/agent/pr/{number}/diff", response_model=PullRequestDiffResponse)
def get_pull_request_diff(
    repository_id: uuid.UUID, number: PullNumber, current_user: CurrentUser, session: DbSession
) -> PullRequestDiffResponse:
    """The pull request's changed files and patches: bounded, secrets redacted, protected files left out."""

    repository = get_owned_repository(session, repository_id, current_user)
    with pull_request_errors(), pull_request_client(repository) as client:
        diff = sanitize_diff(client.get_pull_request_diff(repository.owner, repository.name, number))
    return PullRequestDiffResponse(
        repository_id=repository.id,
        number=number,
        files=[PullRequestFileResponse(**asdict(entry)) for entry in diff.files],
        total_files=diff.total_files,
        withheld=diff.withheld,
        truncated=diff.truncated,
    )


@router.post("/{repository_id}/agent/pr/{number}/review", response_model=ReviewResponse)
def review_pull_request_endpoint(
    repository_id: uuid.UUID, number: PullNumber, payload: ReviewRequest, current_user: CurrentUser, session: DbSession
) -> ReviewResponse:
    """Have the AI review a pull request (requires `approved: true`). Read-only: nothing is changed."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    with _agent_errors():
        provider = get_llm_provider()
        with pull_request_errors(), pull_request_client(repository) as client:
            pull_request = client.get_pull_request(repository.owner, repository.name, number)
            diff = sanitize_diff(client.get_pull_request_diff(repository.owner, repository.name, number))
        result = review_pull_request(
            session,
            current_user,
            repository,
            pull_request,
            diff,
            provider,
            get_embedding_provider,
            max_iterations=get_settings().agent_max_iterations,
        )
    return ReviewResponse(
        repository_id=repository.id,
        pull_request_number=number,
        review=result.review,
        warnings=result.warnings,
        metadata=ReviewMetadata(
            model=result.model,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            stop_reason=result.stop_reason,
            files_reviewed=result.files_reviewed,
            duration_ms=result.duration_ms,
        ),
    )
