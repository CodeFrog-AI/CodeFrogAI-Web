"""Workspace and Git endpoints for a repository the caller owns.

Every endpoint loads the repository through the existing ownership check first. Git write
operations (branch, commit, push) additionally require an explicit `approved: true` decided
by the server from the authenticated user's request: nothing an agent or model says can
trigger them, and none of them runs as a side effect of code editing. Everything that
changes the checkout holds the repository's workspace lock.
"""

import uuid
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.core.exceptions import ForbiddenError
from app.db.database import get_db
from app.db.models import Repository, User
from app.scanner.service import get_owned_repository
from app.schemas.git import (
    ApprovalRequest,
    BranchRequest,
    BranchResponse,
    CommitRequest,
    CommitResponse,
    GitDiffResponse,
    GitLogEntry,
    GitLogResponse,
    GitStatusResponse,
    PushRequest,
    PushResponse,
    SyncResponse,
    WorkspaceStatusResponse,
)
from app.workspace import exclusive_workspace, get_workspace_root
from app.workspace import service as workspace_service

router = APIRouter(prefix="/repositories", tags=["workspace"])

CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]

APPROVAL_REQUIRED = "This action must be explicitly approved before it runs."


def _require_approval(payload: ApprovalRequest) -> None:
    if payload.approved is not True:
        raise ForbiddenError(APPROVAL_REQUIRED)


def _workspace_response(repository: Repository) -> WorkspaceStatusResponse:
    state = workspace_service.get_state(get_workspace_root(), repository)
    return WorkspaceStatusResponse(repository_id=repository.id, **asdict(state))


@router.get("/{repository_id}/workspace", response_model=WorkspaceStatusResponse)
def workspace_status(repository_id: uuid.UUID, current_user: CurrentUser, session: DbSession) -> WorkspaceStatusResponse:
    """Whether the persistent workspace exists and its branch. Creates nothing."""

    return _workspace_response(get_owned_repository(session, repository_id, current_user))


@router.post("/{repository_id}/workspace", response_model=WorkspaceStatusResponse)
def initialize_workspace(repository_id: uuid.UUID, current_user: CurrentUser, session: DbSession) -> WorkspaceStatusResponse:
    """Clone the connected repository into the workspace if it is not there yet. An existing workspace is left as it is."""

    repository = get_owned_repository(session, repository_id, current_user)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        workspace_service.ensure_workspace(root, repository)
    return _workspace_response(repository)


@router.post("/{repository_id}/workspace/sync", response_model=SyncResponse)
def sync_workspace(repository_id: uuid.UUID, current_user: CurrentUser, session: DbSession) -> SyncResponse:
    """Fetch from GitHub and fast-forward a clean default branch. A workspace with local work is never modified."""

    repository = get_owned_repository(session, repository_id, current_user)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        outcome = workspace_service.sync_workspace(root, repository)
    return SyncResponse(repository_id=repository.id, action=outcome.action, reason=outcome.reason)


@router.get("/{repository_id}/git/status", response_model=GitStatusResponse)
def git_status(repository_id: uuid.UUID, current_user: CurrentUser, session: DbSession) -> GitStatusResponse:
    repository = get_owned_repository(session, repository_id, current_user)
    status = workspace_service.require_git(get_workspace_root(), repository).status()
    return GitStatusResponse(repository_id=repository.id, clean=status.clean, **{key: value for key, value in asdict(status).items()})


@router.get("/{repository_id}/git/diff", response_model=GitDiffResponse)
def git_diff(repository_id: uuid.UUID, current_user: CurrentUser, session: DbSession) -> GitDiffResponse:
    """Uncommitted changes against HEAD, redacted and size-limited."""

    repository = get_owned_repository(session, repository_id, current_user)
    diff = workspace_service.require_git(get_workspace_root(), repository).diff()
    return GitDiffResponse.model_validate({"repository_id": repository.id, **asdict(diff)})


@router.get("/{repository_id}/git/log", response_model=GitLogResponse)
def git_log(
    repository_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> GitLogResponse:
    repository = get_owned_repository(session, repository_id, current_user)
    entries = workspace_service.require_git(get_workspace_root(), repository).log(limit)
    return GitLogResponse(repository_id=repository.id, commits=[GitLogEntry(**asdict(entry)) for entry in entries])


@router.post("/{repository_id}/agent/branch", response_model=BranchResponse)
def create_branch(repository_id: uuid.UUID, payload: BranchRequest, current_user: CurrentUser, session: DbSession) -> BranchResponse:
    """Create and switch to a new `codefrog/` branch (requires `approved: true`)."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        branch = workspace_service.require_git(root, repository, current_user).create_branch(payload.name)
    return BranchResponse(repository_id=repository.id, branch=branch)


@router.post("/{repository_id}/agent/commit", response_model=CommitResponse)
def commit_changes(repository_id: uuid.UUID, payload: CommitRequest, current_user: CurrentUser, session: DbSession) -> CommitResponse:
    """Commit the workspace's changes on the current `codefrog/` branch (requires `approved: true`)."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        result = workspace_service.require_git(root, repository, current_user).commit(payload.message)
    return CommitResponse(repository_id=repository.id, **asdict(result))


@router.post("/{repository_id}/agent/push", response_model=PushResponse)
def push_branch(repository_id: uuid.UUID, payload: PushRequest, current_user: CurrentUser, session: DbSession) -> PushResponse:
    """Push the current `codefrog/` branch to GitHub, never forced (requires `approved: true`)."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    with exclusive_workspace(root, repository.id):
        git = workspace_service.require_git(root, repository, current_user)
        remote = workspace_service.resolve_remote(repository)
        result = git.push(remote.token)
    return PushResponse(repository_id=repository.id, **asdict(result))
