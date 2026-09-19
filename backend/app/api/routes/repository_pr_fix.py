"""Endpoints for fixing a pull request review finding.

    POST .../agent/pr/{n}/fix-plan   plan a fix for one finding (read-only)
    POST .../agent/pr/{n}/fix        apply the approved plan to the workspace and test it
    GET  .../agent/pr/{n}/fix        where the fix stands
    POST .../agent/pr/{n}/fix/commit commit the tested changes (approved)
    POST .../agent/pr/{n}/fix/push   push them to the pull request's own branch (approved)

Every endpoint loads the repository through the existing ownership check first. Applying a
fix, committing, and pushing each need an explicit `approved: true` decided by the server;
none of them runs as a side effect of another, a review is never re-run automatically, and
no step creates a branch or a pull request.
"""

import uuid
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from app.agent.llm import get_llm_provider
from app.agent.planner import create_plan
from app.api.routes.repositories import _agent_errors
from app.api.routes.repository_git import _require_approval
from app.auth.dependencies import get_current_user
from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import User
from app.embeddings.provider import get_embedding_provider
from app.integrations.github.pr_diff import sanitize_diff
from app.pullrequests.service import pull_request_client, pull_request_errors
from app.prfix import service as fix_service
from app.prfix import state as fix_state
from app.prfix.signing import sign_plan
from app.scanner.service import get_owned_repository
from app.schemas.plan import PlanMetadata
from app.schemas.pr_fix import (
    FixChange,
    FixCommitRequest,
    FixCommitResponse,
    FixMetadata,
    FixPlanRequest,
    FixPlanResponse,
    FixPushRequest,
    FixPushResponse,
    FixRequest,
    FixResponse,
    FixStatusResponse,
    TestResultResponse,
)
from app.workspace import exclusive_workspace, get_workspace_root
from app.workspace import service as workspace_service

router = APIRouter(prefix="/repositories", tags=["pull-request-fixes"])

CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]
PullNumber = Annotated[int, Path(ge=1, le=10_000_000)]


def _fetch(repository, number):
    """The current pull request and its sanitized diff, fetched from GitHub."""

    with pull_request_errors(), pull_request_client(repository) as client:
        pull_request = client.get_pull_request(repository.owner, repository.name, number)
        diff = sanitize_diff(client.get_pull_request_diff(repository.owner, repository.name, number))
        return pull_request, diff


def _fetch_pull_request(repository, number):
    """Just the current pull request (commit and push need its state, not its diff)."""

    with pull_request_errors(), pull_request_client(repository) as client:
        return client.get_pull_request(repository.owner, repository.name, number)


def _tests_response(tests) -> TestResultResponse:
    return TestResultResponse(**asdict(tests))


@router.post("/{repository_id}/agent/pr/{number}/fix-plan", response_model=FixPlanResponse)
def plan_fix(repository_id: uuid.UUID, number: PullNumber, payload: FixPlanRequest, current_user: CurrentUser, session: DbSession) -> FixPlanResponse:
    """Plan a fix for one review finding. Read-only: nothing is changed."""

    repository = get_owned_repository(session, repository_id, current_user)
    root = get_workspace_root()
    with _agent_errors():
        provider = get_llm_provider()
        with pull_request_errors(), pull_request_client(repository) as client:
            pull_request = client.get_pull_request(repository.owner, repository.name, number)
            diff = sanitize_diff(client.get_pull_request_diff(repository.owner, repository.name, number))
            verified = fix_service.verify_selected_finding(repository, number, payload.finding, pull_request=pull_request, diff=diff)
        git = workspace_service.require_git(root, repository, current_user)
        fix_service.verify_workspace_matches(git, pull_request)
        result = create_plan(
            session,
            current_user,
            repository,
            fix_service.build_fix_message(verified.finding, head_branch=pull_request.head_branch),
            provider,
            get_embedding_provider,
            max_iterations=get_settings().agent_max_iterations,
            checkout=workspace_service.open_checkout(root, repository),
            extra_instructions=fix_service.FIX_PLAN_INSTRUCTIONS.replace("{max_files}", str(fix_service.MAX_PLAN_FILES)),
        )
    fix_service.validate_fix_plan(result.plan, verified.finding)
    return FixPlanResponse(
        repository_id=repository.id,
        pull_request_number=number,
        head_sha=pull_request.head_sha,
        finding=verified.finding,
        plan=result.plan,
        plan_signature=sign_plan(repository.id, number, pull_request.head_sha, verified.finding, result.plan),
        warnings=result.warnings,
        metadata=PlanMetadata(
            model=result.model, iterations=result.iterations, tool_calls=result.tool_calls, stop_reason=result.stop_reason, duration_ms=result.duration_ms
        ),
    )


@router.post("/{repository_id}/agent/pr/{number}/fix", response_model=FixResponse)
def apply_fix(repository_id: uuid.UUID, number: PullNumber, payload: FixRequest, current_user: CurrentUser, session: DbSession) -> FixResponse:
    """Apply the approved fix plan to the workspace on the pull request's own branch, then run the tests.

    Requires `approved: true`. Never commits, pushes, creates a branch, or opens a pull request.
    """

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    with _agent_errors():
        provider = get_llm_provider()
        pull_request, diff = _fetch(repository, number)
        verified = fix_service.verify_selected_finding(repository, number, payload.finding, pull_request=pull_request, diff=diff)
        fix_service.verify_plan_signature(repository, number, payload.finding, payload.plan, payload.plan_signature)
        fix_service.validate_fix_plan(payload.plan, verified.finding)
        with exclusive_workspace(root, repository.id):
            outcome = fix_service.run_fix(
                session, current_user, repository, pull_request, verified.finding, payload.plan, provider, get_embedding_provider,
                base_directory=root, max_iterations=get_settings().agent_max_iterations,
            )
    execution = outcome.execution
    return FixResponse(
        repository_id=repository.id,
        pull_request_number=number,
        status=outcome.status,
        branch=outcome.branch,
        head_sha=outcome.head_sha,
        agent_status=execution.status,
        summary=execution.summary,
        changes=[FixChange(**asdict(entry)) for entry in outcome.diff_files],
        withheld=outcome.withheld,
        tests=_tests_response(outcome.tests),
        warnings=outcome.warnings,
        metadata=FixMetadata(
            model=execution.model, iterations=execution.iterations, tool_calls=execution.tool_calls,
            write_operations=execution.write_operations, duration_ms=execution.duration_ms,
        ),
    )


@router.get("/{repository_id}/agent/pr/{number}/fix", response_model=FixStatusResponse)
def fix_status(repository_id: uuid.UUID, number: PullNumber, current_user: CurrentUser, session: DbSession) -> FixStatusResponse:
    """Where the fix for this pull request stands in the workspace (`open` if there is none)."""

    repository = get_owned_repository(session, repository_id, current_user)
    state = fix_state.read_state(get_workspace_root(), repository.id)
    if state is None or state.pr_number != number:
        return FixStatusResponse(repository_id=repository.id, pull_request_number=number, status="open", branch=None, tests=None, commit=None)
    tests = None
    if state.tests:
        tests = TestResultResponse(passed=0, failed=0, errors=0, duration_ms=0, timed_out=False, output="", reason=None, **state.tests)
    return FixStatusResponse(repository_id=repository.id, pull_request_number=number, status=state.status, branch=state.branch, tests=tests, commit=state.commit)


@router.post("/{repository_id}/agent/pr/{number}/fix/commit", response_model=FixCommitResponse)
def commit_fix(repository_id: uuid.UUID, number: PullNumber, payload: FixCommitRequest, current_user: CurrentUser, session: DbSession) -> FixCommitResponse:
    """Commit the tested fix on the pull request's branch (requires `approved: true` and passing tests)."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    pull_request = _fetch_pull_request(repository, number)
    with exclusive_workspace(root, repository.id):
        result = fix_service.commit_fix(root, repository, current_user, pull_request, payload.message)
    return FixCommitResponse(repository_id=repository.id, pull_request_number=number, **asdict(result))


@router.post("/{repository_id}/agent/pr/{number}/fix/push", response_model=FixPushResponse)
def push_fix(repository_id: uuid.UUID, number: PullNumber, payload: FixPushRequest, current_user: CurrentUser, session: DbSession) -> FixPushResponse:
    """Push the committed fix to the pull request's own branch, never forced (requires `approved: true`)."""

    repository = get_owned_repository(session, repository_id, current_user)
    _require_approval(payload)
    root = get_workspace_root()
    pull_request = _fetch_pull_request(repository, number)
    with exclusive_workspace(root, repository.id):
        remote = workspace_service.resolve_remote(repository)
        result = fix_service.push_fix(root, repository, current_user, pull_request, remote.token)
    return FixPushResponse(repository_id=repository.id, pull_request_number=number, **asdict(result))
