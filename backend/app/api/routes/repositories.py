"""Repository resource routes."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.agent.executor import execute_plan
from app.agent.llm import LLMError, LLMNotConfiguredError, get_llm_provider
from app.agent.planner import create_plan
from app.agent.service import run_agent
from app.analyzer.service import analyze_after_scan, get_repository_analysis
from app.auth.dependencies import get_current_user
from app.context.service import build_repository_context
from app.core.config import get_settings
from app.core.exceptions import (
    BadGatewayError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
)
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, User
from app.embeddings.indexing import index_after_scan, index_repository_embeddings
from app.embeddings.provider import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    get_embedding_provider,
)
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
from app.scanner.semantic import (
    DEFAULT_SEMANTIC_LIMIT,
    MAX_SEMANTIC_LIMIT,
    MAX_SEMANTIC_QUERY_LENGTH,
    semantic_search,
)
from app.scanner.service import get_owned_repository, scan_repository
from app.workspace import get_workspace_root
from app.workspace import service as workspace_service
from app.schemas.availability import ResourceAvailabilityResponse
from app.schemas.agent import AgentMetadata, AgentRequest, AgentResponse, AgentToolCall
from app.schemas.context import ContextRequest, RepositoryContextResponse
from app.schemas.execution import ExecuteMetadata, ExecuteRequest, ExecuteResponse, FileChangeResponse
from app.schemas.plan import PlanMetadata, PlanResponse
from app.schemas.repositories import (
    EmbeddingIndexResponse,
    SemanticSearchResponse,
    SemanticSearchResult,
    CodeSearchResponse,
    CodeSearchResult,
    ConnectedRepositoryResponse,
    ProjectAnalysisResponse,
    ConnectRepositoryRequest,
    GitHubRepositoryListResponse,
    GitHubRepositoryResponse,
    RepositoryScanResponse,
    ScanEmbeddingResult,
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
            analyze_after_scan(session, repository, client)
    except GitHubAuthError:
        raise ForbiddenError(
            "GitHub access could not be verified. Reconnect GitHub and try again."
        ) from None
    except GitHubNotFoundError:
        raise NotFoundError("Repository was not found on GitHub or is not accessible") from None
    except GitHubContentError:
        raise BadGatewayError("GitHub request failed. Try again later.") from None
    outcome = index_after_scan(session, repository, get_embedding_provider)
    embedding_counts = vars(outcome.summary) if outcome.summary else {}
    return RepositoryScanResponse(
        **vars(summary),
        embeddings=ScanEmbeddingResult(
            status=outcome.status, message=outcome.message, **embedding_counts
        ),
    )


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


@contextmanager
def _embedding_errors() -> Iterator[None]:
    """Translate embedding provider failures into safe, client-facing API errors."""

    try:
        yield
    except EmbeddingNotConfiguredError:
        raise ServiceUnavailableError("Semantic search is not configured") from None
    except EmbeddingError:
        raise BadGatewayError("Embedding provider request failed. Try again later.") from None


@router.post("/{repository_id}/embeddings", response_model=EmbeddingIndexResponse)
def index_repository_chunk_embeddings(
    repository_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> EmbeddingIndexResponse:
    """Generate embeddings for new or changed chunks of a repository owned by the caller."""

    repository = get_owned_repository(session, repository_id, current_user)
    with _embedding_errors():
        summary = index_repository_embeddings(session, repository, get_embedding_provider())
    return EmbeddingIndexResponse(
        repository_id=repository.id, status="completed", **vars(summary)
    )


@router.get("/{repository_id}/semantic-search", response_model=SemanticSearchResponse)
def semantic_search_repository(
    repository_id: uuid.UUID,
    query: Annotated[str, Query(min_length=1, max_length=MAX_SEMANTIC_QUERY_LENGTH, pattern=r"\S")],
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=MAX_SEMANTIC_LIMIT)] = DEFAULT_SEMANTIC_LIMIT,
    min_score: Annotated[float | None, Query(ge=-1, le=1)] = None,
) -> SemanticSearchResponse:
    """Find code chunks semantically related to a natural-language query."""

    repository = get_owned_repository(session, repository_id, current_user)
    search_text = query.strip()
    with _embedding_errors():
        hits = semantic_search(
            session, repository, get_embedding_provider(), search_text, limit, min_score
        )
    return SemanticSearchResponse(
        repository_id=repository.id,
        query=search_text,
        results=[SemanticSearchResult(**vars(hit)) for hit in hits],
    )


@router.get("/{repository_id}/analysis", response_model=ProjectAnalysisResponse)
def get_project_analysis(
    repository_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ProjectAnalysisResponse:
    """Return the stored project analysis of a repository owned by the caller."""

    repository = get_owned_repository(session, repository_id, current_user)
    analysis = get_repository_analysis(session, repository)
    if analysis is None:
        raise ConflictError("Repository has not been analyzed yet. Scan the repository first.")
    return ProjectAnalysisResponse.from_analysis(repository.id, analysis)


@router.post("/{repository_id}/context", response_model=RepositoryContextResponse)
def build_context(
    repository_id: uuid.UUID,
    payload: ContextRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> RepositoryContextResponse:
    """Build sanitized, AI-ready context (analysis, exact and semantic hits) for a question.

    Semantic search is optional: when embeddings are missing, unconfigured, or failing,
    the response still succeeds and reports the semantic status in `retrieval`.
    """

    repository = get_owned_repository(session, repository_id, current_user)
    context = build_repository_context(
        session, repository, get_embedding_provider, **payload.model_dump()
    )
    return RepositoryContextResponse.model_validate(asdict(context))


@contextmanager
def _agent_errors() -> Iterator[None]:
    """Translate LLM provider failures into safe, client-facing API errors."""

    try:
        yield
    except LLMNotConfiguredError:
        raise ServiceUnavailableError("The AI agent is not configured") from None
    except LLMError:
        raise BadGatewayError("The AI provider request failed. Try again later.") from None


@router.post("/{repository_id}/agent", response_model=AgentResponse)
def ask_agent(
    repository_id: uuid.UUID,
    payload: AgentRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> AgentResponse:
    """Answer a question about the repository with a tool-calling agent (read-only)."""

    repository = get_owned_repository(session, repository_id, current_user)
    with _agent_errors():
        result = run_agent(
            session,
            current_user,
            repository,
            payload.message,
            get_llm_provider(),
            get_embedding_provider,
            history=[item.model_dump() for item in payload.history],
            max_iterations=get_settings().agent_max_iterations,
            checkout=workspace_service.open_checkout(get_workspace_root(), repository),
        )
    return AgentResponse(
        repository_id=repository.id,
        answer=result.answer,
        tool_calls=[AgentToolCall(**vars(call)) for call in result.tool_calls],
        metadata=AgentMetadata(
            iterations=result.iterations,
            tool_calls=len(result.tool_calls),
            stop_reason=result.stop_reason,
            model=result.model,
            duration_ms=result.duration_ms,
        ),
    )


@router.post("/{repository_id}/agent/plan", response_model=PlanResponse)
def plan_change(
    repository_id: uuid.UUID,
    payload: AgentRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> PlanResponse:
    """Inspect the repository (read-only) and return a structured implementation plan.

    The plan is only returned: nothing is modified, branched, committed, pushed, or opened.
    """

    repository = get_owned_repository(session, repository_id, current_user)
    with _agent_errors():
        result = create_plan(
            session,
            current_user,
            repository,
            payload.message,
            get_llm_provider(),
            get_embedding_provider,
            history=[item.model_dump() for item in payload.history],
            max_iterations=get_settings().agent_max_iterations,
            checkout=workspace_service.open_checkout(get_workspace_root(), repository),
        )
    return PlanResponse(
        repository_id=repository.id,
        plan=result.plan,
        warnings=result.warnings,
        metadata=PlanMetadata(
            model=result.model,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            stop_reason=result.stop_reason,
            duration_ms=result.duration_ms,
        ),
    )


@router.post("/{repository_id}/agent/execute", response_model=ExecuteResponse)
def execute_change(
    repository_id: uuid.UUID,
    payload: ExecuteRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> ExecuteResponse:
    """Carry out a plan the user approved, editing a local working copy and returning the diff.

    Approval is decided here, by the server, from the authenticated user's request; the model
    never sees or sets it. Nothing is committed, pushed, branched, or opened on GitHub.
    """

    repository = get_owned_repository(session, repository_id, current_user)
    if payload.approved is not True:
        raise ForbiddenError("The plan must be explicitly approved before changes are made")
    with _agent_errors():
        result = execute_plan(
            session,
            current_user,
            repository,
            payload.message,
            payload.plan,
            get_llm_provider(),
            get_embedding_provider,
            workspace_root=get_workspace_root(),
            max_iterations=get_settings().agent_max_iterations,
        )
    return ExecuteResponse(
        repository_id=repository.id,
        status=result.status,
        branch=result.branch,
        uncommitted_changes=result.uncommitted_changes,
        changes=[FileChangeResponse(**vars(change)) for change in result.changes],
        summary=result.summary,
        metadata=ExecuteMetadata(
            model=result.model,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            write_operations=result.write_operations,
            duration_ms=result.duration_ms,
        ),
    )
