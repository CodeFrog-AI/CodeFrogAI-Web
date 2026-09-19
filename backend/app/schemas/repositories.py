"""Response schemas for repository operations."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ScanEmbeddingResult(BaseModel):
    """Outcome of the automatic embedding step run after a scan.

    `failed` and `not_configured` mean the scan itself succeeded but semantic search
    is not up to date. Counts are None when the step did not complete.
    """

    status: Literal["completed", "not_configured", "failed"]
    chunks_embedded: int | None = None
    chunks_reused: int | None = None
    chunks_skipped: int | None = None
    message: str | None = None


class RepositoryScanResponse(BaseModel):
    """Outcome of a repository scan. Contains counts only, never GitHub credentials."""

    repository_id: uuid.UUID
    status: Literal["completed"]
    files_discovered: int
    files_indexed: int
    files_skipped: int
    files_removed: int
    chunks_created: int
    embeddings: ScanEmbeddingResult


class GitHubRepositoryResponse(BaseModel):
    """A GitHub repository the user can select, with its CodeFrog connection state."""

    github_repository_id: int
    owner: str
    name: str
    default_branch: str
    private: bool
    connected: bool


class GitHubRepositoryListResponse(BaseModel):
    repositories: list[GitHubRepositoryResponse]


class ConnectRepositoryRequest(BaseModel):
    # GitHub IDs are stored in a 32-bit INTEGER column.
    github_repository_id: int = Field(gt=0, le=2_147_483_647)


class ConnectedRepositoryResponse(BaseModel):
    """A connected repository. Never includes GitHub credentials."""

    id: uuid.UUID
    github_repository_id: int
    owner: str
    name: str
    default_branch: str
    private: bool
    connection_status: str


class CodeSearchResult(BaseModel):
    """One matching region of an indexed file."""

    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str


class CodeSearchResponse(BaseModel):
    repository_id: uuid.UUID
    query: str
    results: list[CodeSearchResult]


class EmbeddingIndexResponse(BaseModel):
    """Outcome of embedding a repository's chunks. Counts only; never vectors or keys."""

    repository_id: uuid.UUID
    status: Literal["completed"]
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    chunks_skipped: int


class SemanticSearchResult(BaseModel):
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    score: float = Field(description="Cosine similarity in [-1, 1]; higher means more relevant.")


class SemanticSearchResponse(BaseModel):
    repository_id: uuid.UUID
    query: str
    results: list[SemanticSearchResult]
