"""Response schemas for repository operations."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class RepositoryScanResponse(BaseModel):
    """Outcome of a repository scan. Contains counts only, never GitHub credentials."""

    repository_id: uuid.UUID
    status: Literal["completed"]
    files_discovered: int
    files_indexed: int
    files_skipped: int
    files_removed: int
    chunks_created: int


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
