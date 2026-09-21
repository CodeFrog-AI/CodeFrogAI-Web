"""Strict request and response schemas for the workspace and Git endpoints.

Responses contain repository-relative paths and structured facts only: never absolute
filesystem paths, remote URLs, credentials, or raw Git output.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class ApprovalRequest(BaseModel):
    """Only a literal JSON `true` counts as approval; missing or false is refused by the server."""

    model_config = ConfigDict(extra="forbid")

    approved: StrictBool = False


class BranchRequest(ApprovalRequest):
    name: str = Field(min_length=1, max_length=100, description="A branch name such as 'codefrog/fix-auth'.")


class CommitRequest(ApprovalRequest):
    message: str = Field(min_length=1, max_length=4_000)


class PushRequest(ApprovalRequest):
    pass


class WorkspaceStatusResponse(BaseModel):
    repository_id: uuid.UUID
    exists: bool
    branch: str | None
    default_branch: str | None
    uncommitted_changes: bool
    commit: str | None


class SyncResponse(BaseModel):
    repository_id: uuid.UUID
    action: Literal["up_to_date", "fast_forwarded", "skipped"]
    reason: str | None


class GitStatusResponse(BaseModel):
    repository_id: uuid.UUID
    branch: str | None
    clean: bool
    modified: list[str]
    added: list[str]
    deleted: list[str]
    untracked: list[str]
    conflicted: list[str]
    withheld: int = Field(description="Changed files that are protected or unsafe: counted but never named.")
    truncated: bool


class GitDiffFile(BaseModel):
    path: str
    status: Literal["modified", "added", "deleted", "untracked"]
    additions: int
    deletions: int
    diff: str = Field(description="Unified diff with secrets redacted, cut at a size limit.")
    diff_truncated: bool
    binary: bool


class GitDiffResponse(BaseModel):
    repository_id: uuid.UUID
    files: list[GitDiffFile]
    withheld: int
    truncated: bool


class GitLogEntry(BaseModel):
    commit: str
    subject: str
    author: str
    date: str


class GitLogResponse(BaseModel):
    repository_id: uuid.UUID
    commits: list[GitLogEntry]


class BranchResponse(BaseModel):
    repository_id: uuid.UUID
    branch: str
    created: Literal[True] = True


class CommitResponse(BaseModel):
    repository_id: uuid.UUID
    commit: str
    branch: str
    files_changed: int


class PushResponse(BaseModel):
    repository_id: uuid.UUID
    branch: str
    commit: str
    pushed: Literal[True] = True
