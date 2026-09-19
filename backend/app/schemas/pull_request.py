"""Strict schemas for pull request creation, inspection, diffs, and AI review.

Nothing here carries tokens, authorization headers, filesystem paths, or raw GitHub
responses.
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from app.schemas.git import ApprovalRequest
from app.schemas.plan import RepositoryPath

MAX_TITLE_CHARS = 200
MAX_BODY_CHARS = 10_000
_ALLOWED_BODY_CONTROLS = (chr(9), chr(10), chr(13))


def _has_control_characters(value: str, allowed: tuple[str, ...] = ()) -> bool:
    return any((ord(c) < 32 or ord(c) == 127) and c not in allowed for c in value)


class CreatePullRequestRequest(ApprovalRequest):
    """Only the title and body come from the caller. Head, base, repository, and credentials are server-decided."""

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS, pattern=r"\S")
    body: str = Field(default="", max_length=MAX_BODY_CHARS)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        value = value.strip()
        if not value or _has_control_characters(value):
            raise ValueError("title must be a single line without control characters")
        return value

    @field_validator("body")
    @classmethod
    def clean_body(cls, value: str) -> str:
        if _has_control_characters(value, _ALLOWED_BODY_CONTROLS):
            raise ValueError("body must not contain control characters")
        return value.strip()


class ReviewRequest(ApprovalRequest):
    """`approved` means: the user explicitly asked for an AI review. It approves no code changes."""


class PullRequestSummaryResponse(BaseModel):
    repository_id: uuid.UUID
    number: int
    url: str
    title: str
    head_branch: str
    base_branch: str
    state: Literal["open", "closed", "merged"]
    created: bool = Field(description="False when an open pull request for this branch already existed and was returned instead.")
    redactions: int = Field(description="Secret-looking values removed from the title and body before sending.")


class PullRequestResponse(BaseModel):
    repository_id: uuid.UUID
    number: int
    url: str
    title: str
    body: str
    body_redactions: int
    state: Literal["open", "closed", "merged"]
    draft: bool
    head_branch: str
    base_branch: str
    created_at: str | None
    updated_at: str | None


class PullRequestFileResponse(BaseModel):
    path: str
    status: Literal["added", "modified", "deleted", "renamed"]
    additions: int
    deletions: int
    patch: str = Field(description="Unified patch, secrets redacted, cut at a size limit.")
    patch_truncated: bool
    patch_available: bool = Field(description="False for binary or very large files GitHub gives no patch for.")
    previous_path: str | None


class PullRequestDiffResponse(BaseModel):
    repository_id: uuid.UUID
    number: int
    files: list[PullRequestFileResponse]
    total_files: int
    withheld: int = Field(description="Protected or unsafe files: counted, never named or shown.")
    truncated: bool


# ------------------------------------------------------------------ the AI review


def _text(max_length: int):
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length)]


Severity = Literal["critical", "high", "medium", "low", "info"]


class ReviewFinding(BaseModel):
    """One observation. `severity` is a category, not a score or a ranking."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    kind: Literal["confirmed_issue", "suggestion"]
    title: _text(200)
    description: _text(2_000)
    evidence: _text(1_000)
    file: RepositoryPath | None
    line: int | None = Field(strict=True, ge=1, le=10_000_000)
    recommendation: _text(1_000)

    @model_validator(mode="after")
    def line_needs_a_file(self) -> "ReviewFinding":
        if self.line is not None and self.file is None:
            raise ValueError("a line number needs a file")
        return self


class ReviewTests(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing: list[_text(500)] = Field(max_length=20)
    suggested: list[_text(500)] = Field(max_length=20)


class PullRequestReview(BaseModel):
    """A structured review. Every field is required and unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    summary: _text(2_000)
    findings: list[ReviewFinding] = Field(max_length=30)
    tests: ReviewTests
    risks: list[_text(1_000)] = Field(max_length=20)
    overall: _text(1_000)


class ReviewMetadata(BaseModel):
    model: str
    iterations: int
    tool_calls: int
    stop_reason: Literal["final_answer", "max_iterations", "empty_response"]
    files_reviewed: int
    duration_ms: int


class ReviewResponse(BaseModel):
    """A read-only review. `changes_made` is always false: findings are never applied automatically."""

    repository_id: uuid.UUID
    pull_request_number: int
    review: PullRequestReview
    warnings: list[str]
    changes_made: Literal[False] = False
    metadata: ReviewMetadata
