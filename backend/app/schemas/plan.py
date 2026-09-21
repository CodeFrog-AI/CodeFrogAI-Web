"""Schemas for implementation plans.

A plan is an analysis artifact only: it describes what a change would involve and never
records that anything was done. The schema is strict: every field is required and unknown
fields are rejected, so a malformed model response fails validation instead of being
silently accepted (for example a response claiming `"changes_applied": true`).
"""

import re
import uuid
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

MAX_PATH_LENGTH = 2_048

_DRIVE_PATH = re.compile(r"^[A-Za-z]:")


def _repository_path(value: str) -> str:
    """A plain repository-relative path: no traversal, no absolute or drive paths, printable only."""

    if not value.isprintable() or chr(92) in value:
        raise ValueError("path must be a repository-relative path using '/'")
    if value.startswith("/") or _DRIVE_PATH.match(value):
        raise ValueError("path must be relative to the repository root")
    if any(segment in ("", ".", "..") for segment in value.split("/")):
        raise ValueError("path must not contain empty, '.' or '..' segments")
    return value


def _text(max_length: int) -> type[str]:
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length)]  # type: ignore[return-value]


RepositoryPath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PATH_LENGTH),
    AfterValidator(_repository_path),
]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: _text(200)
    description: _text(2_000)
    files: list[RepositoryPath] = Field(max_length=30)
    reason: _text(1_000)


class ImplementationPlan(BaseModel):
    """What a requested change would involve, based on the inspected repository."""

    model_config = ConfigDict(extra="forbid")

    summary: _text(1_000)
    steps: list[PlanStep] = Field(min_length=1, max_length=30)
    files_to_create: list[RepositoryPath] = Field(max_length=50)
    files_to_modify: list[RepositoryPath] = Field(max_length=50)
    files_to_delete: list[RepositoryPath] = Field(max_length=50)
    tests_to_add: list[_text(500)] = Field(max_length=30)
    risks: list[_text(1_000)] = Field(max_length=20)
    assumptions: list[_text(1_000)] = Field(max_length=20)


class PlanMetadata(BaseModel):
    model: str
    iterations: int
    tool_calls: int
    stop_reason: Literal["final_answer", "max_iterations", "empty_response"]
    duration_ms: int


class PlanResponse(BaseModel):
    """A validated plan. Nothing has been changed: `applied` is always false."""

    repository_id: uuid.UUID
    plan: ImplementationPlan
    warnings: list[str] = Field(
        description="Server-side checks of the plan against the indexed repository (unverified paths, etc.)."
    )
    applied: Literal[False] = False
    metadata: PlanMetadata
