"""Request and response schemas for executing an approved implementation plan."""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from app.schemas.agent import MAX_MESSAGE_CHARS, _reject_nul
from app.schemas.plan import ImplementationPlan


class ExecuteRequest(BaseModel):
    """The user's approval of a plan. Only a literal JSON `true` counts as approval."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, pattern=r"\S")
    plan: ImplementationPlan
    approved: StrictBool = False

    _no_nul = field_validator("message")(_reject_nul)


class FileChangeResponse(BaseModel):
    path: str
    action: Literal["created", "modified", "deleted"]
    additions: int
    deletions: int
    diff: str = Field(description="Unified diff of the net change, with secrets redacted.")
    diff_truncated: bool


class ExecuteMetadata(BaseModel):
    model: str
    iterations: int
    tool_calls: int
    write_operations: int
    duration_ms: int


class ExecuteResponse(BaseModel):
    """What the agent changed in the local working copy. Nothing is committed, pushed, or opened."""

    repository_id: uuid.UUID
    status: Literal["completed", "incomplete", "limit_reached"]
    changes: list[FileChangeResponse]
    summary: str
    branch: str | None = Field(description="The checkout's current branch.")
    uncommitted_changes: bool = Field(description="Whether the checkout has uncommitted changes (this run's and earlier ones).")
    committed: Literal[False] = False
    metadata: ExecuteMetadata
