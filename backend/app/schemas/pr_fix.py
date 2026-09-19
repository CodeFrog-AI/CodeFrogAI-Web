"""Strict schemas for fixing a pull request review finding.

A finding is only accepted exactly as CodeFrog's own review returned it (with the head commit
that was reviewed and the review's signature); a plan only as the fix-plan endpoint returned
it. Responses carry repository-relative paths and bounded, redacted text only.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.git import ApprovalRequest, CommitRequest, PushRequest
from app.schemas.plan import ImplementationPlan, PlanMetadata
from app.schemas.pull_request import ReviewFinding

FixStatus = Literal[
    "open", "planning", "approved", "fixing", "changes_ready", "test_failed", "ready_to_commit", "committed", "pushed", "no_changes"
]


class SelectedFinding(ReviewFinding):
    """A finding chosen by the user, with the proof that it came from a CodeFrog review of this pull request."""

    head_sha: str = Field(pattern=r"^[0-9a-f]{7,64}$", description="The pull request head commit the review was based on.")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$", description="The signature returned with the review for this finding.")


class FixPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding: SelectedFinding


class FixPlanResponse(BaseModel):
    """A plan to fix one finding. Nothing has been changed."""

    repository_id: uuid.UUID
    pull_request_number: int
    status: Literal["planning"] = "planning"
    head_sha: str
    finding: ReviewFinding
    plan: ImplementationPlan
    plan_signature: str = Field(description="Send this back with the plan to approve it. The plan cannot be edited.")
    warnings: list[str]
    applied: Literal[False] = False
    metadata: PlanMetadata


class FixRequest(ApprovalRequest):
    """Approval to apply the fix plan. `approved` must be exactly true; the plan is the one returned by fix-plan."""

    finding: SelectedFinding
    plan: ImplementationPlan
    plan_signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class TestResultResponse(BaseModel):
    __test__ = False

    status: Literal["passed", "failed", "not_run"]
    command: str | None
    passed: int
    failed: int
    errors: int
    duration_ms: int
    timed_out: bool
    output: str = Field(description="The end of the test output, secrets redacted, bounded.")
    reason: str | None


class FixChange(BaseModel):
    path: str
    status: Literal["modified", "added", "deleted", "untracked"]
    additions: int
    deletions: int
    diff: str
    diff_truncated: bool


class FixMetadata(BaseModel):
    model: str
    iterations: int
    tool_calls: int
    write_operations: int
    duration_ms: int


class FixResponse(BaseModel):
    """The result of applying a fix to the local workspace. Nothing is committed or pushed."""

    repository_id: uuid.UUID
    pull_request_number: int
    status: FixStatus
    branch: str
    head_sha: str
    agent_status: Literal["completed", "incomplete", "limit_reached"]
    summary: str
    changes: list[FixChange]
    withheld: int
    tests: TestResultResponse
    warnings: list[str]
    committed: Literal[False] = False
    pushed: Literal[False] = False
    metadata: FixMetadata


class FixCommitRequest(CommitRequest):
    pass


class FixPushRequest(PushRequest):
    pass


class FixCommitResponse(BaseModel):
    repository_id: uuid.UUID
    pull_request_number: int
    status: Literal["committed"] = "committed"
    commit: str
    branch: str
    files_changed: int


class FixPushResponse(BaseModel):
    repository_id: uuid.UUID
    pull_request_number: int
    status: Literal["pushed"] = "pushed"
    branch: str
    commit: str
    review_again: Literal[True] = Field(default=True, description="The pull request has updated; a new review can be requested.")


class FixStatusResponse(BaseModel):
    repository_id: uuid.UUID
    pull_request_number: int
    status: FixStatus
    branch: str | None
    tests: TestResultResponse | None
    commit: str | None
