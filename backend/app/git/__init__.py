"""A small, structured Git layer: the only code that starts Git processes."""

from app.git.repository import (
    BRANCH_PREFIX,
    CommitResult,
    DiffFile,
    GitDiff,
    GitRepository,
    GitStatus,
    LogEntry,
    PushResult,
    validate_base_branch,
    validate_branch_name,
    validate_commit_message,
)
from app.git.runner import GitError

__all__ = [
    "BRANCH_PREFIX",
    "CommitResult",
    "DiffFile",
    "GitDiff",
    "GitError",
    "GitRepository",
    "GitStatus",
    "LogEntry",
    "PushResult",
    "validate_base_branch",
    "validate_branch_name",
    "validate_commit_message",
]
