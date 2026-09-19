"""Structured Git operations on one workspace checkout.

`GitRepository` is the only interface to Git: a fixed set of operations (status, diff, log,
create_branch, commit, push, fetch), each running one or two fixed subcommands through
`run_git`. Nothing here accepts a Git command, an option, or a path from a caller, so there
is no way to run arbitrary Git. Results are structured, repository-relative, redacted, and
bounded; raw Git output, remote URLs, and absolute paths are never returned.

Write operations (create_branch, commit, push) never run unless the API layer has verified
the user's explicit approval, and they only ever act on a `codefrog/` branch.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.context.redaction import is_sensitive_path, redact_secrets
from app.context.service import truncate_at_line
from app.git.runner import GitError, GitResult, run_git
from app.workspace.workspace import (
    MAX_DIFF_CHARS_PER_FILE,
    MAX_TOTAL_DIFF_CHARS,
    Workspace,
    WorkspaceError,
    diff_text,
    validate_path,
)

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "codefrog/"
PROTECTED_BRANCHES = frozenset({"main", "master"})
REMOTE = "origin"
MAX_BRANCH_LENGTH = 100
MAX_COMMIT_MESSAGE_CHARS = 2_000
MAX_COMMIT_SUBJECT_CHARS = 200
MAX_LISTED_PATHS = 1_000
MAX_DIFF_FILES = 50
MAX_LOG_ENTRIES = 50

NUL = chr(0)
RECORD_SEPARATOR = chr(30)
FIELD_SEPARATOR = chr(31)

_BRANCH_NAME = re.compile(r"^codefrog/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
_BASE_BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_CONFLICT_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})
_AUTH_MARKERS = ("authentication failed", "could not read username", "could not read password", "invalid credentials", "permission denied", "terminal prompts disabled", "http 401", "http 403", "error: 403", "returned error: 403")
_REJECTED_MARKERS = ("non-fast-forward", "fetch first", "[rejected]", "protected branch", "failed to push some refs")

FileStatus = Literal["modified", "added", "deleted", "untracked"]


def validate_branch_name(name: str) -> str:
    """A safe `codefrog/...` branch name, or GitError. Never a way to name main, master, or an option."""

    if (
        not isinstance(name, str)
        or len(name) > MAX_BRANCH_LENGTH
        or not _BRANCH_NAME.match(name)
        or ".." in name
        or name.endswith((".", ".lock"))
        or any(segment.endswith(".lock") for segment in name.split("/"))
    ):
        raise GitError("INVALID_BRANCH", "Branch names must look like 'codefrog/short-task-name' (letters, digits, '.', '_', '-', '/').", 400)
    return name


def validate_base_branch(name: str) -> str:
    """The repository's default branch as stored, checked before it is ever used as a Git argument."""

    if not name or len(name) > 255 or not _BASE_BRANCH_NAME.match(name) or ".." in name or name.endswith((".", ".lock", "/")):
        raise GitError("INVALID_BRANCH", "The repository's default branch name is not supported.", 400)
    return name


def validate_commit_message(message: str) -> str:
    """A clean commit message: one subject line (<=200 chars), optional body, no control characters or secrets."""

    if not isinstance(message, str):
        raise GitError("INVALID_MESSAGE", "The commit message must be text.", 400)
    text = message.replace(chr(13) + chr(10), chr(10)).strip()
    subject = text.split(chr(10), 1)[0].strip()
    if not subject or len(text) > MAX_COMMIT_MESSAGE_CHARS or len(subject) > MAX_COMMIT_SUBJECT_CHARS:
        raise GitError("INVALID_MESSAGE", f"The commit message needs a subject line of at most {MAX_COMMIT_SUBJECT_CHARS} characters and a total of at most {MAX_COMMIT_MESSAGE_CHARS}.", 400)
    if any(ord(character) < 32 and character not in (chr(10), chr(9)) or ord(character) == 127 for character in text):
        raise GitError("INVALID_MESSAGE", "The commit message must not contain control characters.", 400)
    if redact_secrets(text)[1]:
        raise GitError("INVALID_MESSAGE", "The commit message appears to contain a secret.", 400)
    return text


@dataclass(frozen=True)
class GitStatus:
    branch: str | None
    modified: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    conflicted: list[str] = field(default_factory=list)
    withheld: int = 0  # changed paths that are protected or unsafe: counted, never named
    truncated: bool = False

    @property
    def clean(self) -> bool:
        return not (self.modified or self.added or self.deleted or self.untracked or self.conflicted or self.withheld)

    @property
    def changed_files(self) -> int:
        return len(self.modified) + len(self.added) + len(self.deleted) + len(self.untracked) + len(self.conflicted) + self.withheld


@dataclass(frozen=True)
class DiffFile:
    path: str
    status: FileStatus
    additions: int
    deletions: int
    diff: str
    diff_truncated: bool
    binary: bool = False


@dataclass(frozen=True)
class GitDiff:
    files: list[DiffFile]
    withheld: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class LogEntry:
    commit: str
    subject: str
    author: str
    date: str


@dataclass(frozen=True)
class CommitResult:
    commit: str
    branch: str
    files_changed: int


@dataclass(frozen=True)
class PushResult:
    branch: str
    commit: str


class GitRepository:
    def __init__(
        self,
        root: Path,
        *,
        default_branch: str,
        remote_url: str,
        author_name: str = "CodeFrog",
        author_email: str = "codefrog@users.noreply.github.com",
    ) -> None:
        self.root = Path(root)
        self.default_branch = default_branch
        self.remote_url = remote_url
        self._author = (_clean_identity(author_name) or "CodeFrog", _clean_identity(author_email) or "codefrog@users.noreply.github.com")

    # ------------------------------------------------------------------ reading

    def verify_origin(self) -> None:
        """The checkout must belong to this repository: its origin is exactly the expected clean URL."""

        result = self._run(["config", "--get", f"remote.{REMOTE}.url"])
        if not result.ok or result.stdout.strip() != self.remote_url:
            raise GitError("WORKSPACE_MISMATCH", "The workspace does not belong to this repository.", 409)

    def current_branch(self) -> str | None:
        result = self._run(["symbolic-ref", "--short", "-q", "HEAD"])
        return result.stdout.strip() or None if result.ok else None

    def head_commit(self) -> str | None:
        result = self._run(["rev-parse", "--verify", "-q", "HEAD"])
        return result.stdout.strip() if result.ok else None

    def status(self) -> GitStatus:
        result = self._run(["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"])
        if not result.ok:
            raise GitError("GIT_ERROR", "The repository status could not be read.")
        buckets: dict[str, list[str]] = {"modified": [], "added": [], "deleted": [], "untracked": [], "conflicted": []}
        withheld = 0
        for entry in result.stdout.split(NUL):
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            if code == "!!":
                continue
            if not _listable(path):
                withheld += 1
                continue
            if code == "??":
                buckets["untracked"].append(path)
            elif code in _CONFLICT_CODES:
                buckets["conflicted"].append(path)
            elif "D" in code:
                buckets["deleted"].append(path)
            elif "A" in code:
                buckets["added"].append(path)
            else:
                buckets["modified"].append(path)
        truncated = any(len(paths) > MAX_LISTED_PATHS for paths in buckets.values())
        return GitStatus(
            branch=self.current_branch(),
            withheld=withheld,
            truncated=truncated,
            **{name: sorted(paths)[:MAX_LISTED_PATHS] for name, paths in buckets.items()},
        )

    def diff(self) -> GitDiff:
        """Every uncommitted change against HEAD, as bounded, redacted unified diffs."""

        status = self.status()
        entries: list[tuple[str, FileStatus]] = sorted(
            [(path, "modified") for path in status.modified]
            + [(path, "added") for path in status.added]
            + [(path, "deleted") for path in status.deleted]
            + [(path, "untracked") for path in status.untracked]
        )
        reader = Workspace.attach(self.root, uuid.UUID(int=0))
        files: list[DiffFile] = []
        budget = MAX_TOTAL_DIFF_CHARS
        for path, kind in entries[:MAX_DIFF_FILES]:
            limit = max(min(MAX_DIFF_CHARS_PER_FILE, budget), 0)
            entry = self._untracked_diff(reader, path, limit) if kind == "untracked" else self._tracked_diff(path, kind, limit)
            budget -= len(entry.diff)
            files.append(entry)
        return GitDiff(files, withheld=status.withheld, truncated=len(entries) > MAX_DIFF_FILES or status.truncated)

    def log(self, limit: int = 10) -> list[LogEntry]:
        if self.head_commit() is None:
            return []
        count = max(1, min(limit, MAX_LOG_ENTRIES))
        result = self._run(["log", f"-n{count}", "--no-color", f"--format=%H{FIELD_SEPARATOR}%an{FIELD_SEPARATOR}%aI{FIELD_SEPARATOR}%s{RECORD_SEPARATOR}"])
        if not result.ok:
            raise GitError("GIT_ERROR", "The commit history could not be read.")
        entries = []
        for record in result.stdout.split(RECORD_SEPARATOR):
            parts = record.strip().split(FIELD_SEPARATOR)
            if len(parts) == 4:
                entries.append(LogEntry(parts[0], redact_secrets(parts[3])[0], parts[1], parts[2]))
        return entries

    # ------------------------------------------------------------------ writing (approval is checked by the caller)

    def create_branch(self, name: str) -> str:
        """Create and switch to a new `codefrog/` branch from the current commit; never overwrites one."""

        name = validate_branch_name(name)
        if name in PROTECTED_BRANCHES or name == self.default_branch:
            raise GitError("PROTECTED_BRANCH", "That branch name is protected.", 403)
        if not self._run(["check-ref-format", "--branch", name]).ok:
            raise GitError("INVALID_BRANCH", "That is not a valid branch name.", 400)
        if self.head_commit() is None:
            raise GitError("GIT_ERROR", "The workspace has no commits yet.", 409)
        if self._run(["show-ref", "--verify", "--quiet", f"refs/heads/{name}"]).ok:
            raise GitError("BRANCH_EXISTS", "A branch with that name already exists.", 409)
        if not self._run(["switch", "--create", name]).ok:
            raise GitError("GIT_ERROR", "The branch could not be created.")
        return name

    def commit(self, message: str) -> CommitResult:
        """Commit every current change on the CodeFrog branch. Refuses protected branches and protected files."""

        message = validate_commit_message(message)
        branch = self._require_codefrog_branch("Commits are only allowed on a CodeFrog branch. Create one first.")
        status = self.status()
        if status.conflicted:
            raise GitError("MERGE_CONFLICT", "Resolve conflicts before committing.", 409)
        if status.withheld:
            raise GitError("PROTECTED_PATH", "Some changed files are protected and cannot be committed.", 403)
        if status.clean:
            raise GitError("NO_CHANGES", "There are no changes to commit.", 409)
        if not self._run(["add", "--all", "--", "."]).ok:
            raise GitError("GIT_ERROR", "The changes could not be staged.")
        name, email = self._author
        result = self._run(
            ["commit", "--no-verify", "--message", message],
            config=(f"user.name={name}", f"user.email={email}"),
        )
        if not result.ok:
            self._run(["reset", "--quiet"])
            raise GitError("COMMIT_FAILED", "The commit could not be created.", 409)
        commit = self.head_commit()
        if commit is None:
            raise GitError("COMMIT_FAILED", "The commit could not be created.", 409)
        return CommitResult(commit=commit, branch=branch, files_changed=status.changed_files)

    def push(self, token: str | None) -> PushResult:
        """Push only the current CodeFrog branch to origin, never forced. Failures are classified, never echoed."""

        branch = self._require_codefrog_branch("Only CodeFrog branches can be pushed.")
        self.verify_origin()
        commit = self.head_commit()
        if commit is None:
            raise GitError("GIT_ERROR", "The workspace has no commits yet.", 409)
        refspec = f"refs/heads/{branch}:refs/heads/{branch}"
        result = self._run(["push", "--set-upstream", REMOTE, refspec], token=token, network=True)
        if not result.ok:
            reason = result.stderr.lower()
            if any(marker in reason for marker in _AUTH_MARKERS):
                raise GitError("GITHUB_AUTH_FAILED", "GitHub rejected the stored credentials. Reconnect your GitHub account and try again.", 403)
            if any(marker in reason for marker in _REJECTED_MARKERS):
                raise GitError("PUSH_REJECTED", "The remote rejected the push.", 409)
            raise GitError("PUSH_FAILED", "The branch could not be pushed. Try again later.", 502)
        return PushResult(branch=branch, commit=commit)

    def fetch(self, token: str | None) -> None:
        self.verify_origin()
        if not self._run(["fetch", "--no-tags", REMOTE], token=token, network=True).ok:
            raise GitError("FETCH_FAILED", "The repository could not be fetched. Try again later.", 502)

    def fast_forward_default_branch(self) -> bool:
        """Advance a clean checkout of the default branch to origin's tip. Returns whether it moved."""

        before = self.head_commit()
        result = self._run(["merge", "--ff-only", f"{REMOTE}/{validate_base_branch(self.default_branch)}"])
        if not result.ok:
            raise GitError("SYNC_FAILED", "The workspace could not be fast-forwarded.", 409)
        return self.head_commit() != before

    # ------------------------------------------------------------------ internals

    def _run(self, arguments: list[str], *, token: str | None = None, network: bool = False, config: tuple[str, ...] = ()) -> GitResult:
        return run_git(self.root, arguments, token=token, network=network, config=config)

    def _require_codefrog_branch(self, message: str) -> str:
        branch = self.current_branch()
        if branch is None or branch in PROTECTED_BRANCHES or branch == self.default_branch or not branch.startswith(BRANCH_PREFIX):
            raise GitError("PROTECTED_BRANCH", message, 403)
        return branch

    def _tracked_diff(self, path: str, kind: FileStatus, limit: int) -> DiffFile:
        result = self._run(["diff", "HEAD", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=3", "--", path])
        if not result.ok:
            raise GitError("GIT_ERROR", "The diff could not be read.")
        text = result.stdout
        binary = text.startswith("Binary files") or "\nBinary files" in text or "GIT binary patch" in text
        lines = text.split(chr(10))
        additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        body, cut = truncate_at_line(redact_secrets("" if binary else text)[0], limit)
        return DiffFile(path, kind, additions, deletions, body, cut, binary)

    def _untracked_diff(self, reader: Workspace, path: str, limit: int) -> DiffFile:
        try:
            content = reader.read_text(path)
        except WorkspaceError:
            return DiffFile(path, "untracked", 0, 0, "", False, binary=True)  # unreadable or not UTF-8 text
        body, additions, _, cut = diff_text(path, None, content or "", limit)
        return DiffFile(path, "untracked", additions, 0, body, cut)


def _listable(path: str) -> bool:
    """Whether a changed path may be named to the client (safe shape, not a protected file)."""

    try:
        validate_path(path)
    except WorkspaceError:
        return False
    return not is_sensitive_path(path)


def _clean_identity(value: str) -> str:
    return "".join(character for character in value if character.isprintable() and character not in "<>").strip()[:100]
