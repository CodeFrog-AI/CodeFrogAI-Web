"""A local working tree for approved code edits: the only place CodeFrog writes files.

The workspace is a server-side directory holding a copy of a repository's indexed files.
It is rebuilt from the scanner's index at the start of every execution, so edits never
drift from, or leak into, the source of truth. Every path is validated here, every write
is confined to the workspace directory, and all limits are enforced here, so the rules
live in one place. Nothing in this module talks to GitHub or runs Git.
"""

import difflib
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.context.redaction import is_sensitive_path, redact_secrets
from app.context.service import truncate_at_line
from app.core.config import get_settings
from app.core.exceptions import ApplicationError, ConflictError, ForbiddenError
from app.db.models import Repository, RepositoryChunk, RepositoryFile
from app.schemas.plan import ImplementationPlan, RepositoryPath

logger = logging.getLogger(__name__)

MAX_WRITE_OPERATIONS = 20
MAX_TOTAL_BYTES_CHANGED = 1_000_000
MAX_FILE_BYTES = 256 * 1024
MAX_DIFF_CHARS_PER_FILE = 20_000
MAX_TOTAL_DIFF_CHARS = 100_000
MAX_PREVIEW_CHARS = 4_000

PROTECTED_MESSAGE = "This path is protected and cannot be modified"

_PATH = TypeAdapter(RepositoryPath)
_HOSTILE_CHARACTERS = frozenset('<>:"|?*')
_RESERVED_NAMES = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)

Action = Literal["created", "modified", "deleted"]


class WorkspaceError(ApplicationError):
    """A refused or failed file operation. Messages never contain filesystem paths."""

    status_code = 500
    code = "WORKSPACE_ERROR"

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_path(path: str) -> str:
    """A safe repository-relative path, or an INVALID_PATH error.

    Reuses the plan schema's path rules (no traversal, absolute or drive paths, backslashes,
    control characters, or empty segments) and adds workspace-specific ones: nothing inside
    `.git`, and no names that are special on Windows (drives, streams, device names).
    """

    try:
        relative = _PATH.validate_python(path)
    except ValidationError:
        raise WorkspaceError("INVALID_PATH", "The path is not a valid repository-relative path.") from None
    for segment in relative.split("/"):
        if (
            segment.lower() == ".git"
            or segment != segment.rstrip(" .")
            or _HOSTILE_CHARACTERS & set(segment)
            or _RESERVED_NAMES.match(segment)
        ):
            raise WorkspaceError("INVALID_PATH", "The path is not writable.")
    return relative


@dataclass(frozen=True)
class WriteScope:
    """Which files an approved plan lets the agent create, edit, and delete."""

    create: frozenset[str]
    modify: frozenset[str]
    delete: frozenset[str]

    @classmethod
    def from_plan(cls, plan: ImplementationPlan) -> "WriteScope":
        return cls(frozenset(plan.files_to_create), frozenset(plan.files_to_modify), frozenset(plan.files_to_delete))


@dataclass(frozen=True)
class FileOperation:
    """The result of one successful write, small enough to show the model."""

    path: str
    action: Action
    bytes_written: int
    additions: int
    deletions: int
    diff: str


@dataclass(frozen=True)
class FileChange:
    """The net effect of an execution on one file, relative to how it started."""

    path: str
    action: Action
    additions: int
    deletions: int
    diff: str
    diff_truncated: bool


class Workspace:
    def __init__(
        self,
        root: Path,
        repository_id: uuid.UUID,
        scope: WriteScope,
        *,
        max_write_operations: int | None = None,
        max_total_bytes: int | None = None,
    ) -> None:
        self.root = root
        self.repository_id = repository_id
        self.scope = scope
        self.max_write_operations = MAX_WRITE_OPERATIONS if max_write_operations is None else max_write_operations
        self.max_total_bytes = MAX_TOTAL_BYTES_CHANGED if max_total_bytes is None else max_total_bytes
        self.write_operations = 0
        self.bytes_changed = 0
        self.limit_reached = False
        self._originals: dict[str, str | None] = {}

    @classmethod
    def prepare(cls, base_directory: Path, repository_id: uuid.UUID, scope: WriteScope, **limits: int) -> "Workspace":
        """Make a fresh, empty working tree for one repository under `base_directory`."""

        try:
            base = Path(base_directory).resolve()
            base.mkdir(parents=True, exist_ok=True)
            root = base / str(repository_id)
            if root.is_symlink():
                root.unlink()
            elif root.exists():
                shutil.rmtree(root)
            root.mkdir()
            return cls(root.resolve(), repository_id, scope, **limits)
        except OSError as error:
            logger.warning("Workspace could not be prepared (exception type=%s)", type(error).__name__)
            raise WorkspaceError("WORKSPACE_ERROR", "The workspace could not be prepared.") from None

    def populate(self, session: Session, repository: Repository) -> int:
        """Copy the repository's indexed files into the workspace; returns how many.

        Sensitive files (the index may hold config files with secrets) and paths that
        fail validation are never copied.
        """

        rows = (
            session.query(RepositoryFile.path, RepositoryChunk.content)
            .outerjoin(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
            .filter(RepositoryFile.repository_id == repository.id)
            .order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
            .yield_per(500)
        )
        copied = 0
        current: str | None = None
        parts: list[str] = []

        def flush() -> int:
            return int(current is not None and self._copy_in(current, "".join(parts)))

        for path, content in rows:
            if path != current:
                copied += flush()
                current, parts = path, []
            if content is not None:
                parts.append(content)
        return copied + flush()

    def _copy_in(self, path: str, text: str) -> bool:
        try:
            relative = validate_path(path)
        except WorkspaceError:
            return False
        if is_sensitive_path(relative):
            return False
        with self._io():
            target = self._locate(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        return True

    # ---------------------------------------------------------------- reading

    def read_text(self, path: str) -> str | None:
        """The file's current text, or None if it is not a regular file in the workspace."""

        relative = validate_path(path)
        with self._io():
            target = self._locate(relative)
            return self._decode(target.read_bytes()) if target.is_file() else None

    # ---------------------------------------------------------------- writing

    def edit(self, path: str, old_text: str, new_text: str) -> FileOperation:
        """Replace exactly one occurrence of `old_text`; ambiguity is an error, never a guess."""

        relative, target = self._authorize(path, "edit")
        with self._io():
            before = self._existing_text(target)
            old, new = old_text, new_text
            if "\r\n" in before and "\r" not in old:  # the model writes "\n"; match CRLF files
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            if old == new:
                raise WorkspaceError("NO_CHANGE", "new_text is identical to old_text.")
            index = before.find(old)
            if index < 0:
                raise WorkspaceError("OLD_TEXT_NOT_FOUND", "old_text was not found in the file.")
            if before.find(old, index + 1) >= 0:
                raise WorkspaceError(
                    "OLD_TEXT_AMBIGUOUS",
                    "old_text matches more than one place; include more surrounding text to make it unique.",
                )
            after = before[:index] + new + before[index + len(old) :]
            encoded = self._encode(after)
            changed = len(self._encode(old)) + len(self._encode(new))
            self._reserve(changed)
            self._originals.setdefault(relative, before)
            self._replace(target, encoded)
        self._commit(changed)
        return self._operation(relative, "modified", before, after, len(encoded))

    def create(self, path: str, content: str) -> FileOperation:
        relative, target = self._authorize(path, "create")
        with self._io():
            if target.exists() or target.is_symlink():
                raise WorkspaceError("FILE_ALREADY_EXISTS", "A file already exists at that path.")
            encoded = self._encode(content)
            self._reserve(len(encoded))
            self._require_directories(target.parent)
            self._originals.setdefault(relative, None)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "xb") as handle:
                handle.write(encoded)
        self._commit(len(encoded))
        return self._operation(relative, "created", None, content, len(encoded))

    def delete(self, path: str) -> FileOperation:
        """Delete one regular file. Directories are never deleted."""

        relative, target = self._authorize(path, "delete")
        with self._io():
            before = self._existing_text(target)
            size = len(self._encode(before))
            self._reserve(size)
            self._originals.setdefault(relative, before)
            target.unlink()
        self._commit(size)
        return self._operation(relative, "deleted", before, None, 0)

    # ---------------------------------------------------------------- results

    def changes(self) -> list[FileChange]:
        """The net change per file since the workspace was populated (unchanged files omitted)."""

        changes: list[FileChange] = []
        remaining = MAX_TOTAL_DIFF_CHARS
        for relative in sorted(self._originals):
            before = self._originals[relative]
            after = self.read_text(relative)
            if before == after:
                continue
            action: Action = "created" if before is None else "deleted" if after is None else "modified"
            diff, additions, deletions, truncated = _diff(relative, before, after, min(MAX_DIFF_CHARS_PER_FILE, remaining))
            remaining = max(remaining - len(diff), 0)
            changes.append(FileChange(relative, action, additions, deletions, diff, truncated))
        return changes

    # ---------------------------------------------------------------- internals

    def _authorize(self, path: str, action: Literal["edit", "create", "delete"]) -> tuple[str, Path]:
        relative = validate_path(path)
        if is_sensitive_path(relative):  # judged on the path alone: never reveals whether it exists
            raise ForbiddenError(PROTECTED_MESSAGE)
        allowed = {
            "edit": self.scope.modify | self.scope.create,
            "create": self.scope.create,
            "delete": self.scope.delete,
        }[action]
        if relative not in allowed:
            raise WorkspaceError("NOT_IN_APPROVED_PLAN", "That file is not part of the approved plan.")
        return relative, self._locate(relative)

    def _locate(self, relative: str) -> Path:
        """The real location of a validated path, refusing anything that could leave the workspace."""

        current = self.root
        for segment in relative.split("/"):
            current = current / segment
            if current.is_symlink():
                raise WorkspaceError("INVALID_PATH", "The path is not writable.")
        if not current.resolve().is_relative_to(self.root):
            raise WorkspaceError("INVALID_PATH", "The path is not writable.")
        return current

    def _existing_text(self, target: Path) -> str:
        if not target.exists():
            raise WorkspaceError("FILE_NOT_FOUND", "The requested file was not found.")
        if not target.is_file():
            raise WorkspaceError("NOT_A_REGULAR_FILE", "The path is not a regular file.")
        return self._decode(target.read_bytes())

    @staticmethod
    def _decode(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise WorkspaceError("INVALID_CONTENT", "The file is not valid UTF-8 text.") from None

    @staticmethod
    def _encode(text: str) -> bytes:
        if chr(0) in text:
            raise WorkspaceError("INVALID_CONTENT", "Binary content is not allowed.")
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            raise WorkspaceError("INVALID_CONTENT", "The content is not valid UTF-8 text.") from None
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceError("FILE_TOO_LARGE", f"Files are limited to {MAX_FILE_BYTES // 1024} KB.")
        return data

    def _require_directories(self, directory: Path) -> None:
        """Every existing ancestor up to the workspace root must be a real directory."""

        for candidate in (directory, *directory.parents):
            if candidate == self.root:
                return
            if candidate.exists() and not candidate.is_dir():
                raise WorkspaceError("INVALID_PATH", "A parent of that path is a file.")

    def _reserve(self, size: int) -> None:
        if self.write_operations >= self.max_write_operations:
            self.limit_reached = True
            raise WorkspaceError("WRITE_LIMIT_REACHED", f"At most {self.max_write_operations} file changes are allowed per execution.")
        if self.bytes_changed + size > self.max_total_bytes:
            self.limit_reached = True
            raise WorkspaceError("WRITE_LIMIT_REACHED", "The total size of changes allowed for one execution has been reached.")

    def _commit(self, size: int) -> None:
        self.write_operations += 1
        self.bytes_changed += size

    @staticmethod
    def _replace(target: Path, data: bytes) -> None:
        """Write atomically: a crash never leaves a half-written file."""

        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".codefrog-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temporary, target)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise

    @staticmethod
    def _operation(relative: str, action: Action, before: str | None, after: str | None, size: int) -> FileOperation:
        diff, additions, deletions, _ = _diff(relative, before, after, MAX_PREVIEW_CHARS)
        return FileOperation(relative, action, size, additions, deletions, diff)

    @contextmanager
    def _io(self) -> Iterator[None]:
        """Turn raw OS errors into a generic error: no paths or system details reach a caller."""

        try:
            yield
        except OSError as error:
            logger.warning("Workspace I/O failed (exception type=%s)", type(error).__name__)
            raise WorkspaceError("WORKSPACE_ERROR", "The file operation could not be completed.") from None


def _diff(relative: str, before: str | None, after: str | None, limit: int) -> tuple[str, int, int, bool]:
    """A unified diff (secrets redacted, cut on a line boundary) with its line counts."""

    lines = list(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile="/dev/null" if before is None else f"a/{relative}",
            tofile="/dev/null" if after is None else f"b/{relative}",
            n=3,
        )
    )
    additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    text, truncated = truncate_at_line(redact_secrets("".join(lines))[0], max(limit, 0))
    return text, additions, deletions, truncated


_locks: dict[uuid.UUID, threading.Lock] = {}
_locks_guard = threading.Lock()


@contextmanager
def exclusive_workspace(repository_id: uuid.UUID) -> Iterator[None]:
    """Allow one execution per repository at a time in this process."""

    with _locks_guard:
        lock = _locks.setdefault(repository_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise ConflictError("Another execution is already running for this repository.")
    try:
        yield
    finally:
        lock.release()


def get_workspace_root() -> Path:
    """Where workspaces live: WORKSPACE_ROOT, or a folder in the system temp directory."""

    configured = get_settings().workspace_root
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "codefrog-workspaces"
