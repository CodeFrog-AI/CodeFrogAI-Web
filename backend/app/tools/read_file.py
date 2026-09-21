"""read_file: read lines from a file the scanner has indexed (never the disk or GitHub)."""

import re
import uuid

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.context.redaction import is_sensitive_path, redact_secrets
from app.context.service import truncate_at_line
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.db.models import RepositoryChunk, RepositoryFile
from app.tools.base import Tool, ToolContext, ToolInput, owned_repository
from app.workspace import WorkspaceError

MAX_READ_LINES = 300
MAX_READ_CHARS = 16_000
MAX_PATH_LENGTH = 2_048

_DRIVE_PATH = re.compile(r"^[A-Za-z]:")


class ReadFileInput(ToolInput):
    repository_id: uuid.UUID = Field(description="ID of the repository the file belongs to.")
    file_path: str = Field(
        min_length=1,
        max_length=MAX_PATH_LENGTH,
        description="Repository-relative path using '/', exactly as returned by search_code.",
    )
    start_line: int | None = Field(default=None, ge=1, description="First line to read (1-based). Defaults to 1.")
    end_line: int | None = Field(
        default=None,
        ge=1,
        description=f"Last line to read, inclusive. At most {MAX_READ_LINES} lines per call.",
    )

    @field_validator("file_path")
    @classmethod
    def require_a_plain_relative_path(cls, value: str) -> str:
        if chr(0) in value or chr(92) in value:
            raise ValueError("file_path must be a repository-relative path using '/'")
        if value.startswith("/") or _DRIVE_PATH.match(value):
            raise ValueError("file_path must be relative to the repository root")
        if any(segment in ("", ".", "..") for segment in value.split("/")):
            raise ValueError("file_path must not contain empty, '.' or '..' segments")
        return value

    @model_validator(mode="after")
    def check_the_requested_range(self) -> "ReadFileInput":
        if self.end_line is not None:
            start = self.start_line or 1
            if self.end_line < start:
                raise ValueError("end_line must not be before start_line")
            if self.end_line - start + 1 > MAX_READ_LINES:
                raise ValueError(f"at most {MAX_READ_LINES} lines can be read at once")
        return self


class ReadFileOutput(BaseModel):
    """Requested lines of an indexed file. Secrets are redacted without shifting line numbers."""

    repository_id: uuid.UUID
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    total_lines: int
    has_more: bool = Field(description="True when the file has lines after end_line.")
    truncated: bool = Field(description="True when the size cap cut the requested range short.")
    redactions: int
    source: Literal["workspace", "index"] = Field(
        description="workspace: the current local checkout (includes uncommitted edits). index: the last repository scan."
    )
    content: str


def _read_file(context: ToolContext, arguments: ReadFileInput) -> ReadFileOutput:
    repository = owned_repository(context, arguments.repository_id)
    # Judged on the path alone, so the answer never reveals whether such a file exists.
    if is_sensitive_path(arguments.file_path):
        raise ForbiddenError("This file is sensitive and cannot be read")

    file = (
        context.session.query(RepositoryFile)
        .filter(RepositoryFile.repository_id == repository.id, RepositoryFile.path == arguments.file_path)
        .first()
    )
    view = context.workspace or context.checkout
    if view is not None:
        # A local checkout exists: it is the source of truth, and includes earlier edits.
        try:
            text = view.read_text(arguments.file_path)
        except WorkspaceError:
            raise NotFoundError("File not found in the indexed repository") from None
        if text is None:
            raise NotFoundError("File not found in the indexed repository")
    else:
        if file is None:
            raise NotFoundError("File not found in the indexed repository")
        # Only the text column: chunk rows also carry large embedding vectors we must not load.
        chunks = (
            context.session.query(RepositoryChunk.content)
            .filter(RepositoryChunk.repository_file_id == file.id)
            .order_by(RepositoryChunk.chunk_index)
        )
        text = "".join(content for (content,) in chunks)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total_lines = len(lines)

    start = arguments.start_line or 1
    if start > max(total_lines, 1):
        raise BadRequestError(f"start_line is beyond the end of the file ({total_lines} lines)")
    end = min(arguments.end_line or start + MAX_READ_LINES - 1, total_lines)

    text, cut = truncate_at_line("\n".join(lines[start - 1 : end]), MAX_READ_CHARS)
    text, redactions = redact_secrets(text)
    text, recut = truncate_at_line(text, MAX_READ_CHARS)
    cut = cut or recut
    kept = text.count("\n") + 1 if text else 0

    return ReadFileOutput(
        repository_id=repository.id,
        file_path=arguments.file_path,
        language=file.language if file else None,
        start_line=start,
        end_line=start + kept - 1,
        total_lines=total_lines,
        has_more=start + kept - 1 < total_lines,
        truncated=cut,
        redactions=redactions,
        source="workspace" if view is not None else "index",
        content=text,
    )


READ_FILE = Tool(
    name="read_file",
    description=(
        "Read lines from a file of this repository. Reads the repository's local checkout when one exists "
        "(current content, including earlier edits) and otherwise the last scanned copy; the result's "
        f"source field says which. It never contacts GitHub. Returns up to {MAX_READ_LINES} lines per call with the line range, "
        "total line count, and whether more lines follow. Sensitive files (env files, keys, "
        "credentials) cannot be read and secrets in other files are redacted."
    ),
    input_model=ReadFileInput,
    handler=_read_file,
)
