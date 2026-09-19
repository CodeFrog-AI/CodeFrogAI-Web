"""edit_file, create_file, delete_file: the write tools.

They use the same Tool abstraction as the read tools. They only run when the server has put
an approved `Workspace` in the ToolContext; every path check, size limit, and write happens
in that workspace, never here, and never against GitHub.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ForbiddenError
from app.tools.base import Tool, ToolContext, owned_repository
from app.tools.read_file import MAX_PATH_LENGTH
from app.workspace import FileOperation, Workspace

MAX_TEXT_CHARS = 262_144


class WriteInput(BaseModel):
    """Like ToolInput, but text is kept exactly as given: whitespace in code is significant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    repository_id: uuid.UUID = Field(description="ID of the repository.")
    path: str = Field(
        min_length=1,
        max_length=MAX_PATH_LENGTH,
        description="Repository-relative path using '/'. Must be listed in the approved plan.",
    )


class EditFileInput(WriteInput):
    old_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="Exact text to replace. Must match exactly once.")
    new_text: str = Field(max_length=MAX_TEXT_CHARS, description="Replacement text.")


class CreateFileInput(WriteInput):
    content: str = Field(max_length=MAX_TEXT_CHARS, description="Full UTF-8 text content of the new file.")


class DeleteFileInput(WriteInput):
    pass


class WriteOutput(BaseModel):
    path: str
    action: Literal["created", "modified", "deleted"]
    additions: int
    deletions: int
    bytes_written: int


def _workspace(context: ToolContext, repository_id: uuid.UUID) -> Workspace:
    repository = owned_repository(context, repository_id)
    workspace = context.workspace
    if workspace is None or workspace.repository_id != repository.id:
        raise ForbiddenError("Changes can only be made after the user approves a plan.")
    return workspace


def _output(operation: FileOperation) -> WriteOutput:
    return WriteOutput(
        path=operation.path,
        action=operation.action,
        additions=operation.additions,
        deletions=operation.deletions,
        bytes_written=operation.bytes_written,
    )


def _edit_file(context: ToolContext, arguments: EditFileInput) -> WriteOutput:
    workspace = _workspace(context, arguments.repository_id)
    return _output(workspace.edit(arguments.path, arguments.old_text, arguments.new_text))


def _create_file(context: ToolContext, arguments: CreateFileInput) -> WriteOutput:
    workspace = _workspace(context, arguments.repository_id)
    return _output(workspace.create(arguments.path, arguments.content))


def _delete_file(context: ToolContext, arguments: DeleteFileInput) -> WriteOutput:
    workspace = _workspace(context, arguments.repository_id)
    return _output(workspace.delete(arguments.path))


EDIT_FILE = Tool(
    name="edit_file",
    description=(
        "Replace exactly one occurrence of old_text with new_text in an existing file that the approved "
        "plan lists. Fails if old_text is not found or matches more than once (include more surrounding "
        "lines to make it unique). Read the file first so old_text matches exactly."
    ),
    input_model=EditFileInput,
    handler=_edit_file,
    writes=True,
)

CREATE_FILE = Tool(
    name="create_file",
    description="Create a new UTF-8 text file listed in the approved plan. Fails if the file already exists.",
    input_model=CreateFileInput,
    handler=_create_file,
    writes=True,
)

DELETE_FILE = Tool(
    name="delete_file",
    description="Delete one file that the approved plan lists for deletion. Directories cannot be deleted.",
    input_model=DeleteFileInput,
    handler=_delete_file,
    writes=True,
)
